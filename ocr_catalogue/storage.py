from __future__ import annotations

import json
import re
import shutil
import stat
import threading
import time
import uuid
from pathlib import Path

from .models import Product


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
JOBS = DATA / "jobs"
_lock = threading.RLock()
_cleanup_started = False
_deletions: dict[str, dict] = {}


class JobCancelled(RuntimeError):
    pass


def _cancel_marker(job_id: str) -> Path:
    return job_folder(job_id) / ".cancel-requested"


def cancel_requested(job_id: str) -> bool:
    try:
        return _cancel_marker(job_id).is_file()
    except ValueError:
        return False


def request_cancel(job_id: str) -> dict:
    folder = job_folder(job_id)
    with _lock:
        metadata = folder / "job.json"
        if not metadata.is_file():
            raise FileNotFoundError(job_id)
        job = json.loads(metadata.read_text(encoding="utf-8"))
        status = job.get("status")
        if status == "Annulation":
            return {"id": job_id, "status": "Annulation", "progress": job.get("progress", 0)}
        if status not in {"Importé", "Traitement"}:
            raise RuntimeError("Ce catalogue n'est pas en cours de traitement")
        _cancel_marker(job_id).write_text("cancel", encoding="ascii")
        job["status"] = "Annulation"
        job["error"] = ""
        save_job(job_id, job)
        return {"id": job_id, "status": "Annulation", "progress": job.get("progress", 0)}


def finalize_cancelled_job(job_id: str) -> dict:
    folder = job_folder(job_id)
    with _lock:
        job = load_job(job_id)
        source = next((path for path in folder.glob("source.*") if path.is_file()), None)
        if source is None:
            raise FileNotFoundError("Le fichier source importé est introuvable")

        marker = _cancel_marker(job_id)
        trash = folder / f".cancel-trash-{uuid.uuid4().hex}"
        trash.mkdir()
        preserve = {source.resolve(), (folder / "job.json").resolve(), marker.resolve(), trash.resolve()}

        for child in list(folder.iterdir()):
            if child.resolve() in preserve:
                continue
            child.replace(trash / child.name)

        for name in ("pages", "crops", "products"):
            (folder / name).mkdir(exist_ok=True)

        marker.unlink(missing_ok=True)
        job.update(
            status="Annulé",
            progress=0,
            products=[],
            error="",
            asset_version=uuid.uuid4().hex,
        )
        save_job(job_id, job)
        threading.Thread(target=_remove_stale_tree, args=(trash,), daemon=True).start()
        return {"id": job_id, "filename": job.get("filename", ""), "status": "Annulé", "progress": 0}


def ensure_dirs() -> None:
    global _cleanup_started
    JOBS.mkdir(parents=True, exist_ok=True)
    with _lock:
        if not _cleanup_started:
            _cleanup_started = True
            for marker in JOBS.glob("*/.delete-requested"):
                deletion_id = marker.read_text(encoding="ascii").strip()
                _deletions[deletion_id] = {"id": deletion_id, "status": "deleting"}
                threading.Thread(target=_remove_tree_eventually, args=(marker.parent, deletion_id), daemon=True).start()
            # A restart may have interrupted the final cleanup of an earlier
            # reprocessing. These folders are already detached from the live
            # job, so they can safely be removed in the background.
            for stale in list(JOBS.glob("*/.reprocess-trash-*")) + list(JOBS.glob("*/.cancel-trash-*")):
                threading.Thread(target=_remove_stale_tree, args=(stale,), daemon=True).start()


def _remove_stale_tree(folder: Path) -> None:
    def remove_readonly(function, path, _error_info):
        Path(path).chmod(stat.S_IWRITE)
        function(path)

    for attempt in range(120):
        try:
            shutil.rmtree(folder, onerror=remove_readonly)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(min(3.0, .25 + attempt * .08))


def _remove_tree_eventually(folder: Path, deletion_id: str) -> None:
    """Retry a user-confirmed deletion without holding an HTTP connection."""
    def remove_readonly(function, path, _error_info):
        Path(path).chmod(stat.S_IWRITE)
        function(path)

    for attempt in range(120):
        if not folder.exists():
            with _lock:
                _deletions[deletion_id] = {"id": deletion_id, "status": "deleted"}
            return
        try:
            shutil.rmtree(folder, onerror=remove_readonly)
            with _lock:
                _deletions[deletion_id] = {"id": deletion_id, "status": "deleted"}
            return
        except FileNotFoundError:
            with _lock:
                _deletions[deletion_id] = {"id": deletion_id, "status": "deleted"}
            return
        except OSError as exc:
            last_error = str(exc)
            time.sleep(min(3.0, .25 + attempt * .08))
    with _lock:
        _deletions[deletion_id] = {
            "id": deletion_id,
            "status": "error",
            "error": f"Windows n'a pas pu supprimer tous les fichiers : {last_error}",
        }


def new_job(filename: str) -> tuple[str, Path]:
    ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS / job_id
    (folder / "pages").mkdir(parents=True)
    (folder / "crops").mkdir()
    (folder / "products").mkdir()
    save_job(job_id, {"id": job_id, "filename": filename, "status": "Importé", "progress": 0, "products": [], "asset_version": uuid.uuid4().hex})
    return job_id, folder


def job_folder(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id or ""):
        raise ValueError("Identifiant invalide")
    path = (JOBS / job_id).resolve()
    if JOBS.resolve() not in path.parents:
        raise ValueError("Identifiant invalide")
    return path


def _replace_job_file(temp: Path, target: Path, attempts: int = 24) -> None:
    """Atomically replace job metadata despite short OneDrive/AV locks."""
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            temp.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            # Windows sharing violations can surface either as PermissionError
            # or as a plain OSError with winerror 5/32/33.
            if getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            last_error = exc
        time.sleep(min(.4, .025 * (attempt + 1)))
    assert last_error is not None
    raise last_error


def save_job(job_id: str, payload: dict) -> None:
    ensure_dirs()
    target = job_folder(job_id) / "job.json"
    # A unique name prevents a synchronisation client from confusing two
    # successive temporary generations of job.json.
    temp = target.with_name(f"job-{uuid.uuid4().hex}.tmp")
    with _lock:
        try:
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _replace_job_file(temp, target)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                # OneDrive may still hold the abandoned temporary file; it is
                # harmless and must not turn a successful extraction into an error.
                pass


def load_job(job_id: str) -> dict:
    with _lock:
        job = json.loads((job_folder(job_id) / "job.json").read_text(encoding="utf-8"))
        if "products" in job:
            job["products"] = [Product.from_dict(value).to_dict() for value in job.get("products", [])]
        return job


def list_jobs() -> list[dict]:
    ensure_dirs()
    jobs = []
    for path in JOBS.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            jobs.append({key: job.get(key) for key in ("id", "filename", "status", "progress")})
        except (OSError, json.JSONDecodeError):
            continue
    return jobs


def update_products(job_id: str, values: list[dict]) -> dict:
    job = load_job(job_id)
    job["products"] = [Product.from_dict(value).to_dict() for value in values]
    save_job(job_id, job)
    return job


def prepare_reprocessing(job_id: str) -> Path:
    """Discard every derived artifact while preserving the imported source."""
    folder = job_folder(job_id)
    with _lock:
        job = load_job(job_id)
        if job.get("status") in {"Importé", "Traitement", "Annulation"}:
            raise RuntimeError("Le catalogue est déjà en cours de traitement")
        _cancel_marker(job_id).unlink(missing_ok=True)
        source = next((path for path in folder.glob("source.*") if path.is_file()), None)
        if source is None:
            raise FileNotFoundError("Le fichier source importé est introuvable")

        # Detach all previous outputs atomically from their public paths first.
        # This prevents the extraction or the browser from ever seeing a mix of
        # old and new images. Physical removal then continues in the background.
        trash = folder / f".reprocess-trash-{uuid.uuid4().hex}"
        trash.mkdir()
        for child in list(folder.iterdir()):
            if child in {source, folder / "job.json", trash}:
                continue
            child.replace(trash / child.name)

        for name in ("pages", "crops", "products"):
            (folder / name).mkdir()
        job.update(
            status="Importé",
            progress=0,
            products=[],
            error="",
            asset_version=uuid.uuid4().hex,
        )
        save_job(job_id, job)
        threading.Thread(target=_remove_stale_tree, args=(trash,), daemon=True).start()
        return source


def delete_job(job_id: str) -> dict:
    """Permanently remove one completed upload and every derived artifact."""
    folder = job_folder(job_id)
    with _lock:
        metadata = folder / "job.json"
        if not metadata.is_file():
            raise FileNotFoundError(job_id)
        job = json.loads(metadata.read_text(encoding="utf-8"))
        if job.get("status") in {"Importé", "Traitement", "Annulation"}:
            raise RuntimeError("Le catalogue est encore en cours de traitement")
        filename = job.get("filename", "")
        # Hide the catalogue atomically from list_jobs before deleting large
        # image trees. This makes the UI immediate and prevents new thumbnail
        # requests from racing with Windows/OneDrive file removal.
        (folder / ".delete-requested").write_text(job_id, encoding="ascii")
        metadata.replace(folder / "job.deleting.json")
        _deletions[job_id] = {"id": job_id, "status": "deleting"}
        threading.Thread(target=_remove_tree_eventually, args=(folder, job_id), daemon=True).start()
    return {"id": job_id, "filename": filename, "status": "deleting"}


def deletion_status(job_id: str) -> dict:
    job_folder(job_id)  # validates without broadening the filesystem target
    with _lock:
        status = _deletions.get(job_id)
        if status:
            return dict(status)
    raise FileNotFoundError(job_id)


def copy_upload(job_id: str, source: Path, suffix: str) -> Path:
    destination = job_folder(job_id) / ("source" + suffix.lower())
    shutil.copyfile(source, destination)
    return destination
