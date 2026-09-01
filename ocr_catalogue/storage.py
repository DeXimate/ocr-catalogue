from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

from .models import Product


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
JOBS = DATA / "jobs"
_lock = threading.RLock()


def ensure_dirs() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)


def new_job(filename: str) -> tuple[str, Path]:
    ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS / job_id
    (folder / "pages").mkdir(parents=True)
    (folder / "crops").mkdir()
    (folder / "products").mkdir()
    save_job(job_id, {"id": job_id, "filename": filename, "status": "Importé", "progress": 0, "products": []})
    return job_id, folder


def job_folder(job_id: str) -> Path:
    path = (JOBS / job_id).resolve()
    if JOBS.resolve() not in path.parents:
        raise ValueError("Identifiant invalide")
    return path


def save_job(job_id: str, payload: dict) -> None:
    ensure_dirs()
    target = job_folder(job_id) / "job.json"
    temp = target.with_suffix(".tmp")
    with _lock:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)


def load_job(job_id: str) -> dict:
    with _lock:
        return json.loads((job_folder(job_id) / "job.json").read_text(encoding="utf-8"))


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


def copy_upload(job_id: str, source: Path, suffix: str) -> Path:
    destination = job_folder(job_id) / ("source" + suffix.lower())
    shutil.copyfile(source, destination)
    return destination

