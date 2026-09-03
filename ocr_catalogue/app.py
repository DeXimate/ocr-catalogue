from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
import threading
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import storage
from .engines import extract
from .exporter import export_csv, export_xlsx


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
_processing_jobs: set[str] = set()
_processing_lock = threading.RLock()


def export_download_name(uploaded_filename: str, fmt: str) -> str:
    """Build a readable, Windows-safe download name from the upload name."""
    stem = Path(uploaded_filename or "").stem.strip()
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if not stem:
        stem = "catalogue"
    return f"produits-{stem[:140]}.{fmt}"


def process_job(job_id: str, source: Path) -> None:
    with _processing_lock:
        if job_id in _processing_jobs:
            return
        _processing_jobs.add(job_id)
    try:
        job = storage.load_job(job_id)
        job["status"] = "Traitement"
        storage.save_job(job_id, job)

        def progress(done: int, total: int) -> None:
            current = storage.load_job(job_id)
            current["progress"] = round(done / total * 100)
            storage.save_job(job_id, current)

        products = extract(source, storage.job_folder(job_id), progress)
        job = storage.load_job(job_id)
        job.update(status="Terminé", progress=100, products=[p.to_dict() for p in products], error="")
    except Exception as exc:
        try:
            job = storage.load_job(job_id)
            job.update(status="Erreur", error=str(exc))
            storage.save_job(job_id, job)
        except FileNotFoundError:
            pass
    else:
        storage.save_job(job_id, job)
    finally:
        with _processing_lock:
            _processing_jobs.discard(job_id)


def resume_incomplete_jobs() -> None:
    """Resume uploads whose worker was interrupted by an application restart."""
    for summary in storage.list_jobs():
        if summary.get("status") not in {"Importé", "Traitement"}:
            continue
        job_id = summary["id"]
        folder = storage.job_folder(job_id)
        source = next((path for path in folder.glob("source.*") if path.is_file()), None)
        if source is None:
            job = storage.load_job(job_id)
            job.update(status="Erreur", error="Le fichier source importé est introuvable")
            storage.save_job(job_id, job)
            continue
        threading.Thread(target=process_job, args=(job_id, source), daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "OCRCatalogue/0.1"

    def _json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, download_name: str | None = None):
        if not path.exists() or not path.is_file():
            return self.send_error(404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/api/jobs":
            return self._json(storage.list_jobs())
        if len(parts) == 3 and parts[:2] == ["api", "deletions"]:
            try:
                return self._json(storage.deletion_status(parts[2]))
            except FileNotFoundError:
                return self._json({"error": "Suppression introuvable"}, 404)
            except ValueError:
                return self._json({"error": "Identifiant invalide"}, 400)
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            try:
                return self._json(storage.load_job(parts[2]))
            except FileNotFoundError:
                return self._json({"error": "Introuvable"}, 404)
        if len(parts) >= 4 and parts[:2] == ["api", "assets"]:
            job_id = parts[2]
            rel = Path(*parts[3:])
            target = (storage.job_folder(job_id) / rel).resolve()
            if storage.job_folder(job_id) not in target.parents:
                return self.send_error(403)
            return self._file(target)
        if parsed.path == "/":
            return self._file(STATIC / "index.html")
        target = (STATIC / parsed.path.lstrip("/")).resolve()
        if STATIC.resolve() in target.parents:
            return self._file(target)
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/api/shutdown":
            # The server is bound to 127.0.0.1, so this control endpoint is
            # available only from the same Windows session. Shutdown must run
            # outside the request thread to let the response finish cleanly.
            self._json({"status": "stopping"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if parsed.path == "/api/import":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                return self._json({"error": "Fichier requis"}, 400)
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            message = BytesParser(policy=default).parsebytes((f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode() + raw)
            part = next((p for p in message.iter_parts() if p.get_filename()), None)
            if not part:
                return self._json({"error": "Fichier requis"}, 400)
            filename = Path(part.get_filename()).name
            suffix = Path(filename).suffix.lower()
            if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
                return self._json({"error": "Format non accepté"}, 400)
            job_id, folder = storage.new_job(filename)
            source = folder / ("source" + suffix)
            source.write_bytes(part.get_payload(decode=True))
            threading.Thread(target=process_job, args=(job_id, source), daemon=True).start()
            return self._json({"id": job_id}, 201)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "reprocess":
            job_id = parts[2]
            try:
                with _processing_lock:
                    if job_id in _processing_jobs:
                        raise RuntimeError("Le catalogue est déjà en cours de traitement")
                source = storage.prepare_reprocessing(job_id)
                threading.Thread(target=process_job, args=(job_id, source), daemon=True).start()
                return self._json({"id": job_id, "status": "Importé", "progress": 0}, 202)
            except FileNotFoundError as exc:
                return self._json({"error": str(exc) or "Catalogue introuvable"}, 404)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            except (ValueError, OSError) as exc:
                return self._json({"error": str(exc)}, 400)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "export":
            job_id = parts[2]
            options = self._body()
            job = storage.load_job(job_id)
            fmt = "csv" if options.get("format") == "csv" else "xlsx"
            include = bool(options.get("include_photos"))
            scope = options.get("scope", "all")
            target = storage.job_folder(job_id) / f"export.{fmt}"
            if fmt == "csv":
                export_csv(job["products"], storage.job_folder(job_id), target, include, scope)
            else:
                export_xlsx(job["products"], storage.job_folder(job_id), target, include, scope)
            return self._json({
                "url": f"/api/assets/{job_id}/{target.name}",
                "filename": export_download_name(job.get("filename", ""), fmt),
            })
        self.send_error(404)

    def do_PUT(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "products":
            payload = self._body()
            job = storage.update_products(parts[2], payload.get("products", []))
            return self._json(job)
        self.send_error(404)

    def do_DELETE(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            try:
                return self._json(storage.delete_job(parts[2]))
            except FileNotFoundError:
                return self._json({"error": "Catalogue introuvable"}, 404)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            except ValueError:
                return self._json({"error": "Identifiant invalide"}, 400)
            except OSError as exc:
                return self._json({"error": str(exc)}, 423)
        self.send_error(404)


def main() -> None:
    storage.ensure_dirs()
    host, port = "127.0.0.1", 8765
    print(f"OCR Catalogue disponible sur http://{host}:{port}")
    with ThreadingHTTPServer((host, port), Handler) as server:
        resume_incomplete_jobs()
        server.serve_forever()


if __name__ == "__main__":
    main()
