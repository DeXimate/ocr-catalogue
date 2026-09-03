from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / "ocr_catalogue" / "app.py"
STORAGE = ROOT / "ocr_catalogue" / "storage.py"
ENGINES = ROOT / "ocr_catalogue" / "engines.py"
PIPELINE = ROOT / "ocr_catalogue" / "pipeline.py"
INDEX = ROOT / "static" / "index.html"
APPJS = ROOT / "static" / "app.js"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"{label}: motif introuvable")
    return text.replace(old, new, 1)


def replace_function(text: str, name: str, next_name: str, replacement: str) -> str:
    start_marker = f"def {name}("
    end_marker = f"\ndef {next_name}("
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"fonction {name} introuvable")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"fonction {next_name} introuvable après {name}")
    return text[:start] + replacement.rstrip() + "\n" + text[end:]


PROCESS_JOB = r'''def process_job(job_id: str, source: Path) -> None:
    with _processing_lock:
        if job_id in _processing_jobs:
            return
        _processing_jobs.add(job_id)

    def check_cancelled() -> None:
        if storage.cancel_requested(job_id):
            raise storage.JobCancelled("Traitement annulé par l'utilisateur")

    try:
        check_cancelled()
        job = storage.load_job(job_id)
        job["status"] = "Traitement"
        storage.save_job(job_id, job)

        def progress(done: int, total: int) -> None:
            check_cancelled()
            current = storage.load_job(job_id)
            current["progress"] = round(done / max(1, total) * 100)
            storage.save_job(job_id, current)

        products = extract(
            source,
            storage.job_folder(job_id),
            progress,
            cancel=check_cancelled,
        )
        check_cancelled()

        job = storage.load_job(job_id)
        job.update(
            status="Terminé",
            progress=100,
            products=[p.to_dict() for p in products],
            error="",
        )
    except storage.JobCancelled:
        try:
            storage.finalize_cancelled_job(job_id)
        except Exception as exc:
            try:
                job = storage.load_job(job_id)
                job.update(status="Erreur", error=f"Annulation incomplète : {exc}")
                storage.save_job(job_id, job)
            except FileNotFoundError:
                pass
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
'''


RESUME = r'''def resume_incomplete_jobs() -> None:
    # Resume interrupted work, but honor an earlier cancellation request.
    for summary in storage.list_jobs():
        job_id = summary["id"]
        status = summary.get("status")

        if status == "Annulation" or storage.cancel_requested(job_id):
            try:
                storage.finalize_cancelled_job(job_id)
            except FileNotFoundError:
                pass
            continue

        if status not in {"Importé", "Traitement"}:
            continue

        folder = storage.job_folder(job_id)
        source = next((path for path in folder.glob("source.*") if path.is_file()), None)
        if source is None:
            job = storage.load_job(job_id)
            job.update(status="Erreur", error="Le fichier source importé est introuvable")
            storage.save_job(job_id, job)
            continue

        threading.Thread(target=process_job, args=(job_id, source), daemon=True).start()
'''


STORAGE_CANCEL = r'''
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

'''


RENDER_PDF = r'''def render_pdf(
    source: Path,
    pages_dir: Path,
    dpi: int = 150,
    cancel=None,
) -> list[Path]:
    tool = _poppler_binary("pdftoppm")
    if not tool:
        raise RuntimeError("Poppler/pdftoppm est requis pour rendre les pages PDF")

    prefix = pages_dir / "page"
    command = [str(tool), "-jpeg", "-r", str(dpi), str(source), str(prefix)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while process.poll() is None:
            if cancel:
                cancel()
            time.sleep(.12)
        stdout, stderr = process.communicate()
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        raise

    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    if cancel:
        cancel()
    return sorted(pages_dir.glob("page-*.jpg"))
'''


def patch_storage() -> None:
    text = STORAGE.read_text(encoding="utf-8")
    if "class JobCancelled(" not in text:
        marker = "_deletions: dict[str, dict] = {}\n\n"
        if marker not in text:
            raise RuntimeError("storage globals introuvables")
        text = text.replace(marker, marker + STORAGE_CANCEL, 1)

    text = replace_once(
        text,
        '            for stale in JOBS.glob("*/.reprocess-trash-*"):\n                threading.Thread(target=_remove_stale_tree, args=(stale,), daemon=True).start()\n',
        '            for stale in list(JOBS.glob("*/.reprocess-trash-*")) + list(JOBS.glob("*/.cancel-trash-*")):\n                threading.Thread(target=_remove_stale_tree, args=(stale,), daemon=True).start()\n',
        "cleanup cancel-trash",
    )
    text = replace_once(
        text,
        '        if job.get("status") in {"Importé", "Traitement"}:\n            raise RuntimeError("Le catalogue est déjà en cours de traitement")\n',
        '        if job.get("status") in {"Importé", "Traitement", "Annulation"}:\n            raise RuntimeError("Le catalogue est déjà en cours de traitement")\n        _cancel_marker(job_id).unlink(missing_ok=True)\n',
        "prepare_reprocessing status",
    )
    text = replace_once(
        text,
        '        if job.get("status") in {"Importé", "Traitement"}:\n            raise RuntimeError("Le catalogue est encore en cours de traitement")\n',
        '        if job.get("status") in {"Importé", "Traitement", "Annulation"}:\n            raise RuntimeError("Le catalogue est encore en cours de traitement")\n',
        "delete_job status",
    )
    compile(text, str(STORAGE), "exec")
    STORAGE.write_text(text, encoding="utf-8")


def patch_app() -> None:
    text = APP.read_text(encoding="utf-8")
    text = replace_function(text, "process_job", "resume_incomplete_jobs", PROCESS_JOB)
    text = replace_function(text, "resume_incomplete_jobs", "Handler", RESUME)

    endpoint = '''        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
            job_id = parts[2]
            try:
                return self._json(storage.request_cancel(job_id), 202)
            except FileNotFoundError:
                return self._json({"error": "Catalogue introuvable"}, 404)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            except ValueError:
                return self._json({"error": "Identifiant invalide"}, 400)
'''
    marker = '        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "reprocess":\n'
    if 'parts[3] == "cancel"' not in text:
        if marker not in text:
            raise RuntimeError("endpoint reprocess introuvable")
        text = text.replace(marker, endpoint + marker, 1)

    compile(text, str(APP), "exec")
    APP.write_text(text, encoding="utf-8")


def patch_engines() -> None:
    text = ENGINES.read_text(encoding="utf-8")
    if "import time\n" not in text:
        text = replace_once(text, "import subprocess\n", "import subprocess\nimport time\n", "engines import time")
    text = replace_function(text, "render_pdf", "_merge_price_tokens", RENDER_PDF)

    text = replace_once(
        text,
        '''def extract_pdf(source: Path, folder: Path, progress=None) -> list[Product]:
    pages = render_pdf(source, folder / "pages")
    from .pipeline import extract_offers
    return extract_offers(source, folder, pages, progress)
''',
        '''def extract_pdf(source: Path, folder: Path, progress=None, cancel=None) -> list[Product]:
    pages = render_pdf(source, folder / "pages", cancel=cancel)
    if cancel:
        cancel()
    from .pipeline import extract_offers
    return extract_offers(source, folder, pages, progress, cancel=cancel)
''',
        "extract_pdf cancellation",
    )
    text = replace_once(
        text,
        '''def extract(source: Path, folder: Path, progress=None) -> list[Product]:
    if source.suffix.lower() == ".pdf":
        return extract_pdf(source, folder, progress)
    return import_image(source, folder)
''',
        '''def extract(source: Path, folder: Path, progress=None, cancel=None) -> list[Product]:
    if cancel:
        cancel()
    if source.suffix.lower() == ".pdf":
        return extract_pdf(source, folder, progress, cancel=cancel)
    result = import_image(source, folder)
    if cancel:
        cancel()
    return result
''',
        "extract cancellation",
    )
    compile(text, str(ENGINES), "exec")
    ENGINES.write_text(text, encoding="utf-8")


def patch_pipeline() -> None:
    text = PIPELINE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'def extract_offers(source: Path, folder: Path, raster_pages: list[Path], progress=None) -> list[Product]:\n',
        'def extract_offers(source: Path, folder: Path, raster_pages: list[Path], progress=None, cancel=None) -> list[Product]:\n    if cancel:\n        cancel()\n',
        "pipeline signature",
    )
    text = replace_once(
        text,
        "    document = extract_document_scene(source, raster_pages)\n    classify_document(document)\n    infer_catalogue_style(document)\n    offers = resolve_document_offers(document)\n",
        "    document = extract_document_scene(source, raster_pages)\n    if cancel:\n        cancel()\n    classify_document(document)\n    if cancel:\n        cancel()\n    infer_catalogue_style(document)\n    if cancel:\n        cancel()\n    offers = resolve_document_offers(document)\n    if cancel:\n        cancel()\n",
        "pipeline checkpoints",
    )
    text = replace_once(
        text,
        "    for index, offer in enumerate(offers):\n        crop_rel, product_rel = _crop_offer(offer, raster_pages[offer.page - 1], page_sizes[offer.page], folder)\n",
        "    for index, offer in enumerate(offers):\n        if cancel:\n            cancel()\n        crop_rel, product_rel = _crop_offer(offer, raster_pages[offer.page - 1], page_sizes[offer.page], folder)\n",
        "pipeline offer checkpoint",
    )
    text = replace_once(
        text,
        "    diagnostic_images = _write_diagnostic_contact_sheets(products, folder)\n",
        "    if cancel:\n        cancel()\n    diagnostic_images = _write_diagnostic_contact_sheets(products, folder)\n    if cancel:\n        cancel()\n",
        "pipeline diagnostics checkpoint",
    )
    compile(text, str(PIPELINE), "exec")
    PIPELINE.write_text(text, encoding="utf-8")


def patch_index() -> None:
    text = INDEX.read_text(encoding="utf-8")
    if 'id="cancelProcessing"' not in text:
        marker = '        <button id="reprocess" class="button button-secondary"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>Retraiter</button>\n'
        button = '        <button id="cancelProcessing" class="button button-danger hidden"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7h10v10H7z"/></svg>Annuler le traitement</button>\n'
        if marker not in text:
            raise RuntimeError("bouton Retraiter introuvable")
        text = text.replace(marker, marker + button, 1)
    INDEX.write_text(text, encoding="utf-8")


def patch_appjs() -> None:
    text = APPJS.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  if(['Importé','Traitement'].includes(state.job.status)){\n",
        "  if(['Importé','Traitement','Annulation'].includes(state.job.status)){\n",
        "poll Annulation",
    )
    text = replace_once(
        text,
        "  if(!state.job||state.job.status==='Terminé'){element.classList.add('hidden');return}\n",
        "  if(!state.job||!['Importé','Traitement','Annulation'].includes(state.job.status)){element.classList.add('hidden');return}\n",
        "progress active statuses",
    )
    text = replace_once(
        text,
        "  $('#save').disabled=!hasJob;\n  $('#reprocess').disabled=!hasJob||['Importé','Traitement'].includes(state.job?.status);\n  $('#delete').disabled=!hasJob||['Importé','Traitement'].includes(state.job?.status);\n  $('#export').disabled=!hasJob;\n",
        "  const processing=Boolean(hasJob&&['Importé','Traitement','Annulation'].includes(state.job?.status));\n  $('#save').disabled=!hasJob;\n  $('#reprocess').disabled=!hasJob||processing;\n  $('#delete').disabled=!hasJob||processing;\n  $('#export').disabled=!hasJob;\n  $('#cancelProcessing').classList.toggle('hidden',!processing);\n  $('#cancelProcessing').disabled=!processing||state.job?.status==='Annulation';\n",
        "render cancel button",
    )

    if "$('#cancelProcessing').onclick" not in text:
        handler = r'''
$('#cancelProcessing').onclick=async()=>{
  if(!state.job||!['Importé','Traitement'].includes(state.job.status))return;
  const filename=state.job.filename||'ce catalogue';
  const confirmed=window.confirm(
    `Annuler le traitement de "${filename}" ?\n\nLe PDF importé sera conservé. Les résultats partiels, images et crops créés pendant ce traitement seront supprimés.`
  );
  if(!confirmed)return;
  const id=state.job.id;
  const button=$('#cancelProcessing');
  button.disabled=true;
  button.textContent='Annulation…';
  try{
    await api(`/api/jobs/${id}/cancel`,{method:'POST'});
    toast('Annulation demandée — arrêt du traitement en cours');
    await loadJob(id);
  }catch(error){
    toast(`Annulation impossible : ${error.message}`);
  }finally{
    button.textContent='Annuler le traitement';
  }
};
'''
        marker = "$('#reprocess').onclick=()=>{\n"
        if marker not in text:
            raise RuntimeError("handler Retraiter introuvable")
        text = text.replace(marker, handler + marker, 1)

    APPJS.write_text(text, encoding="utf-8")


def main() -> None:
    for path in (APP, STORAGE, ENGINES, PIPELINE, INDEX, APPJS):
        if not path.exists():
            raise RuntimeError(f"fichier introuvable: {path}")

    patch_storage()
    patch_app()
    patch_engines()
    patch_pipeline()
    patch_index()
    patch_appjs()

    print("CANCEL CATALOGUE PROCESSING FIX APPLIQUE")
    print(" - bouton Annuler le traitement pendant Importé / Traitement")
    print(" - arrêt coopératif du worker et de pdftoppm")
    print(" - PDF source conservé")
    print(" - résultats partiels supprimés")
    print(" - statut final Annulé")
    print(" - catalogue Annulé peut ensuite être Retraité ou Supprimé")
    print()
    print(r"Etape suivante: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise
