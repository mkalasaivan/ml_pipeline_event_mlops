"""FastAPI service: prediction, bulk upload (stored in SQLite), retraining, uptime and dashboard data."""
import asyncio
import io
import json
import os
import socket
import time
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import database as db
from src.config import EDA_PATH, MODEL_PATH, MODELS_DIR, UPLOAD_DIR, class_names, load_meta
from src.model import get_model, model_version, start_retraining
from src.prediction import predict_bytes
from src.preprocessing import InvalidImage, image_features, is_image_name, validate_image_bytes

STARTED = time.time()
# Cap concurrent TensorFlow inferences per container: unbounded threads exhaust the memory limit.
INFER_SLOTS = asyncio.Semaphore(int(os.getenv("INFER_CONCURRENCY", "2")))
HOST = socket.gethostname()           # identifies the container when scaled behind nginx
STATIC = Path(__file__).parent / "static"

# Folder names used by the external "Soccer Events TestSet", mapped to our classes,
# so its zip can be uploaded as-is.
ALIASES = {"corner": "Corner_Kick", "free-kick": "Free_Kick", "penalty": "Penalty_kick",
           "red-cards": "Red_Card", "yellow-cards": "Yellow_Card", "tackle": "tackle",
           "to-subtitue": "substitute", "to-substitue": "substitute", "substitution": "substitute"}

app = FastAPI(title="Football Event Recognition API", version="1.0",
              description="Classifies football broadcast images into 7 match events and retrains on new data.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    db.init_db()
    db.fail_stale_runs()
    if MODEL_PATH.exists() and os.getenv("PRELOAD_MODEL", "1") == "1":
        get_model()


def resolve_label(name: str) -> Optional[str]:
    classes = class_names()
    lower = {c.lower(): c for c in classes}
    key = name.strip().lower()
    return lower.get(key) or ALIASES.get(key)


# ------------------------------------------------------------------ status
@app.get("/health")
def health():
    up = time.time() - STARTED
    return {"status": "ok" if MODEL_PATH.exists() else "no-model",
            "uptime_seconds": round(up, 1), "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(STARTED)),
            "container": HOST, "model_version": model_version(),
            "model_updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(MODEL_PATH.stat().st_mtime))
            if MODEL_PATH.exists() else None}


@app.get("/classes")
def classes():
    return class_names()


@app.get("/stats")
def stats():
    """Everything the dashboard needs: model metrics, EDA features, uploads, predictions, retraining runs."""
    meta = load_meta()
    eda = json.loads(EDA_PATH.read_text()) if EDA_PATH.exists() else []
    return {"model": {k: meta.get(k) for k in ("version", "trained_at", "retrained_at", "tensorflow_version",
                                                "n_train", "n_val", "n_test", "test_metrics", "external_metrics",
                                                "last_retrain", "history", "class_counts")},
            "eda": eda, "uploads": db.upload_counts(), "predictions": db.prediction_summary(),
            "runs": db.runs(10), "container": HOST}


@app.get("/figures/{name}")
def figure(name: str):
    p = (MODELS_DIR / "figures" / name).resolve()
    if p.parent != (MODELS_DIR / "figures").resolve() or not p.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(p)


# ------------------------------------------------------------------ prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        validate_image_bytes(data)
        async with INFER_SLOTS:                     # bounded, and off the event loop
            result = await run_in_threadpool(predict_bytes, data)
    except InvalidImage as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    result["features"] = await run_in_threadpool(image_features, data)
    result["container"] = HOST
    db.log_prediction(file.filename, result["predicted_class"], result["confidence"],
                      result["latency_ms"], result["model_version"])
    return result


# ------------------------------------------------------------------ upload
def _store(data: bytes, filename: str, label: str, source: str) -> bool:
    try:
        validate_image_bytes(data)
    except InvalidImage:
        return False
    folder = UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex[:12]}{Path(filename).suffix.lower()}"
    path.write_bytes(data)
    db.add_upload(Path(filename).name, path, label, source)
    return True


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), label: Optional[str] = Form(None)):
    """Bulk-upload labelled images for retraining.

    - Images + a `label` form field: every image gets that label.
    - A .zip with one folder per class (e.g. Corner_Kick/, Red_Card/ or the test-set names Corner/, Red-Cards/):
      labels come from the folder names.
    """
    fixed = resolve_label(label) if label else None
    if label and not fixed:
        raise HTTPException(400, f"unknown label '{label}'. Use one of {class_names()}")
    saved, skipped, per_class = 0, [], {}

    def add(data, name, lab, src):
        nonlocal saved
        if _store(data, name, lab, src):
            saved += 1
            per_class[lab] = per_class.get(lab, 0) + 1
        else:
            skipped.append(name)

    for f in files:
        data = await f.read()
        if f.filename.lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                raise HTTPException(400, f"{f.filename} is not a valid zip")
            for info in zf.infolist():
                if info.is_dir() or not is_image_name(info.filename) or "__MACOSX" in info.filename:
                    continue
                parent = Path(info.filename).parent.name
                lab = fixed or resolve_label(parent)
                if not lab:
                    skipped.append(info.filename)
                    continue
                add(zf.read(info), info.filename, lab, f"zip:{f.filename}")
        elif is_image_name(f.filename):
            if not fixed:
                raise HTTPException(400, "single images need a `label` form field")
            add(data, f.filename, fixed, "api")
        else:
            skipped.append(f.filename)
    return {"saved": saved, "per_class": per_class, "skipped": len(skipped),
            "skipped_examples": skipped[:10], "pending_by_class": db.upload_counts()}


# ------------------------------------------------------------------ retraining
@app.post("/retrain")
def retrain():
    run_id = start_retraining()
    if run_id is None:
        raise HTTPException(409, "a retraining run is already in progress")
    return {"run_id": run_id, "status": "running",
            "message": "Retraining started in the background. Poll /retrain/status."}


@app.get("/retrain/status")
def retrain_status():
    runs = db.runs(1)
    if not runs:
        return {"status": "never-run"}
    r = runs[0]
    if r.get("message", "") and r["message"].startswith("{"):
        r["details"] = json.loads(r["message"])
    return r


# ------------------------------------------------------------------ UI
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
