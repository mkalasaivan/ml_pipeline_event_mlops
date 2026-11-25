"""Paths and settings shared by the API, preprocessing, training and prediction code."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"            # images added through the API, one folder per class
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "football_events.db"))

MODEL_PATH = MODELS_DIR / "football_event_model.keras"   # current production model
MODEL_H5_PATH = MODELS_DIR / "football_event_model.h5"   # same model in HDF5 format
META_PATH = MODELS_DIR / "model_meta.json"
EDA_PATH = MODELS_DIR / "eda_stats.json"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_CLASSES = ["Corner_Kick", "Free_Kick", "Penalty_kick", "Red_Card",
                   "Yellow_Card", "substitute", "tackle"]


def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"class_names": DEFAULT_CLASSES, "img_size": 224}


def class_names() -> list:
    return load_meta().get("class_names", DEFAULT_CLASSES)


def img_size() -> int:
    return int(load_meta().get("img_size", 224))


for d in (MODELS_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)
