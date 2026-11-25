"""Single-image prediction."""
import time

import numpy as np

from src.config import class_names
from src.model import get_model, model_version
from src.preprocessing import bytes_to_array


def predict_bytes(data: bytes, top_k: int = 3) -> dict:
    t = time.perf_counter()
    x = bytes_to_array(data)[None, ...]
    proba = np.asarray(get_model()(x, training=False))[0]
    classes = class_names()
    order = proba.argsort()[::-1]
    return {
        "predicted_class": classes[order[0]],
        "confidence": round(float(proba[order[0]]), 4),
        "top_k": [{"class": classes[i], "probability": round(float(proba[i]), 4)} for i in order[:top_k]],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "model_version": model_version(),
        "latency_ms": round((time.perf_counter() - t) * 1000, 1),
    }
