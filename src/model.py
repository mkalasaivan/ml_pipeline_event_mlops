"""Model loading (with hot reload across containers) and retraining from the saved model."""
import json
import os
import random
import shutil
import threading
import time
from collections import defaultdict

import numpy as np

from src import database as db
from src.config import META_PATH, MODEL_H5_PATH, MODEL_PATH, MODELS_DIR, class_names, load_meta
from src.preprocessing import make_dataset

MIN_RETRAIN_IMAGES = int(os.getenv("MIN_RETRAIN_IMAGES", "14"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "6"))
FINE_TUNE_LAYERS = 40          # same as notebook phase 2

_lock = threading.Lock()
_model = None
_loaded_mtime = None


def _tf():
    import tensorflow as tf
    return tf


def get_model():
    """Return the current model, reloading it if another worker/container saved a newer one."""
    global _model, _loaded_mtime
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Unzip the Colab artifacts into models/.")
    mtime = MODEL_PATH.stat().st_mtime
    if _model is None or mtime != _loaded_mtime:
        with _lock:
            if _model is None or mtime != _loaded_mtime:
                _model = _tf().keras.models.load_model(MODEL_PATH, compile=False)
                _loaded_mtime = mtime
    return _model


def model_version() -> str:
    return str(load_meta().get("version", "v1"))


# ---------------------------------------------------------------- evaluation
def _metrics(y_true, proba):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y_pred = proba.argmax(1)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def _split(uploads, holdout=0.2, seed=42):
    """Stratified split of the *new* uploads into train / held-out evaluation."""
    by_label = defaultdict(list)
    for u in uploads:
        by_label[u["label"]].append(u)
    rng = random.Random(seed)
    train, held = [], []
    for items in by_label.values():
        rng.shuffle(items)
        k = int(round(len(items) * holdout)) if len(items) >= 3 else 0
        held += items[:k]
        train += items[k:]
    return train, held


def _prepare_for_fine_tuning(model):
    """Unfreeze the top of the MobileNetV2 backbone and the head; keep BatchNorm frozen."""
    tf = _tf()
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is not None:
        base.trainable = True
        for layer in base.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------- retraining
def retrain(run_id: int):
    """Retrain the saved model on uploaded images.

    1. Take the uploads not used by a previous run and hold out 20% (stratified) for evaluation.
    2. Train on the rest plus every image used in earlier runs (so earlier uploads aren't forgotten).
    3. Fine-tune a copy of the *current saved model* (not a model from scratch).
    4. Compare old vs new on the held-out uploads; promote the new model only if it is at least as good.
    """
    tf = _tf()
    try:
        classes = class_names()
        idx = {c: i for i, c in enumerate(classes)}
        new = [u for u in db.pending_uploads() if u["label"] in idx and os.path.exists(u["path"])]
        if len(new) < MIN_RETRAIN_IMAGES:
            db.finish_run(run_id, "failed", n_images=len(new),
                          message=f"need at least {MIN_RETRAIN_IMAGES} new uploaded images, have {len(new)}")
            return
        train_new, held = _split(new)
        old = [u for u in db.all_uploads() if u["used_in_run"] is not None and u["label"] in idx
               and os.path.exists(u["path"])]
        train = train_new + old
        if not held:                       # tiny upload: evaluate on the training images
            held = train_new

        tr_ds = make_dataset([u["path"] for u in train], [idx[u["label"]] for u in train], shuffle=True)
        ev_paths, y_ev = [u["path"] for u in held], np.array([idx[u["label"]] for u in held])
        ev_ds = make_dataset(ev_paths, y_ev)

        current = get_model()
        before = _metrics(y_ev, current.predict(ev_ds, verbose=0))

        candidate = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _prepare_for_fine_tuning(candidate)
        counts = np.bincount([idx[u["label"]] for u in train], minlength=len(classes))
        class_weight = {i: float(len(train) / (len(classes) * c)) for i, c in enumerate(counts) if c > 0}
        hist = candidate.fit(tr_ds, epochs=RETRAIN_EPOCHS, class_weight=class_weight, verbose=2,
                             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2,
                                                                         restore_best_weights=True)])
        after = _metrics(y_ev, candidate.predict(ev_ds, verbose=0))

        promoted = after["accuracy"] >= before["accuracy"]
        meta = load_meta()
        version = model_version()
        if promoted:
            n = int(str(meta.get("version", "v1")).lstrip("v") or 1) + 1
            version = f"v{n}"
            _save(candidate, meta, version, run_id, before, after, len(train), len(held))
        db.mark_used([u["id"] for u in new], run_id)
        db.finish_run(run_id, "completed" if promoted else "rejected",
                      n_images=len(new), old_accuracy=before["accuracy"], new_accuracy=after["accuracy"],
                      promoted=int(promoted), model_version=version,
                      message=json.dumps({"before": before, "after": after,
                                          "epochs": len(hist.history["loss"]),
                                          "n_train": len(train), "n_eval": len(held)}))
    except Exception as e:  # noqa: BLE001 - record any failure on the run
        db.finish_run(run_id, "failed", message=f"{type(e).__name__}: {e}")
        raise


def _save(model, meta, version, run_id, before, after, n_train, n_eval):
    """Back up the current model, then atomically replace it so other workers pick it up."""
    backup = MODELS_DIR / "previous"
    backup.mkdir(exist_ok=True)
    for p in (MODEL_PATH, MODEL_H5_PATH, META_PATH):
        if p.exists():
            shutil.copy2(p, backup / p.name)
    tmp = MODELS_DIR / f"tmp_{run_id}.keras"
    model.save(tmp)
    # HDF5 copy is best-effort: Keras 3 can fail to serialise some layer configs to .h5,
    # in which case the weights are saved in HDF5 instead. The .keras file is what the API loads.
    h5_tmp = MODELS_DIR / f"tmp_{run_id}.h5"
    try:
        model.save(h5_tmp)
        os.replace(h5_tmp, MODEL_H5_PATH)
    except Exception:  # noqa: BLE001
        h5_tmp.unlink(missing_ok=True)
        model.save_weights(MODELS_DIR / "football_event_model.weights.h5")
    meta.update(version=version, retrained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                last_retrain={"run_id": run_id, "before": before, "after": after,
                              "n_train": n_train, "n_eval": n_eval})
    meta.setdefault("history", []).append({"version": version, "run_id": run_id,
                                           "accuracy": after["accuracy"], "at": meta["retrained_at"]})
    META_PATH.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, MODEL_PATH)       # last: the mtime change triggers reload everywhere


def start_retraining() -> int | None:
    """Create a run and train in a background thread. Returns None if a run is already going."""
    run_id = db.start_run()
    if run_id is None:
        return None
    threading.Thread(target=retrain, args=(run_id,), daemon=True, name=f"retrain-{run_id}").start()
    return run_id
