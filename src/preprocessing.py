"""Image preprocessing. Mirrors section 4 of the notebook.

Rescaling to [-1, 1] and augmentation live *inside* the model, so the only
preprocessing needed here is: decode -> RGB -> resize to IMG_SIZE x IMG_SIZE -> float32.
Training and serving therefore always see identically prepared pixels.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import ALLOWED_EXTS, img_size


class InvalidImage(ValueError):
    pass


def validate_image_bytes(data: bytes) -> str:
    """Return the image format if `data` is a readable JPEG/PNG/BMP, else raise InvalidImage."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
            fmt = im.format
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise InvalidImage(f"not a readable image: {e}") from e
    if fmt not in {"JPEG", "PNG", "BMP"}:
        raise InvalidImage(f"unsupported image format {fmt}")
    return fmt


def bytes_to_array(data: bytes, size: int | None = None) -> np.ndarray:
    """Decode raw image bytes into a (size, size, 3) float32 array with values in [0, 255]."""
    size = size or img_size()
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImage(str(e)) from e


def load_image_file(path: str | Path, size: int | None = None) -> np.ndarray:
    return bytes_to_array(Path(path).read_bytes(), size)


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTS


def make_dataset(paths, labels, batch_size=32, shuffle=False, seed=42):
    """tf.data pipeline over image files (same steps as the notebook's make_ds)."""
    import tensorflow as tf

    size = img_size()

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), np.asarray(labels, dtype=np.int32)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def image_features(data: bytes) -> dict:
    """The colour features used in the EDA (grass-green / red / yellow pixel share, brightness)."""
    a = bytes_to_array(data, 128)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return {
        "brightness": round(float(a.mean()), 2),
        "green_ratio": round(float(((g > r + 10) & (g > b + 10)).mean()), 4),
        "red_ratio": round(float(((r > 150) & (r > g + 60) & (r > b + 60)).mean()), 4),
        "yellow_ratio": round(float(((r > 150) & (g > 150) & (b < 100)).mean()), 4),
    }
