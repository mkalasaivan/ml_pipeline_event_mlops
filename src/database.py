"""SQLite storage for uploaded training images, predictions and retraining runs."""
import sqlite3
import time
from contextlib import contextmanager

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    source TEXT,
    uploaded_at REAL NOT NULL,
    used_in_run INTEGER            -- id of the training run that consumed it (NULL = not yet used)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    predicted TEXT NOT NULL,
    confidence REAL NOT NULL,
    latency_ms REAL,
    model_version TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,          -- running | completed | rejected | failed
    started_at REAL NOT NULL,
    finished_at REAL,
    n_images INTEGER,
    old_accuracy REAL,
    new_accuracy REAL,
    promoted INTEGER DEFAULT 0,
    model_version TEXT,
    message TEXT
);
"""


@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


def add_upload(filename, path, label, source="api"):
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO uploads (filename, path, label, source, uploaded_at) VALUES (?,?,?,?,?)",
                    (filename, str(path), label, source, time.time()))


def pending_uploads():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM uploads WHERE used_in_run IS NULL")]


def all_uploads():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM uploads")]


def upload_counts():
    with connect() as con:
        rows = con.execute("""SELECT label, COUNT(*) AS total,
                              SUM(CASE WHEN used_in_run IS NULL THEN 1 ELSE 0 END) AS pending
                              FROM uploads GROUP BY label ORDER BY label""")
        return [dict(r) for r in rows]


def mark_used(upload_ids, run_id):
    with connect() as con:
        con.executemany("UPDATE uploads SET used_in_run=? WHERE id=?", [(run_id, i) for i in upload_ids])


def log_prediction(filename, predicted, confidence, latency_ms, model_version):
    with connect() as con:
        con.execute("INSERT INTO predictions (filename, predicted, confidence, latency_ms, model_version, created_at)"
                    " VALUES (?,?,?,?,?,?)", (filename, predicted, confidence, latency_ms, model_version, time.time()))


def prediction_summary():
    with connect() as con:
        by_class = [dict(r) for r in con.execute(
            "SELECT predicted, COUNT(*) AS n, AVG(confidence) AS avg_conf FROM predictions GROUP BY predicted")]
        tot = con.execute("SELECT COUNT(*), AVG(latency_ms) FROM predictions").fetchone()
        return {"total": tot[0], "avg_latency_ms": tot[1], "by_class": by_class}


def start_run():
    with connect() as con:
        if con.execute("SELECT 1 FROM training_runs WHERE status='running'").fetchone():
            return None
        cur = con.execute("INSERT INTO training_runs (status, started_at) VALUES ('running', ?)", (time.time(),))
        return cur.lastrowid


def finish_run(run_id, status, **fields):
    fields.update(status=status, finished_at=time.time())
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE training_runs SET {cols} WHERE id=?", (*fields.values(), run_id))


def fail_stale_runs():
    """Mark runs left 'running' by a crashed/restarted container as failed."""
    with connect() as con:
        con.execute("UPDATE training_runs SET status='failed', message='interrupted (server restart)',"
                    " finished_at=? WHERE status='running' AND started_at < ?", (time.time(), time.time() - 3600))


def runs(limit=10):
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))]
