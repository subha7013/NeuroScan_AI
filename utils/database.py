# utils/database.py
"""SQLite helper utilities for storing prediction history.
The database file will be located at `database/history.db` within the project.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Tuple, Dict

DB_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "database")
DB_PATH = os.path.join(DB_DIR, "history.db")

def _ensure_db():
    """Create the database directory and the predictions table if they do not exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def insert_prediction(filename: str, patient_id: str, predicted_class: str, confidence: float):
    """Insert a new prediction record.

    Args:
        filename: Name of the uploaded MRI file.
        patient_id: Identifier supplied by the user.
        predicted_class: Human‑readable class label.
        confidence: Confidence (0‑1) of the prediction.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    cursor.execute(
        "INSERT INTO predictions (timestamp, filename, patient_id, predicted_class, confidence) VALUES (?,?,?,?,?)",
        (timestamp, filename, patient_id, predicted_class, confidence),
    )
    conn.commit()
    conn.close()

def fetch_history(limit: int = 100) -> List[Tuple[int, str, str, str, str, float]]:
    """Return recent prediction rows ordered by newest first.
    Returns a list of tuples matching the table columns.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, filename, patient_id, predicted_class, confidence FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
