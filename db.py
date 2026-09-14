"""
db.py

Handles the SQLite database that stores detection history:
image name, detected class, confidence score, and timestamp.
"""

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

DB_PATH = "sonaris_history.db"


def init_db():
    """Creates the detections table if it doesn't already exist."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            target_class TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_detection(image_name: str, target_class: str, confidence: float):
    """Inserts one detection record into the database."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d-%m-%Y %I:%M:%S %p")

    cursor.execute(
        """
        INSERT INTO detections
        (image_name, target_class, confidence, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
            image_name,
            target_class,
            confidence,
            current_time,
        ),
    )

    conn.commit()
    conn.close()


def get_all_detections():
    """Returns all detection records, most recent first."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT image_name, target_class, confidence, timestamp
        FROM detections
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows
