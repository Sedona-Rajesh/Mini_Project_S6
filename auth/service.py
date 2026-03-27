from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from datetime import datetime
from pathlib import Path


def init_auth_db(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'clinician',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

    return path


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return dk.hex()


def _new_salt() -> str:
    return os.urandom(16).hex()


def create_user(
    db_path: str | Path,
    username: str,
    password: str,
    role: str = "clinician",
) -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    salt_hex = _new_salt()
    password_hash = _hash_password(password, salt_hex)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, password_hash, salt_hex, role, datetime.utcnow().isoformat()),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "User already exists."

    return True, "Signup successful. Please login."


def verify_user(db_path: str | Path, username: str, password: str) -> tuple[bool, dict | None]:
    username = username.strip().lower()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT username, password_hash, salt, role FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()

    if not row:
        return False, None

    db_username, db_hash, db_salt, db_role = row
    calc_hash = _hash_password(password, db_salt)
    ok = hmac.compare_digest(calc_hash, db_hash)

    if not ok:
        return False, None

    return True, {"username": db_username, "role": db_role}
