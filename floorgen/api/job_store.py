"""
Persistent Background Job Store for FloorGen API.
Provides an SQLite-backed queue and result cache that persists across server restarts.
"""

import os
import json
import sqlite3
import threading
import time
from typing import Dict, Any, Optional, List

from floorgen.logging_config import get_logger

log = get_logger("job_store")


class PersistentJobStore:
    """Thread-safe SQLite persistent job repository."""
    _instance: Optional["PersistentJobStore"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = "data/jobs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: str = "data/jobs.db") -> "PersistentJobStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT,
                    total_requests INTEGER,
                    created_at REAL,
                    updated_at REAL,
                    results_json TEXT,
                    error TEXT
                )
            """)
            conn.commit()

    def create_job(self, job_id: str, total_requests: int):
        now = time.time()
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "total_requests": total_requests,
            "created_at": now,
            "updated_at": now,
            "results": None,
            "error": None
        }
        with self._lock:
            self._cache[job_id] = job_data

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO batch_jobs (job_id, status, total_requests, created_at, updated_at, results_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (job_id, "queued", total_requests, now, now, None, None))
            conn.commit()

    def update_job_status(self, job_id: str, status: str, results: Optional[List[Dict[str, Any]]] = None, error: Optional[str] = None):
        now = time.time()
        results_json = json.dumps(results) if results is not None else None
        
        with self._lock:
            if job_id in self._cache:
                self._cache[job_id]["status"] = status
                self._cache[job_id]["results"] = results
                self._cache[job_id]["error"] = error
                self._cache[job_id]["updated_at"] = now

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_jobs 
                SET status = ?, updated_at = ?, results_json = ?, error = ?
                WHERE job_id = ?
            """, (status, now, results_json, error, job_id))
            conn.commit()

    def save_job(self, job_id: str, job_dict: Dict[str, Any]):
        status = job_dict.get("status", "completed")
        results = job_dict.get("results")
        error = job_dict.get("error")
        total = job_dict.get("total_requests", 1)
        now = time.time()
        
        with self._lock:
            self._cache[job_id] = {
                "job_id": job_id,
                "status": status,
                "total_requests": total,
                "created_at": now,
                "updated_at": now,
                "results": results,
                "error": error,
                **job_dict
            }

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO batch_jobs (job_id, status, total_requests, created_at, updated_at, results_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (job_id, status, total, now, now, json.dumps(results) if results else None, error))
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if job_id in self._cache:
                return self._cache[job_id]

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT job_id, status, total_requests, created_at, updated_at, results_json, error
                FROM batch_jobs WHERE job_id = ?
            """, (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            results = json.loads(row[5]) if row[5] else None
            job_data = {
                "job_id": row[0],
                "status": row[1],
                "total_requests": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "results": results,
                "error": row[6]
            }
            with self._lock:
                self._cache[job_id] = job_data
            return job_data

