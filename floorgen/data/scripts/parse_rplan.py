"""
Data ingestion & canonical SQLite/JSON storage for RPLAN & ResPlan formats.
Converts raw raster/vector representations into canonical FloorGen JSON and SQLite database.
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
from floorgen.data.scripts.filter_invalid import is_valid_floorplan


def init_sqlite_db(db_path: str = "data/processed/floorplans.db") -> sqlite3.Connection:
    """Initializes the canonical SQLite store for fast querying and metadata retrieval."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS floorplans (
            id TEXT PRIMARY KEY,
            archetype TEXT,
            num_rooms INTEGER,
            room_categories TEXT,
            description TEXT,
            total_area REAL,
            json_data TEXT
        )
    """)
    conn.commit()
    return conn


def save_plan_to_sqlite(plan: Dict[str, Any], conn: sqlite3.Connection):
    """Inserts or updates a canonical floorplan in the SQLite database."""
    cursor = conn.cursor()
    room_cats = ",".join([r["category"] for r in plan.get("rooms", [])])
    total_area = sum(r.get("area", 0) for r in plan.get("rooms", []))
    
    cursor.execute("""
        INSERT OR REPLACE INTO floorplans (id, archetype, num_rooms, room_categories, description, total_area, json_data)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        plan["id"],
        plan.get("archetype", "unknown"),
        plan.get("num_rooms", len(plan.get("rooms", []))),
        room_cats,
        plan.get("description", ""),
        total_area,
        json.dumps(plan)
    ))
    conn.commit()


def load_plans_from_dir(dir_path: str, filter_valid: bool = True) -> List[Dict[str, Any]]:
    """Loads all JSON floorplans from a directory."""
    if not os.path.exists(dir_path):
        return []
    plans = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".json") and fname != "dataset_index.json":
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    plan = json.load(f)
                if filter_valid:
                    ok, _ = is_valid_floorplan(plan)
                    if ok:
                        plans.append(plan)
                else:
                    plans.append(plan)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
    return plans
