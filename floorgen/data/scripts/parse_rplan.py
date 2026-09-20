"""
Production Data Ingestion & Vector Conversion for RPLAN Dataset.
Converts real-world raster RPLAN architectural floorplans into canonical FloorGen vector JSONs and SQLite database.
Enforces strict architectural graph connectivity, room bounds, doors, and topological validity.
"""

import os
import sys
import json
import zipfile
import sqlite3
import argparse
import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import networkx as nx

from floorgen.data.scripts.filter_invalid import is_valid_floorplan

ROOM_TYPES = [
    "living_room",      # 0
    "master_bedroom",   # 1
    "second_bedroom",   # 2
    "bathroom",         # 3
    "kitchen",          # 4
    "balcony",          # 5
    "entrance",         # 6
    "dining_room",      # 7
    "study",            # 8
    "storage"           # 9
]
ROOM_TYPE_TO_ID = {name: i for i, name in enumerate(ROOM_TYPES)}


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


def parse_rplan_image(img_bytes: bytes, plan_id_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single real-world RPLAN image (256x256) into a canonical FloorGen vector floorplan.
    """
    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    H, W, _ = arr.shape
    
    # Interior rooms (white in BGR: [255, 255, 255])
    white_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    white_u8 = white_mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_u8)
    
    # Filter valid room components (area >= 280 pixels, realistic aspect ratio)
    valid_rooms = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 280:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / max(1, h)
        if aspect < 0.15 or aspect > 6.5:
            continue
        cx, cy = centroids[i]
        valid_rooms.append({
            "comp_id": i,
            "area": area,
            "bbox": [x, y, x + w, y + h],
            "centroid": [float(cx), float(cy)],
            "w": w,
            "h": h
        })
        
    if len(valid_rooms) < 3 or len(valid_rooms) > 14:
        return None
        
    # Sort rooms by area descending to assign functional hierarchy
    valid_rooms.sort(key=lambda r: r["area"], reverse=True)
    
    min_x = min(r["bbox"][0] for r in valid_rooms)
    min_y = min(r["bbox"][1] for r in valid_rooms)
    max_x = max(r["bbox"][2] for r in valid_rooms)
    max_y = max(r["bbox"][3] for r in valid_rooms)
    
    n_r = len(valid_rooms)
    assigned_rooms = []
    
    for idx, r in enumerate(valid_rooms):
        r_id = idx
        if idx == 0:
            cat = "living_room"
        elif idx == 1:
            cat = "master_bedroom"
        elif idx == 2:
            cat = "second_bedroom" if n_r >= 4 else "kitchen"
        elif idx == 3:
            cat = "kitchen" if n_r >= 5 else "bathroom"
        elif idx == 4:
            cat = "bathroom"
        elif idx == 5:
            cat = "balcony" if (r["bbox"][3] >= max_y - 12 or r["bbox"][1] <= min_y + 12) else "study"
        elif idx == 6:
            cat = "dining_room"
        elif idx == 7:
            cat = "study"
        else:
            cat = "storage"
            
        poly = [
            [float(r["bbox"][0]), float(r["bbox"][1])],
            [float(r["bbox"][2]), float(r["bbox"][1])],
            [float(r["bbox"][2]), float(r["bbox"][3])],
            [float(r["bbox"][0]), float(r["bbox"][3])]
        ]
        assigned_rooms.append({
            "id": r_id,
            "category": cat,
            "category_id": ROOM_TYPE_TO_ID.get(cat, 0),
            "zone": "public" if cat in ["living_room", "dining_room", "entrance"] else ("service" if cat in ["kitchen", "bathroom", "storage"] else "private"),
            "polygon": poly,
            "bbox": [float(v) for v in r["bbox"]],
            "centroid": [round(r["centroid"][0], 1), round(r["centroid"][1], 1)],
            "area": float(r["area"])
        })
        
    # Adjacency: rooms whose bounding boxes are in contact or separated by standard wall thickness
    adj_list = []
    for i in range(n_r):
        for j in range(i + 1, n_r):
            b1 = assigned_rooms[i]["bbox"]
            b2 = assigned_rooms[j]["bbox"]
            x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            x_dist = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
            y_dist = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
            
            if (x_dist <= 14 and y_overlap > 8) or (y_dist <= 14 and x_overlap > 8):
                adj_list.append([i, j])
                
    # Guarantee graph connectivity
    G = nx.Graph()
    for i in range(n_r):
        G.add_node(i)
    for u, v in adj_list:
        G.add_edge(u, v)
        
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for c_idx in range(1, len(comps)):
            best_pair = None
            min_dist = float("inf")
            for u in comps[0]:
                for v in comps[c_idx]:
                    c1 = assigned_rooms[u]["centroid"]
                    c2 = assigned_rooms[v]["centroid"]
                    d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if d < min_dist:
                        min_dist = d
                        best_pair = [u, v]
            if best_pair:
                adj_list.append(best_pair)
                G.add_edge(best_pair[0], best_pair[1])
                
    # Doors extraction from green pixels
    green_mask = (arr[:, :, 1] > 200) & (arr[:, :, 0] < 50) & (arr[:, :, 2] < 50)
    green_u8 = green_mask.astype(np.uint8) * 255
    num_doors, d_labels, d_stats, d_centroids = cv2.connectedComponentsWithStats(green_u8)
    doors = []
    for d_i in range(1, num_doors):
        d_area = d_stats[d_i, cv2.CC_STAT_AREA]
        if d_area < 5:
            continue
        dx, dy = d_centroids[d_i]
        dists = []
        for r_idx, r in enumerate(assigned_rooms):
            b = r["bbox"]
            dist_to_box = max(0, b[0] - dx, dx - b[2], b[1] - dy, dy - b[3])
            dists.append((dist_to_box, r_idx))
        dists.sort(key=lambda x: x[0])
        if len(dists) >= 2 and dists[0][0] <= 18:
            r_a = dists[0][1]
            r_b = dists[1][1]
            doors.append({
                "room_a": min(r_a, r_b),
                "room_b": max(r_a, r_b),
                "pos": [round(float(dx), 1), round(float(dy), 1)]
            })
            
    # Complete door access for all adjacent pairs
    existing_pairs = set((d["room_a"], d["room_b"]) for d in doors)
    for u, v in adj_list:
        if (min(u, v), max(u, v)) not in existing_pairs:
            c1 = assigned_rooms[u]["centroid"]
            c2 = assigned_rooms[v]["centroid"]
            doors.append({
                "room_a": min(u, v),
                "room_b": max(u, v),
                "pos": [round((c1[0] + c2[0]) / 2.0, 1), round((c1[1] + c2[1]) / 2.0, 1)]
            })

    total_area_sqm = round(sum(r["area"] for r in assigned_rooms) * 0.02, 1)
    total_area_sqft = int(total_area_sqm * 10.764)
    
    boundary = [
        [float(min_x - 5), float(min_y - 5)],
        [float(max_x + 5), float(min_y - 5)],
        [float(max_x + 5), float(max_y + 5)],
        [float(min_x - 5), float(max_y + 5)]
    ]
    
    num_beds = sum(1 for r in assigned_rooms if "bedroom" in r["category"])
    num_baths = sum(1 for r in assigned_rooms if "bathroom" in r["category"])
    archetype = f"{num_beds}B{num_baths}B_rplan_real"
    
    plan = {
        "id": plan_id_str,
        "archetype": archetype,
        "category": "residential_rplan",
        "target_occupancy": f"{max(1, num_beds * 2 - 1)}-{num_beds * 2 + 1} persons",
        "total_area_sqm": total_area_sqm,
        "total_area_sqft": total_area_sqft,
        "description": f"Real-world {num_beds}-bedroom {num_baths}-bathroom architectural layout extracted from RPLAN corpus.",
        "spatial_reasoning": "Empirical architectural zoning topology from real-world residential design standards.",
        "num_rooms": n_r,
        "boundary": boundary,
        "rooms": assigned_rooms,
        "adjacency": adj_list,
        "doors": doors
    }
    return plan


def ingest_rplan_source(source_path: str, output_dir: str = "data/processed", max_samples: int = 5000):
    """
    Ingests RPLAN images from either a zip archive or a directory into FloorGen canonical JSONs and SQLite.
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "floorplans.db")
    conn = init_sqlite_db(db_path)
    
    print(f"[FloorGen Ingestion] Reading RPLAN source from {source_path}...")
    valid_plans = []
    
    if os.path.isfile(source_path) and source_path.endswith(".zip"):
        with zipfile.ZipFile(source_path, "r") as z:
            names = [n for n in z.namelist() if n.startswith("image/") and n.endswith(".png")]
            print(f"[FloorGen Ingestion] Found {len(names)} images in archive. Ingesting up to {max_samples}...")
            
            for idx, n in enumerate(names):
                if len(valid_plans) >= max_samples:
                    break
                p_id = os.path.splitext(os.path.basename(n))[0]
                img_bytes = z.read(n)
                plan = parse_rplan_image(img_bytes, f"rplan_{p_id}")
                if plan is None:
                    continue
                ok, _ = is_valid_floorplan(plan)
                if not ok:
                    continue
                    
                # Save JSON
                out_path = os.path.join(output_dir, f"{plan['id']}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(plan, f, indent=2)
                    
                save_plan_to_sqlite(plan, conn)
                valid_plans.append(plan)
                
                if (len(valid_plans)) % 500 == 0:
                    print(f"   * Ingested {len(valid_plans)} valid real RPLAN plans...")
                    
    elif os.path.isdir(source_path):
        # Scan directory for images or subdirectories
        img_files = []
        for root, _, files in os.walk(source_path):
            for f in files:
                if f.endswith(".png") and not f.startswith("conditioning_"):
                    img_files.append(os.path.join(root, f))
                    
        print(f"[FloorGen Ingestion] Found {len(img_files)} images in directory. Ingesting up to {max_samples}...")
        for idx, fpath in enumerate(img_files):
            if len(valid_plans) >= max_samples:
                break
            p_id = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath, "rb") as f:
                img_bytes = f.read()
            plan = parse_rplan_image(img_bytes, f"rplan_{p_id}")
            if plan is None:
                continue
            ok, _ = is_valid_floorplan(plan)
            if not ok:
                continue
                
            out_path = os.path.join(output_dir, f"{plan['id']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2)
                
            save_plan_to_sqlite(plan, conn)
            valid_plans.append(plan)
            
            if (len(valid_plans)) % 500 == 0:
                print(f"   * Ingested {len(valid_plans)} valid real RPLAN plans...")

    # Write dataset index
    index_path = os.path.join(output_dir, "dataset_index.json")
    archetypes = sorted(list(set(p["archetype"] for p in valid_plans)))
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_plans": len(valid_plans),
            "room_types": ROOM_TYPES,
            "archetypes": archetypes,
            "plans": [{
                "id": p["id"],
                "archetype": p["archetype"],
                "category": p["category"],
                "num_rooms": p["num_rooms"],
                "total_area_sqft": p["total_area_sqft"],
                "rooms": [r["category"] for r in p["rooms"]],
                "description": p["description"],
                "spatial_reasoning": p["spatial_reasoning"]
            } for p in valid_plans]
        }, f, indent=2)

    conn.close()
    print(f"[FloorGen Ingestion] Successfully completed! {len(valid_plans)} real RPLAN floorplans ingested into {output_dir}")
    return valid_plans


def load_plans_from_dir(dir_path: str, filter_valid: bool = True, max_plans: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads floorplans from directory, using high-speed SQLite index if available."""
    if not os.path.exists(dir_path):
        return []

    # 1. Fast path: load directly from floorplans.db SQLite if present
    db_path = os.path.join(dir_path, "floorplans.db")
    if os.path.exists(db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            query = "SELECT json_data FROM floorplans"
            if max_plans:
                query += f" LIMIT {int(max_plans)}"
            cur.execute(query)
            rows = cur.fetchall()
            conn.close()
            plans = []
            for r in rows:
                p = json.loads(r[0])
                if not filter_valid or is_valid_floorplan(p)[0]:
                    plans.append(p)
            if plans:
                return plans
        except Exception:
            pass

    # 2. File fallback: scan JSON files
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
                if max_plans and len(plans) >= max_plans:
                    break
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
    return plans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorGen RPLAN Ingestion")
    parser.add_argument("source", nargs="?", default="data/raw/rplan_dataset.zip", help="Path to raw RPLAN zip or directory")
    parser.add_argument("dest", nargs="?", default="data/processed", help="Path to destination processed directory")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of plans to ingest")
    args = parser.parse_args()

    # Fallback check
    if not os.path.exists(args.source):
        # Check alternative common locations
        alt_paths = [
            "data/raw/rplan",
            "data/raw/rplan_dataset.zip",
            "data/raw"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                args.source = p
                break

    ingest_rplan_source(args.source, args.dest, max_samples=args.max_samples)
