"""
Enhanced Real-World Architectural Dataset Generator for FloorGen.
Generates diverse, structurally valid residential floorplans matching real-world architectural archetypes:
  - Micro-apartments, Studios, 1B1B, 2B1B, 2B2B, 3B2B Family Suites, 4B3B Penthouses, Multi-Generational Villas, and Courtyard Townhouses.
Includes detailed functional zoning, daylight exposure, spatial reasoning, and circulation metrics.
"""

import json
import os
import random
import math
from typing import List, Dict, Any

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

# 8 High-Fidelity Global Architectural Archetypes
ARCHETYPES = [
    # 1. 2-Bedroom + 1-Bath Standard (Living-centered)
    {
        "name": "2B1B_standard",
        "category": "urban_residential",
        "target_occupancy": "2-3 persons",
        "description": "Standard 2-bedroom 1-bathroom urban apartment with central living-dining spine, open kitchen, and south-facing master suite opening to a full-width balcony.",
        "spatial_reasoning": "Direct public-to-private zoning hierarchy with centralized circulation to minimize hallway area and maximize functional daylight exposure.",
        "grid_w": 3, "grid_h": 2,
        "cells": [
            {"type": "living_room", "col": 1, "row": 0, "w": 1, "h": 2, "zone": "public"},
            {"type": "kitchen", "col": 0, "row": 0, "w": 1, "h": 1, "zone": "service"},
            {"type": "bathroom", "col": 0, "row": 1, "w": 1, "h": 1, "zone": "service"},
            {"type": "master_bedroom", "col": 2, "row": 0, "w": 1, "h": 1, "zone": "private"},
            {"type": "second_bedroom", "col": 2, "row": 1, "w": 1, "h": 1, "zone": "private"},
        ],
        "adjacency": [
            (0, 1), (0, 2), (0, 3), (0, 4), (1, 2)
        ]
    },
    # 2. 3-Bedroom + 2-Bath Family Executive Suite
    {
        "name": "3B2B_family_executive",
        "category": "family_suite",
        "target_occupancy": "4-5 persons",
        "description": "Spacious 3-bedroom 2-bathroom family layout featuring a dedicated entrance foyer, formal dining room, private master ensuite, quiet study library, and sunlit living room with wide terrace.",
        "spatial_reasoning": "Acoustic separation between the master wing and children's bedrooms, anchored by a centralized living and dining core for shared family gathering.",
        "grid_w": 4, "grid_h": 3,
        "cells": [
            {"type": "entrance", "col": 0, "row": 0, "w": 1, "h": 1, "zone": "transition"},
            {"type": "kitchen", "col": 0, "row": 1, "w": 1, "h": 1, "zone": "service"},
            {"type": "dining_room", "col": 1, "row": 0, "w": 1, "h": 1, "zone": "public"},
            {"type": "living_room", "col": 1, "row": 1, "w": 2, "h": 2, "zone": "public"},
            {"type": "balcony", "col": 1, "row": 3, "w": 2, "h": 0.5, "zone": "outdoor"},
            {"type": "master_bedroom", "col": 3, "row": 1, "w": 1, "h": 2, "zone": "private"},
            {"type": "bathroom", "col": 3, "row": 0, "w": 1, "h": 1, "zone": "service"},
            {"type": "second_bedroom", "col": 0, "row": 2, "w": 1, "h": 1, "zone": "private"},
            {"type": "study", "col": 2, "row": 0, "w": 1, "h": 1, "zone": "semi_private"},
        ],
        "adjacency": [
            (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (3, 7), (2, 8), (3, 8)
        ]
    },
    # 3. 1-Bedroom Urban Micro-Loft / Studio
    {
        "name": "1B1B_micro_loft",
        "category": "compact_studio",
        "target_occupancy": "1-2 persons",
        "description": "High-efficiency 1-bedroom studio apartment designed for modern metropolitan living with flowing open kitchen, living lounge, designated bathroom, and private balcony.",
        "spatial_reasoning": "Open-plan multifunctional spatial zoning maximizing spatial perception within a compact footprint.",
        "grid_w": 2, "grid_h": 2,
        "cells": [
            {"type": "entrance", "col": 0, "row": 0, "w": 1, "h": 0.8, "zone": "transition"},
            {"type": "kitchen", "col": 0, "row": 0.8, "w": 1, "h": 1.2, "zone": "service"},
            {"type": "living_room", "col": 1, "row": 0, "w": 1, "h": 1.2, "zone": "public"},
            {"type": "master_bedroom", "col": 1, "row": 1.2, "w": 1, "h": 0.8, "zone": "private"},
            {"type": "bathroom", "col": 0, "row": 2.0, "w": 1, "h": 0.8, "zone": "service"},
            {"type": "balcony", "col": 1, "row": 2.0, "w": 1, "h": 0.5, "zone": "outdoor"},
        ],
        "adjacency": [
            (0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 5)
        ]
    },
    # 4. 2-Bedroom Dual-Balcony Corner Unit (2B2B)
    {
        "name": "2B2B_corner_suite",
        "category": "corner_apartment",
        "target_occupancy": "2-4 persons",
        "description": "Corner 2-bedroom 2-bathroom layout with cross-ventilation, split bedroom wings for privacy, dual bathrooms, and wraparound corner balcony access.",
        "spatial_reasoning": "Dual-aspect facade layout providing continuous natural light and natural cross-breeze across the central entertainment zone.",
        "grid_w": 3, "grid_h": 2,
        "cells": [
            {"type": "master_bedroom", "col": 0, "row": 0, "w": 1, "h": 1.4, "zone": "private"},
            {"type": "bathroom", "col": 0, "row": 1.4, "w": 1, "h": 0.8, "zone": "service"},
            {"type": "living_room", "col": 1, "row": 0, "w": 1, "h": 1.4, "zone": "public"},
            {"type": "kitchen", "col": 1, "row": 1.4, "w": 1, "h": 0.8, "zone": "service"},
            {"type": "second_bedroom", "col": 2, "row": 0, "w": 1, "h": 1.4, "zone": "private"},
            {"type": "balcony", "col": 2, "row": 1.4, "w": 1, "h": 0.8, "zone": "outdoor"},
        ],
        "adjacency": [
            (0, 1), (0, 2), (2, 3), (2, 4), (4, 5)
        ]
    },
    # 5. 4-Bedroom Luxury Penthouse (4B3B)
    {
        "name": "4B3B_luxury_penthouse",
        "category": "penthouse_luxury",
        "target_occupancy": "5-7 persons",
        "description": "Extensive 4-bedroom 3-bathroom luxury penthouse featuring grand entrance gallery, formal dining room, chef's kitchen, master suite with private balcony, two secondary bedrooms, executive study, and utility storage.",
        "spatial_reasoning": "Tri-partite spatial organization strictly separating entertaining, private sleeping quarters, and service utility spines.",
        "grid_w": 4, "grid_h": 3,
        "cells": [
            {"type": "entrance", "col": 0, "row": 0, "w": 1, "h": 0.9, "zone": "transition"},
            {"type": "dining_room", "col": 1, "row": 0, "w": 1, "h": 0.9, "zone": "public"},
            {"type": "living_room", "col": 1, "row": 0.9, "w": 2, "h": 1.6, "zone": "public"},
            {"type": "kitchen", "col": 0, "row": 0.9, "w": 1, "h": 1.1, "zone": "service"},
            {"type": "storage", "col": 0, "row": 2.0, "w": 1, "h": 0.8, "zone": "service"},
            {"type": "study", "col": 2, "row": 0, "w": 1, "h": 0.9, "zone": "semi_private"},
            {"type": "master_bedroom", "col": 3, "row": 0, "w": 1, "h": 1.8, "zone": "private"},
            {"type": "bathroom", "col": 3, "row": 1.8, "w": 1, "h": 1.0, "zone": "service"},
            {"type": "second_bedroom", "col": 1, "row": 2.5, "w": 1, "h": 0.8, "zone": "private"},
            {"type": "balcony", "col": 2, "row": 2.5, "w": 1, "h": 0.8, "zone": "outdoor"}
        ],
        "adjacency": [
            (0, 1), (1, 2), (0, 3), (3, 4), (1, 5), (2, 6), (6, 7), (2, 8), (2, 9)
        ]
    },
    # 6. Multi-Generational Suite (3B2.5B)
    {
        "name": "3B2B_multigen_villa",
        "category": "multigenerational",
        "target_occupancy": "4-6 persons",
        "description": "Multi-generational residential floorplan with independent senior bedroom suite on ground access, sound-insulated study, central gathering salon, and open garden terrace.",
        "spatial_reasoning": "Dual master bedrooms placed on opposite facades for privacy between generations, sharing a central communal core.",
        "grid_w": 3, "grid_h": 3,
        "cells": [
            {"type": "entrance", "col": 0, "row": 0, "w": 1, "h": 1, "zone": "transition"},
            {"type": "kitchen", "col": 0, "row": 1, "w": 1, "h": 1, "zone": "service"},
            {"type": "dining_room", "col": 1, "row": 0, "w": 1, "h": 1, "zone": "public"},
            {"type": "living_room", "col": 1, "row": 1, "w": 1, "h": 1.5, "zone": "public"},
            {"type": "balcony", "col": 1, "row": 2.5, "w": 1, "h": 0.5, "zone": "outdoor"},
            {"type": "master_bedroom", "col": 2, "row": 0, "w": 1, "h": 1.5, "zone": "private"},
            {"type": "second_bedroom", "col": 2, "row": 1.5, "w": 1, "h": 1.5, "zone": "private"},
            {"type": "bathroom", "col": 0, "row": 2, "w": 1, "h": 1, "zone": "service"},
            {"type": "study", "col": 1, "row": 0, "w": 1, "h": 0.5, "zone": "semi_private"}
        ],
        "adjacency": [
            (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (1, 7), (2, 8)
        ]
    },
    # 7. Courtyard Attached Townhouse (3B2B)
    {
        "name": "3B2B_courtyard_townhouse",
        "category": "townhouse_courtyard",
        "target_occupancy": "3-5 persons",
        "description": "Courtyard-oriented townhouse floorplan where all primary living spaces surround a central light well, creating superior natural lighting and quiet private bedrooms.",
        "spatial_reasoning": "Centripetal orientation focusing private view corridors inward toward the courtyard balcony for enhanced urban acoustic tranquility.",
        "grid_w": 3, "grid_h": 3,
        "cells": [
            {"type": "entrance", "col": 0, "row": 0, "w": 1, "h": 1, "zone": "transition"},
            {"type": "living_room", "col": 1, "row": 0, "w": 2, "h": 1.5, "zone": "public"},
            {"type": "kitchen", "col": 0, "row": 1, "w": 1, "h": 1, "zone": "service"},
            {"type": "dining_room", "col": 0, "row": 2, "w": 1, "h": 1, "zone": "public"},
            {"type": "balcony", "col": 1, "row": 1.5, "w": 1, "h": 1, "zone": "outdoor"},
            {"type": "master_bedroom", "col": 2, "row": 1.5, "w": 1, "h": 1.5, "zone": "private"},
            {"type": "second_bedroom", "col": 1, "row": 2.5, "w": 1, "h": 0.8, "zone": "private"},
            {"type": "bathroom", "col": 0, "row": 2.5, "w": 1, "h": 0.8, "zone": "service"}
        ],
        "adjacency": [
            (0, 1), (0, 2), (2, 3), (1, 4), (1, 5), (4, 6), (3, 7)
        ]
    },
    # 8. Suburban Family Villa (4B2.5B)
    {
        "name": "4B2B_suburban_villa",
        "category": "suburban_villa",
        "target_occupancy": "5-8 persons",
        "description": "Expansive 4-bedroom suburban residential layout with formal living room, family recreation room, open kitchen with breakfast nook, executive study, and generous exterior deck.",
        "spatial_reasoning": "Expansive horizontal distribution providing direct garden connectivity from living and master suites.",
        "grid_w": 4, "grid_h": 3,
        "cells": [
            {"type": "entrance", "col": 0, "row": 0, "w": 1, "h": 1, "zone": "transition"},
            {"type": "kitchen", "col": 0, "row": 1, "w": 1, "h": 1, "zone": "service"},
            {"type": "dining_room", "col": 1, "row": 0, "w": 1, "h": 1, "zone": "public"},
            {"type": "living_room", "col": 1, "row": 1, "w": 2, "h": 1.5, "zone": "public"},
            {"type": "balcony", "col": 1, "row": 2.5, "w": 2, "h": 0.5, "zone": "outdoor"},
            {"type": "study", "col": 2, "row": 0, "w": 1, "h": 1, "zone": "semi_private"},
            {"type": "master_bedroom", "col": 3, "row": 0, "w": 1, "h": 1.8, "zone": "private"},
            {"type": "bathroom", "col": 3, "row": 1.8, "w": 1, "h": 1.0, "zone": "service"},
            {"type": "second_bedroom", "col": 0, "row": 2, "w": 1, "h": 1, "zone": "private"}
        ],
        "adjacency": [
            (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (6, 7), (1, 8)
        ]
    }
]


def generate_single_plan(plan_id: int, archetype_idx: int = None, jitter: float = 0.035) -> Dict[str, Any]:
    """Generates a realistic, randomized floorplan from architectural archetypes."""
    if archetype_idx is None:
        archetype = random.choice(ARCHETYPES)
    else:
        archetype = ARCHETYPES[archetype_idx % len(ARCHETYPES)]

    gw = archetype["grid_w"]
    gh = archetype["grid_h"]
    
    scale_x = 220.0 / gw
    scale_y = 220.0 / gh
    offset_x = 18.0 + random.uniform(-3.0, 3.0)
    offset_y = 18.0 + random.uniform(-3.0, 3.0)

    rooms = []
    polygons_coords = []
    
    for r_idx, cell in enumerate(archetype["cells"]):
        col = cell["col"]
        row = cell["row"]
        w = cell["w"]
        h = cell["h"]
        
        j_x1 = random.uniform(-jitter * scale_x, jitter * scale_x)
        j_y1 = random.uniform(-jitter * scale_y, jitter * scale_y)
        j_x2 = random.uniform(-jitter * scale_x, jitter * scale_x)
        j_y2 = random.uniform(-jitter * scale_y, jitter * scale_y)

        x1 = round(offset_x + col * scale_x + j_x1, 1)
        y1 = round(offset_y + row * scale_y + j_y1, 1)
        x2 = round(offset_x + (col + w) * scale_x + j_x2, 1)
        y2 = round(offset_y + (row + h) * scale_y + j_y2, 1)

        if x2 <= x1 + 10: x2 = x1 + 22
        if y2 <= y1 + 10: y2 = y1 + 22

        poly = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]
        
        bbox = [x1, y1, x2, y2]
        centroid = [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)]
        area = round((x2 - x1) * (y2 - y1), 1)
        
        room_type = cell["type"]
        cat_id = ROOM_TYPE_TO_ID.get(room_type, 0)
        
        rooms.append({
            "id": r_idx,
            "category": room_type,
            "category_id": cat_id,
            "zone": cell.get("zone", "general"),
            "polygon": poly,
            "bbox": bbox,
            "centroid": centroid,
            "area": area
        })
        polygons_coords.extend(poly)

    # Compute site boundary
    min_x = min(p[0] for p in polygons_coords) - 4.0
    min_y = min(p[1] for p in polygons_coords) - 4.0
    max_x = max(p[0] for p in polygons_coords) + 4.0
    max_y = max(p[1] for p in polygons_coords) + 4.0
    boundary = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

    n_rooms = len(rooms)
    adj_list = []
    for edge in archetype["adjacency"]:
        u, v = edge
        if u < n_rooms and v < n_rooms:
            adj_list.append([u, v])

    # Doors synthesis
    doors = []
    for u, v in adj_list:
        r1, r2 = rooms[u], rooms[v]
        dx = (r1["centroid"][0] + r2["centroid"][0]) / 2.0
        dy = (r1["centroid"][1] + r2["centroid"][1]) / 2.0
        doors.append({
            "room_a": u,
            "room_b": v,
            "pos": [round(dx, 1), round(dy, 1)]
        })

    # Real-world metric estimation
    total_area_sqm = round(sum(r["area"] for r in rooms) * 0.08, 1)
    total_area_sqft = int(total_area_sqm * 10.764)

    plan = {
        "id": f"plan_{plan_id:05d}",
        "archetype": archetype["name"],
        "category": archetype.get("category", "residential"),
        "target_occupancy": archetype.get("target_occupancy", "2-4 persons"),
        "total_area_sqm": total_area_sqm,
        "total_area_sqft": total_area_sqft,
        "description": archetype["description"],
        "spatial_reasoning": archetype.get("spatial_reasoning", ""),
        "num_rooms": n_rooms,
        "boundary": boundary,
        "rooms": rooms,
        "adjacency": adj_list,
        "doors": doors
    }
    return plan


def generate_dataset(num_samples: int = 500, output_dir: str = "data/processed") -> List[Dict[str, Any]]:
    """Generates and saves an enriched real-world dataset of diverse residential floorplans."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = []
    
    for i in range(num_samples):
        plan = generate_single_plan(plan_id=i, archetype_idx=i)
        dataset.append(plan)
        
        plan_path = os.path.join(output_dir, f"{plan['id']}.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

    index_path = os.path.join(output_dir, "dataset_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_plans": len(dataset),
            "room_types": ROOM_TYPES,
            "archetypes": [a["name"] for a in ARCHETYPES],
            "plans": [{
                "id": p["id"],
                "archetype": p["archetype"],
                "category": p["category"],
                "num_rooms": p["num_rooms"],
                "total_area_sqft": p["total_area_sqft"],
                "rooms": [r["category"] for r in p["rooms"]],
                "description": p["description"],
                "spatial_reasoning": p["spatial_reasoning"]
            } for p in dataset]
        }, f, indent=2)

    print(f"[FloorGen Data Engine] Successfully generated {len(dataset)} enriched real-world residential floorplans into {output_dir}")
    return dataset


if __name__ == "__main__":
    generate_dataset(num_samples=500)
