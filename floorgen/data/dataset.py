"""
PyTorch Dataset and DataLoaders for FloorGen models.
Provides unified batching for:
- Room categories (categorical token IDs)
- Room polygon vertices and bounding boxes (continuous coordinates in [-1, 1])
- Adjacency matrices (bubble graph connectivity)
- Retrieval conditioning context (RAG top-k exemplars)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional

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
NUM_ROOM_TYPES = len(ROOM_TYPES)


def normalize_coords(val: float, min_val: float = 0.0, max_val: float = 256.0) -> float:
    """Normalizes a coordinate from [min_val, max_val] to [-1.0, 1.0]."""
    return 2.0 * ((val - min_val) / (max_val - min_val)) - 1.0


def denormalize_coords(val: float, min_val: float = 0.0, max_val: float = 256.0) -> float:
    """Denormalizes a coordinate from [-1.0, 1.0] to [min_val, max_val]."""
    return min_val + (val + 1.0) / 2.0 * (max_val - min_val)


class FloorplanDataset(Dataset):
    """
    PyTorch Dataset for FloorGen floorplans.
    Each item contains:
      - room_types: (max_rooms,) int64 tensor of room category IDs
      - room_boxes: (max_rooms, 4) float32 tensor of [xmin, ymin, xmax, ymax] in [-1, 1]
      - room_polys: (max_rooms, max_vertices, 2) float32 tensor of polygon vertices
      - room_mask:  (max_rooms,) bool tensor (True for valid rooms, False for padding)
      - adj_matrix: (max_rooms, max_rooms) float32 tensor of bubble graph adjacency
      - boundary:   (4,) float32 tensor of site boundary bbox
    """

    def __init__(
        self,
        plans: List[Dict[str, Any]],
        max_rooms: int = 12,
        max_vertices: int = 4,
        coord_range: float = 256.0
    ):
        self.plans = plans
        self.max_rooms = max_rooms
        self.max_vertices = max_vertices
        self.coord_range = coord_range

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        plan = self.plans[idx]
        rooms = plan.get("rooms", [])
        num_rooms = min(len(rooms), self.max_rooms)

        # Allocate tensors
        room_types = torch.zeros(self.max_rooms, dtype=torch.long)
        room_boxes = torch.zeros((self.max_rooms, 4), dtype=torch.float32)
        room_polys = torch.zeros((self.max_rooms, self.max_vertices, 2), dtype=torch.float32)
        room_mask = torch.zeros(self.max_rooms, dtype=torch.bool)
        adj_matrix = torch.zeros((self.max_rooms, self.max_rooms), dtype=torch.float32)

        for i in range(num_rooms):
            r = rooms[i]
            cat_name = r.get("category", "living_room")
            cat_id = ROOM_TYPE_TO_ID.get(cat_name, 0)
            room_types[i] = cat_id
            room_mask[i] = True

            # Normalized bounding box [xmin, ymin, xmax, ymax]
            bbox = r.get("bbox", [0, 0, 50, 50])
            norm_box = [normalize_coords(v, 0, self.coord_range) for v in bbox]
            room_boxes[i] = torch.tensor(norm_box, dtype=torch.float32)

            # Normalized polygon vertices
            poly = r.get("polygon", [])
            for v_idx in range(min(len(poly), self.max_vertices)):
                vx = normalize_coords(poly[v_idx][0], 0, self.coord_range)
                vy = normalize_coords(poly[v_idx][1], 0, self.coord_range)
                room_polys[i, v_idx] = torch.tensor([vx, vy], dtype=torch.float32)

        # Adjacency matrix
        for edge in plan.get("adjacency", []):
            u, v = edge
            if u < self.max_rooms and v < self.max_rooms:
                adj_matrix[u, v] = 1.0
                adj_matrix[v, u] = 1.0

        # Boundary box
        b_poly = plan.get("boundary", [[0, 0], [256, 0], [256, 256], [0, 256]])
        xs = [p[0] for p in b_poly]
        ys = [p[1] for p in b_poly]
        boundary = torch.tensor([
            normalize_coords(min(xs), 0, self.coord_range),
            normalize_coords(min(ys), 0, self.coord_range),
            normalize_coords(max(xs), 0, self.coord_range),
            normalize_coords(max(ys), 0, self.coord_range)
        ], dtype=torch.float32)

        return {
            "room_types": room_types,
            "room_boxes": room_boxes,
            "room_polys": room_polys,
            "room_mask": room_mask,
            "adj_matrix": adj_matrix,
            "boundary": boundary,
            "plan_id": plan.get("id", f"plan_{idx}")
        }


def create_dataloader(
    plans: List[Dict[str, Any]],
    batch_size: int = 16,
    shuffle: bool = True,
    max_rooms: int = 12
) -> DataLoader:
    """Factory to create a DataLoader for training or evaluation."""
    dataset = FloorplanDataset(plans=plans, max_rooms=max_rooms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
