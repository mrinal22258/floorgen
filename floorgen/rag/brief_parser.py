"""
Architectural Natural-Language Brief Parser and Constraint Reasoner for FloorGen.
Inspired by HouseTune (Paraiso et al., 2023).
Parses unstructured client text briefs into structured room inventories,
adjacency affinities, and area constraints for the diffusion core.
"""

import re
from typing import Dict, Any, List, Set, Tuple, Optional


ROOM_KEYWORDS = {
    "living_room": ["living", "lounge", "salon", "family room", "great room", "drawing room"],
    "master_bedroom": ["master bedroom", "primary bedroom", "master suite", "parents room", "main bedroom"],
    "second_bedroom": ["second bedroom", "secondary bedroom", "kids room", "children room", "guest bedroom", "guest room", "bedroom 2", "bedroom 3", "2nd bedroom", "3rd bedroom"],
    "bathroom": ["bathroom", "bath", "washroom", "toilet", "wc", "ensuite", "restroom", "powder room"],
    "kitchen": ["kitchen", "cooking", "galley", "kitchenette", "chef"],
    "balcony": ["balcony", "terrace", "deck", "veranda", "patio", "loggia"],
    "dining_room": ["dining", "dining room", "eating area", "dining space"],
    "entrance": ["entrance", "foyer", "entryway", "vestibule", "hallway", "corridor", "lobby"],
    "study": ["study", "office", "home office", "library", "den", "work room"],
    "storage": ["storage", "closet", "pantry", "utility", "laundry", "linen"]
}


class ParsedArchitecturalBrief:
    def __init__(
        self,
        original_brief: str,
        detected_rooms: List[str],
        affinity_pairs: List[Tuple[str, str]],
        estimated_sqft: Optional[int],
        style_keywords: List[str]
    ):
        self.original_brief = original_brief
        self.detected_rooms = detected_rooms
        self.affinity_pairs = affinity_pairs
        self.estimated_sqft = estimated_sqft
        self.style_keywords = style_keywords

    def to_dict(self) -> Dict[str, Any]:
        style = "open_plan" if "open_plan" in self.style_keywords else (self.style_keywords[0] if self.style_keywords else "standard")
        aspect = "spacious" if "spacious" in self.original_brief.lower() else "standard"
        return {
            "required_rooms": self.detected_rooms,
            "detected_rooms": self.detected_rooms,
            "num_rooms": len(self.detected_rooms),
            "affinity_pairs": [list(p) for p in self.affinity_pairs],
            "adjacency_constraints": [list(p) for p in self.affinity_pairs],
            "style_preference": style,
            "aspect_ratio_preference": aspect,
            "estimated_sqft": self.estimated_sqft,
            "style_keywords": self.style_keywords
        }

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.to_dict()



def parse_architectural_brief(brief_text: str) -> ParsedArchitecturalBrief:
    """
    Translates free-text architectural briefs into structured room lists and relational constraints.
    """
    text_lower = brief_text.lower()
    detected_rooms: List[str] = []
    affinities: List[Tuple[str, str]] = []
    style_tags: List[str] = []

    # 1. Detect Bedroom Counts
    bed_match = re.search(r'(\d+)\s*(-|\s)?(bed|bedroom|bhk|br)', text_lower)
    num_beds = int(bed_match.group(1)) if bed_match else 2

    # 2. Detect Bathroom Counts
    bath_match = re.search(r'(\d+)\s*(-|\s)?(bath|bathroom|ba)', text_lower)
    num_baths = int(bath_match.group(1)) if bath_match else 1

    # 3. Detect Square Footage
    sqft_match = re.search(r'(\d{3,5})\s*(sqft|sq\s*ft|square\s*feet|sqm|m2)', text_lower)
    est_sqft = int(sqft_match.group(1)) if sqft_match else None

    # Construct baseline room list based on detected counts
    detected_rooms.append("living_room")
    if num_beds >= 1:
        detected_rooms.append("master_bedroom")
    for b_i in range(1, num_beds):
        detected_rooms.append("second_bedroom")

    detected_rooms.append("kitchen")
    for b_i in range(num_baths):
        detected_rooms.append("bathroom")

    # Check for supplemental functional spaces
    if any(k in text_lower for k in ROOM_KEYWORDS["balcony"]):
        detected_rooms.append("balcony")
    if any(k in text_lower for k in ROOM_KEYWORDS["dining_room"]):
        detected_rooms.append("dining_room")
    if any(k in text_lower for k in ROOM_KEYWORDS["study"]):
        detected_rooms.append("study")
    if any(k in text_lower for k in ROOM_KEYWORDS["entrance"]):
        detected_rooms.append("entrance")
    if any(k in text_lower for k in ROOM_KEYWORDS["storage"]):
        detected_rooms.append("storage")

    # 4. Detect Relational Affinities
    if "open kitchen" in text_lower or "kitchen to living" in text_lower or "connected kitchen" in text_lower:
        affinities.append(("kitchen", "living_room"))
    if "kitchen to dining" in text_lower or "formal dining" in text_lower:
        affinities.append(("kitchen", "dining_room"))
    if "master suite" in text_lower or "ensuite" in text_lower:
        affinities.append(("master_bedroom", "bathroom"))
    if "living" in text_lower and "balcony" in text_lower:
        affinities.append(("living_room", "balcony"))

    # Style detection
    styles = ["modern", "luxury", "minimalist", "open concept", "traditional", "compact", "duplex", "villa"]
    for s in styles:
        if s in text_lower:
            style_tags.append(s)
    if "open kitchen" in text_lower or "open plan" in text_lower or "open concept" in text_lower:
        style_tags.append("open_plan")


    return ParsedArchitecturalBrief(
        original_brief=brief_text,
        detected_rooms=detected_rooms,
        affinity_pairs=affinities,
        estimated_sqft=est_sqft,
        style_keywords=style_tags
    )
