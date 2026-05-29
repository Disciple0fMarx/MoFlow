"""Canonical ETH/UCY scene names and mappings to Introvert's on-disk layout."""
from __future__ import annotations

CANONICAL_SCENES = ("eth", "hotel", "univ", "zara1", "zara2")

# canonical -> folder name inside data_trajpred/
_FOLDER_MAP = {
    "eth": "eth",
    "hotel": "hotel",
    "univ": "university",
    "zara1": "zara_01",
    "zara2": "zara_02",
}

# canonical -> base filename inside data_trajpred/raw/all_data/
# Standard ETH/UCY split: univ uses students003; uni_examples, students001, zara03 are excluded.
_RAW_TXT_MAP = {
    "eth": "biwi_eth",
    "hotel": "biwi_hotel",
    "univ": "students003",
    "zara1": "crowds_zara01",
    "zara2": "crowds_zara02",
}


def scene_to_folder(scene: str) -> str:
    if scene not in _FOLDER_MAP:
        raise ValueError(f"Unknown scene '{scene}'. Expected one of {CANONICAL_SCENES}.")
    return _FOLDER_MAP[scene]


def scene_to_raw_txt(scene: str) -> str:
    if scene not in _RAW_TXT_MAP:
        raise ValueError(f"Unknown scene '{scene}'. Expected one of {CANONICAL_SCENES}.")
    return _RAW_TXT_MAP[scene]


def others(leave_out: str) -> tuple[str, ...]:
    if leave_out not in CANONICAL_SCENES:
        raise ValueError(f"Unknown scene '{leave_out}'. Expected one of {CANONICAL_SCENES}.")
    return tuple(s for s in CANONICAL_SCENES if s != leave_out)
