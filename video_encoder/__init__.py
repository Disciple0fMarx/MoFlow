"""Video encoder module for ETH/UCY trajectory prediction (Introvert-style data).

Heavy submodules (encoder, datasets) are imported lazily to keep `import video_encoder`
usable without torch installed (e.g. for splits-only usage and unit tests).
"""
from .scenes import CANONICAL_SCENES, scene_to_folder, scene_to_raw_txt
from .splits import build_loso_splits, temporal_split

__version__ = "0.1.0"
__all__ = [
    "CANONICAL_SCENES",
    "scene_to_folder",
    "scene_to_raw_txt",
    "build_loso_splits",
    "temporal_split",
]


def __getattr__(name):
    if name in ("GlobalVideoEncoder", "BaseFrameEncoder"):
        from . import encoder as _e
        return getattr(_e, name)
    if name == "AgentVideoEncoder":
        from . import agent_encoder as _ae
        return getattr(_ae, name)
    if name == "GlobalFrameDataset":
        from . import datasets as _d
        return _d.GlobalFrameDataset
    raise AttributeError(name)
