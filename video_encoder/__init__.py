"""Video encoding utilities.

ETH/UCY visual inputs are read exclusively from native ``video.avi`` files via
the strict TU-Clausthal layout.  Pre-extracted frame directories are not
supported.  SDD helpers (which consume the dataset's native JPEG frames) live
in :mod:`video_encoder.sdd_adapter` and are outside the ETH/UCY pipeline.
"""
from .scenes import CANONICAL_SCENES, scene_to_folder, scene_to_raw_txt

__version__ = "0.2.0"
__all__ = [
    "CANONICAL_SCENES",
    "scene_to_folder",
    "scene_to_raw_txt",
]


def __getattr__(name):
    if name == "AgentVideoEncoder":
        from . import agent_encoder as _ae

        return getattr(_ae, name)
    raise AttributeError(name)
