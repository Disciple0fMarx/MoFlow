import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from utils.eth_ucy_geometry import (
    SCENE_RESOLUTIONS,
    annotation_step,
    assert_temporal_alignment,
    observed_frame_ids,
    scene_video_path,
    temporal_anchor_frame,
    video_index_for_frame,
    world_to_pixel as project_world_to_pixel,
)
from utils.normalization import normalize_min_max
from utils.norm_bounds import (
    NormBoundsError,
    bounds_from_cfg,
    maybe_persist_bounds_from_cfg,
)
from utils.video_reader import get_video_reader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from video_encoder.moflow_adapter import FrameFeatureLookup

# TU-Clausthal raw parser (Kaggle-ready, no local pickle dependency)
try:
    from data.tu_clausthal_eth_ucy_parser import (
        TU_CLAUSTHAL_MAP,
        find_tu_clausthal_root,
        load_loso_windows,
        load_scene_data,
    )
except Exception:
    try:
        from tu_clausthal_eth_ucy_parser import (
            TU_CLAUSTHAL_MAP,
            find_tu_clausthal_root,
            load_loso_windows,
            load_scene_data,
        )
    except Exception:
        find_tu_clausthal_root = None
        load_scene_data = None
        load_loso_windows = None
        TU_CLAUSTHAL_MAP = {}


def seq_collate_eth(batch):
    (
        index,
        past_traj,
        fut_traj,
        past_traj_orig,
        fut_traj_orig,
        traj_vel,
        z_video,
        agent_crops,
    ) = zip(*batch)
    pre_motion_3D = torch.stack(past_traj, dim=0)
    fut_motion_3D = torch.stack(fut_traj, dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel_stack = torch.stack(traj_vel, dim=0)

    # z_video may be a degenerate `torch.zeros(1)` when USE_VIDEO=False — detect.
    if z_video[0].numel() > 1 and z_video[0].dim() >= 1:
        # Could be [T_obs, D_raw] or [D_raw]
        try:
            z_video_stack = torch.stack(z_video, dim=0)
            if z_video_stack.dim() == 3:  # [B, T_obs, D_raw]
                pass
            elif z_video_stack.dim() == 2:  # [B, D_raw] - rare but handle
                pass
        except Exception:
            z_video_stack = None
    else:
        z_video_stack = None

    # agent_crops may be a degenerate `torch.zeros(1)` when USE_AGENT_VIDEO=False.
    if agent_crops[0].numel() > 1 and agent_crops[0].dim() >= 4:
        agent_crops_stack = torch.stack(agent_crops, dim=0)
        if agent_crops_stack.dim() == 5:  # [B, P, 3, h, w]
            agent_crops_stack = agent_crops_stack.unsqueeze(1)  # [B, A=1, P, ...]
        if agent_crops_stack.dim() != 6:
            raise ValueError(
                "agent_crops must be [A, P, 3, h, w] or [P, 3, h, w] per sample; "
                f"got stacked shape {tuple(agent_crops_stack.shape)}"
            )
    else:
        agent_crops_stack = None

    data_dict = {
        "batch_size": torch.tensor(pre_motion_3D.shape[0]),
        "index": torch.cat(index, dim=0),
        "past_traj": pre_motion_3D,
        "fut_traj": fut_motion_3D,
        "past_traj_original_scale": pre_motion_3D_orig,
        "fut_traj_original_scale": fut_motion_3D_orig,
        "fut_traj_vel": fut_traj_vel_stack,
    }
    if z_video_stack is not None:
        data_dict["z_video_global"] = z_video_stack
    if agent_crops_stack is not None:
        data_dict["agent_crops"] = agent_crops_stack
    return data_dict


def seq_collate_imle_train(batch):
    # Removed video_frames unpacking to reflect the optimized lightweight pipeline
    (
        past_traj,
        fut_traj,
        past_traj_orig,
        fut_traj_orig,
        traj_vel,
        y_t,
        y_pred_data,
        agent_crops,
    ) = zip(*batch)

    pre_motion_3D = torch.stack(past_traj, dim=0)
    fut_motion_3D = torch.stack(fut_traj, dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    y_t = torch.stack(y_t, dim=0)
    y_pred_data = torch.stack(y_pred_data, dim=0)

    # agent_crops may be a degenerate `torch.zeros(1)` when USE_AGENT_VIDEO=False.
    if agent_crops[0].numel() > 1 and agent_crops[0].dim() >= 4:
        agent_crops_stack = torch.stack(agent_crops, dim=0)
        if agent_crops_stack.dim() == 5:  # [B, P, 3, h, w]
            agent_crops_stack = agent_crops_stack.unsqueeze(1)  # [B, A=1, P, ...]
        if agent_crops_stack.dim() != 6:
            raise ValueError(
                "agent_crops must be [A, P, 3, h, w] or [P, 3, h, w] per sample; "
                f"got stacked shape {tuple(agent_crops_stack.shape)}"
            )
    else:
        agent_crops_stack = None

    batch_size = torch.tensor(pre_motion_3D.shape[0])  ### bt
    data = {
        "batch_size": batch_size,
        "past_traj": pre_motion_3D,
        "fut_traj": fut_motion_3D,
        "past_traj_original_scale": pre_motion_3D_orig,
        "fut_traj_original_scale": fut_motion_3D_orig,
        "fut_traj_vel": fut_traj_vel,
        "y_t": y_t,
        "y_pred_data": y_pred_data,
    }
    if agent_crops_stack is not None:
        data["agent_crops"] = agent_crops_stack

    return data


def rotate_traj(
    past_rel, future_rel, past_abs, agents=2, rotate_time_frame=0, subset="eth"
):
    past_rel = rearrange(past_rel, "b a p d -> (b a) p d")
    past_abs = rearrange(past_abs, "b a p d -> (b a) p d")
    future_rel = rearrange(future_rel, "b a f d -> (b a) f d")
    past_diff = past_rel[:, rotate_time_frame]

    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-5))
    past_theta = torch.where((past_diff[:, 0] < 0), past_theta + math.pi, past_theta)

    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = -torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(
        1, 2
    )
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(
        1, 2
    )
    past_after = rearrange(past_after, "(b a) p d -> b a p d", a=agents)
    future_after = rearrange(future_after, "(b a) f d -> b a f d", a=agents)
    past_abs_after = rearrange(past_abs_after, "(b a) p d -> b a p d", a=agents)

    return past_after, future_after, past_abs_after


class ETHDataset(object):
    def __init__(
        self,
        cfg,
        training=True,
        data_dir=None,
        subset=None,
        rotate_time_frame=0,
        imle=False,
        type="original",
        video_mode=None,
    ):
        self.type = type
        self.imle = imle
        self.data_dir = data_dir
        self.subset = subset
        self.frame_ids = []
        self.video_mode = (
            video_mode if video_mode is not None else getattr(cfg, "VIDEO_MODE", None)
        )
        if (
            self.video_mode is None
            and hasattr(cfg, "MODEL")
            and hasattr(cfg.MODEL, "CONTEXT_ENCODER")
        ):
            self.video_mode = getattr(cfg.MODEL.CONTEXT_ENCODER, "VIDEO_MODE", None)
        if self.video_mode is None:
            self.video_mode = (
                "full"
                if bool(
                    getattr(
                        getattr(cfg, "MODEL", object()), "CONTEXT_ENCODER", object()
                    )
                    and getattr(cfg.MODEL.CONTEXT_ENCODER, "USE_VIDEO", False)
                )
                else "off"
            )
        # Ensure common cfg attrs expected by some trainers/tools
        if not hasattr(cfg, "rotate"):
            cfg.rotate = False
        if not hasattr(cfg, "data_norm"):
            cfg.data_norm = "min_max"
        if not hasattr(cfg, "past_frames"):
            cfg.past_frames = 8
        if not hasattr(cfg, "future_frames"):
            cfg.future_frames = 12
        if not hasattr(cfg, "agents"):
            cfg.agents = 1
        if not hasattr(cfg, "MODEL"):

            class _Dummy:
                pass

            cfg.MODEL = _Dummy()
        if not hasattr(cfg.MODEL, "CONTEXT_ENCODER"):

            class _Dummy2:
                pass

            cfg.MODEL.CONTEXT_ENCODER = _Dummy2()
        if not hasattr(cfg.MODEL.CONTEXT_ENCODER, "AGENTS"):
            cfg.MODEL.CONTEXT_ENCODER.AGENTS = getattr(cfg, "agents", 1)
        cfg.VIDEO_MODE = self.video_mode
        try:
            cfg.MODEL.CONTEXT_ENCODER.VIDEO_MODE = self.video_mode
        except Exception:
            pass

        ### LED version of preprocessed data
        if self.type == "LED":
            data_file_path = os.path.join(
                data_dir,
                self.type,
                "{:s}_data_{:s}.npy".format(subset, "train" if training else "test"),
            )
            num_file_path = os.path.join(
                data_dir,
                self.type,
                "{:s}_num_{:s}.npy".format(subset, "train" if training else "test"),
            )

            all_data = np.load(data_file_path)
            all_num = np.load(num_file_path)

            # === SET self.frame_ids FOR LED ===
            self.frame_ids = all_num[:, 0, 0]

            self.all_data = torch.Tensor(all_data)
            self.all_num = torch.Tensor(all_num)
        elif self.type == "original":
            if load_loso_windows is None:
                raise ImportError(
                    "Raw TU-Clausthal parser is unavailable; cannot load ETH/UCY."
                )
            raw = load_loso_windows(
                data_root=data_dir,
                held_out=subset,
                past_frames=int(cfg.past_frames),
                future_frames=int(cfg.future_frames),
            )
            split = raw["train"] if training else raw["test"]
            all_data = split["windows"]
            if all_data.size == 0:
                raise ValueError(
                    f"No raw windows for subset={subset}, training={training}"
                )
            self.frame_ids = np.asarray(split["start_frames"], dtype=np.int64)
            self.ped_ids = np.asarray(split["ped_ids"], dtype=np.int64)
            self.scene_by_sample = np.asarray(split["scenes"], dtype=object)
            self.all_data = torch.Tensor(all_data)
            self.all_data = self.all_data[:, None, :, :]  # [N, A=1, T, 2]
        else:
            raise ValueError("Invalid type")

        self.cfg = cfg
        self.rotate_time_frame = rotate_time_frame
        self.imle = imle

        ### set the agent_num in the cfg
        cfg.agents = self.all_data.shape[1]
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = cfg.agents

        ### compute past and future trajectories
        past_traj_abs = self.all_data[:, :, : cfg.past_frames]
        initial_pos = past_traj_abs[:, :, -1:]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj = (self.all_data[:, :, cfg.past_frames :] - initial_pos).contiguous()
        if cfg.rotate:
            past_traj_rel, fut_traj, past_traj_abs = rotate_traj(
                past_traj_rel,
                fut_traj,
                past_traj_abs,
                cfg.agents,
                rotate_time_frame,
                subset,
            )
        past_traj_vel = torch.cat(
            (
                past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1],
                torch.zeros_like(past_traj_rel[:, :, -1:]),
            ),
            dim=2,
        )
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat(
            (
                fut_traj[:, :, 1:] - fut_traj[:, :, :-1],
                torch.zeros_like(fut_traj[:, :, -1:]),
            ),
            dim=2,
        )

        self.rotate_aug = cfg.rotate and training

        if training:
            cfg.fut_traj_max = fut_traj.max()
            cfg.fut_traj_min = fut_traj.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()
            # Persist the exact training bounds so evaluation/visualization
            # processes never have to guess (no [-1, 1] fallback).
            maybe_persist_bounds_from_cfg(cfg)
        elif getattr(cfg, "data_norm", "min_max") == "min_max":
            # Strict: evaluation/viz configs must carry bounds loaded from the
            # training artifact.  Missing bounds raise instead of defaulting.
            bounds_from_cfg(cfg, source="evaluation config")

        ### record the original to avoid numerical errors
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj

        self.use_video = bool(getattr(cfg.MODEL.CONTEXT_ENCODER, "USE_VIDEO", False))
        if self.video_mode == "off":
            self.use_video = False
        elif self.video_mode in {"static", "full"}:
            self.use_video = True
        try:
            cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = self.use_video
        except Exception:
            pass
        self.use_agent_video = bool(
            getattr(cfg.MODEL.CONTEXT_ENCODER, "USE_AGENT_VIDEO", False)
        )
        self.agent_crop_size = getattr(
            cfg.MODEL.CONTEXT_ENCODER, "AGENT_CROP_SIZE", [64, 64]
        )
        self.z_video_global = None
        if self.use_video:
            if not hasattr(self, "scene_by_sample") or self.scene_by_sample is None:
                self.scene_by_sample = np.asarray(
                    [subset] * self.all_data.shape[0], dtype=object
                )
            scenes = np.asarray(self.scene_by_sample, dtype=object)
            if len(scenes) != self.all_data.shape[0]:
                raise ValueError(
                    f"scene_by_sample length {len(scenes)} != dataset length "
                    f"{self.all_data.shape[0]}"
                )
            start_fids = np.asarray(self.frame_ids, dtype=np.int64)
            if len(start_fids) != self.all_data.shape[0]:
                raise ValueError(
                    f"frame_ids length {len(start_fids)} != dataset length "
                    f"{self.all_data.shape[0]}"
                )

            features_root = getattr(
                cfg.MODEL.CONTEXT_ENCODER, "VIDEO_FEATURES_ROOT", "features/resnet18"
            )
            lookups = {}
            for scene_name in np.unique(scenes):
                scene_name = str(scene_name)
                try:
                    lookups[scene_name] = FrameFeatureLookup.from_root(
                        features_root, scene=scene_name
                    )
                except Exception as e:
                    raise FileNotFoundError(
                        f"USE_VIDEO=True but video features are missing for scene "
                        f"'{scene_name}' under {features_root}: {e}. "
                        "Build the per-scene feature cache before training."
                    ) from e

            T_obs = int(cfg.past_frames)
            D = next(iter(lookups.values())).features.shape[1]
            z = np.zeros((self.all_data.shape[0], T_obs, D), dtype=np.float32)
            if self.video_mode == "static":
                static_by_scene = {}
                for scene_name, lookup in lookups.items():
                    static = np.zeros(D, dtype=np.float32)
                    ids = lookup.sorted_frame_ids
                    if ids.size:
                        sample = ids[:: max(1, ids.size // 8)][:8]
                        static = np.stack(
                            [lookup(int(f)) for f in sample], axis=0
                        ).mean(axis=0).astype(np.float32)
                    static_by_scene[scene_name] = static
                for i, scene_name in enumerate(scenes):
                    z[i] = np.tile(
                        static_by_scene[str(scene_name)][None, :], (T_obs, 1)
                    )
            else:
                coverage = {}
                steps: dict[str, int] = {}
                for scene_name, lookup in lookups.items():
                    step = annotation_step(scene_name)
                    steps[scene_name] = step
                    available = set(int(x) for x in lookup.sorted_frame_ids)
                    scene_mask = scenes == scene_name
                    required = np.unique(
                        (
                            start_fids[scene_mask, None]
                            + np.arange(T_obs)[None, :] * step
                        ).ravel()
                    )
                    missing = np.asarray(
                        [f for f in required if int(f) not in available],
                        dtype=np.int64,
                    )
                    if missing.size:
                        coverage[scene_name] = (
                            int(missing.size),
                            int(missing.min()),
                            int(missing.max()),
                        )
                if coverage:
                    raise FileNotFoundError(
                        "Video feature cache does not cover every raw frame "
                        f"needed by this split under '{features_root}'. "
                        f"Missing per scene (count,min,max): {coverage}. "
                        "Rebuild features at each scene's annotation step "
                        "(see video_encoder/scripts/build_ethucy_features.py)."
                    )
                for i, (fid, scene_name) in enumerate(zip(start_fids, scenes)):
                    step = steps[str(scene_name)]
                    z[i] = lookups[str(scene_name)].window(
                        int(fid), n_frames=T_obs, stride=step, policy="nearest"
                    )
            self.z_video_global = torch.from_numpy(z)

        ### min-max linear normalization
        if cfg.data_norm == "min_max":
            self.past_traj = normalize_min_max(
                past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1
            ).contiguous()
            self.fut_traj = normalize_min_max(
                fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
            ).contiguous()
        elif cfg.data_norm == "original":
            self.past_traj = past_traj
            self.fut_traj = fut_traj

        """IMLE distillation targets are disabled with the raw-only loader."""
        if imle:
            raise NotImplementedError(
                "IMLE pkl distillation targets are disabled. The ETH/UCY loader "
                "is raw-only; no .pkl files are read."
            )

        self.subset = subset

        # Authoritative scene geometry lives in utils/eth_ucy_geometry.py.
        self.SCENE_RESOLUTIONS = dict(SCENE_RESOLUTIONS)

        self.SCENE_WORLD_BOUNDS = {
            "eth": (-7.69, 13.89, -1.81, 12.67),
            "hotel": (-10.31, 4.31, -2.77, 4.04),
            "univ": (-0.46, 15.47, -0.32, 13.89),
            "zara1": (-0.14, 15.48, -0.37, 12.39),
            "zara2": (-0.36, 15.56, -0.19, 13.48),
        }

        # Load raw H.txt directly from every scene present in this dataset.
        if load_scene_data is None:
            raise ImportError(
                "Raw TU-Clausthal parser is unavailable; cannot load H.txt."
            )
        self.scene_H = {}
        scene_names = getattr(self, "scene_by_sample", None)
        if scene_names is None:
            scene_names = np.asarray([subset] * self.all_data.shape[0], dtype=object)
        for scene_name in np.unique(scene_names):
            scene_name = str(scene_name)
            self.scene_H[scene_name] = load_scene_data(data_dir, scene_name)["H"]
        if subset not in self.scene_H:
            self.scene_H[subset] = load_scene_data(data_dir, subset)["H"]
        self.H = self.scene_H[subset]
        self.H_inv = np.linalg.inv(self.H)
        self.orig_res = self.SCENE_RESOLUTIONS.get(subset, (720, 576))

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, item):
        if self.imle:
            out = [
                self.imle_data_dict["past_traj"][item],
                self.imle_data_dict["fut_traj"][item],
                self.imle_data_dict["past_traj_original_scale"][item],
                self.imle_data_dict["fut_traj_original_scale"][item],
                self.imle_data_dict["fut_traj_vel"][item],
                self.imle_data_dict["y_t"][item],
                self.imle_data_dict["y_pred_data"][item],
            ]
            start_frame = int(self.imle_data_dict["start_frame"][item])
        else:
            past_traj_norm_scale = self.past_traj[item]  # [A, P, 6]
            fut_traj_norm_scale = self.fut_traj[item]  # [A, F, 2]
            past_traj_original_scale = self.past_traj_original_scale[item]  # [A, P, 6]
            fut_traj_original_scale = self.fut_traj_original_scale[item]  # [A, F, 2]
            fut_traj_vel = self.fut_traj_vel[item]  # [A, F, 2]

            if self.rotate_aug:
                A = past_traj_norm_scale.size(0)
                rot_angle = torch.rand(A) * 2 * math.pi

                past_traj_abs_o, past_traj_rel_o, past_traj_vel_o = (
                    past_traj_original_scale.chunk(3, dim=-1)
                )  # [A, P, 2] * 3

                rotate_matrix = torch.zeros((rot_angle.size(0), 2, 2)).to(
                    past_traj_norm_scale.device
                )  # [A, 2, 2]
                rotate_matrix[:, 0, 0] = torch.cos(rot_angle)
                rotate_matrix[:, 0, 1] = torch.sin(rot_angle)
                rotate_matrix[:, 1, 0] = -torch.sin(rot_angle)
                rotate_matrix[:, 1, 1] = torch.cos(rot_angle)

                past_traj_abs_o_rot = torch.matmul(
                    rotate_matrix, past_traj_abs_o.transpose(1, 2)
                ).transpose(1, 2)
                past_traj_rel_o_rot = torch.matmul(
                    rotate_matrix, past_traj_rel_o.transpose(1, 2)
                ).transpose(1, 2)
                fut_traj_o_rot = torch.matmul(
                    rotate_matrix, fut_traj_original_scale.transpose(1, 2)
                ).transpose(1, 2)

                past_traj_vel_o_rot = torch.cat(
                    (
                        past_traj_rel_o_rot[:, :, 1:] - past_traj_rel_o_rot[:, :, :-1],
                        torch.zeros_like(past_traj_rel_o_rot[:, :, -1:]),
                    ),
                    dim=2,
                )
                past_traj_rot = torch.cat(
                    (past_traj_abs_o_rot, past_traj_rel_o_rot, past_traj_vel_o_rot),
                    dim=-1,
                )

                fut_traj_vel_o = torch.cat(
                    (
                        fut_traj_o_rot[:, :, 1:] - fut_traj_o_rot[:, :, :-1],
                        torch.zeros_like(fut_traj_o_rot[:, :, -1:]),
                    ),
                    dim=2,
                )

                past_traj_norm_scale = normalize_min_max(
                    past_traj_rot, self.cfg.past_traj_min, self.cfg.past_traj_max, -1, 1
                ).contiguous()
                fut_traj_norm_scale = normalize_min_max(
                    fut_traj_o_rot, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1
                ).contiguous()
                past_traj_original_scale = past_traj_rot
                fut_traj_original_scale = fut_traj_o_rot
                fut_traj_vel = fut_traj_vel_o

            # Extract agent-centric video crops if enabled
            agent_crops = None
            if self.use_agent_video:
                scene = (
                    str(self.scene_by_sample[item])
                    if getattr(self, "scene_by_sample", None) is not None
                    else str(self.subset)
                )
                self.current_scene = scene
                agent_crops = self._extract_agent_crops(
                    past_traj_original_scale,
                    start_frame=int(self.frame_ids[item]),
                    scene=scene,
                )

            z_video = (
                self.z_video_global[item]
                if self.z_video_global is not None
                else torch.zeros(1)
            )
            out = [
                torch.Tensor([item]).to(torch.int32),
                past_traj_norm_scale,
                fut_traj_norm_scale,
                past_traj_original_scale,
                fut_traj_original_scale,
                fut_traj_vel,
                z_video,
                agent_crops if agent_crops is not None else torch.zeros(1),
            ]

        return out

    def world_to_pixel(self, traj_pts, target_res=(224, 224), scene=None):
        """Project world metres to pixels via the single authoritative mapping."""
        scene = scene or self.subset
        H = self.scene_H[scene] if hasattr(self, "scene_H") else self.H
        return project_world_to_pixel(
            np.asarray(traj_pts, dtype=np.float64),
            H,
            scene=scene,
            target_res=target_res,
        )

    world_to_pixel_backup = world_to_pixel  # kept for compatibility

    def _extract_agent_crops(
        self,
        past_traj_original_scale,
        start_frame: int | None = None,
        scene: str | None = None,
    ):
        """Extract agent-centric crops directly from the native ``video.avi``.

        ``video.avi`` is the sole visual source: no pre-extracted frame dumps,
        no frame-directory fallbacks, and no nearest-frame substitution.  Frames
        are read at the exact scene annotation stride for the observed window.

        Args:
            past_traj_original_scale: ``[A, P, 6]`` with absolute world ``xy`` first.
            start_frame: first observed annotation frame id for this sample.
            scene: canonical scene name (defaults to the current sample's scene).

        Returns:
            ``[A, P, 3, h, w]`` RGB crops, or ``None`` when agent video is off.
        """
        if not self.use_agent_video:
            return None

        if start_frame is None:
            raise ValueError(
                "start_frame is required for agent-centric video cropping; "
                "refusing to guess (previous behaviour used frame 0)."
            )
        if not getattr(self, "data_dir", None):
            raise FileNotFoundError(
                "ETHDataset.data_dir must point at the TU-Clausthal root to "
                "extract agent crops from video.avi."
            )

        scene = scene or getattr(self, "current_scene", self.subset)
        A, P, _ = past_traj_original_scale.shape
        h, w = self.agent_crop_size
        frame_ids = observed_frame_ids(int(start_frame), scene, P)
        anchor = assert_temporal_alignment(
            int(start_frame), scene, P, frame_ids=frame_ids
        )
        if frame_ids[-1] != anchor:
            raise AssertionError(
                f"Temporal anchor mismatch for scene={scene}: {frame_ids[-1]} != {anchor}"
            )

        video_path = scene_video_path(self.data_dir, scene)
        reader = get_video_reader(video_path)
        native_w, native_h = reader.size if reader.size is not None else (None, None)
        if native_w is None or native_h is None:
            raise ValueError(f"Could not determine native video size for {video_path}")

        H = self.scene_H[str(scene)] if hasattr(self, "scene_H") else self.H
        world_xy = (
            past_traj_original_scale[:, :, :2].detach().cpu().numpy().reshape(-1, 2)
        )
        pixels = project_world_to_pixel(
            world_xy, H, scene=scene, target_res=(native_w, native_h)
        ).reshape(A, P, 2)

        crops = torch.zeros(A, P, 3, h, w, dtype=torch.float32)
        for p in range(P):
            frame_id = int(frame_ids[p])
            video_index = video_index_for_frame(scene, frame_id)
            frame = reader.read(video_index, color="rgb")
            if frame.shape[0] != native_h or frame.shape[1] != native_w:
                raise ValueError(
                    f"Decoded frame {video_index} has shape {frame.shape[:2]}, "
                    f"expected {(native_h, native_w)} for {video_path}"
                )
            for a in range(A):
                px = int(round(float(pixels[a, p, 0])))
                py = int(round(float(pixels[a, p, 1])))
                x1, y1 = px - w // 2, py - h // 2
                x2, y2 = x1 + w, y1 + h

                canvas = np.zeros((h, w, 3), dtype=frame.dtype)
                sx1, sy1 = max(0, x1), max(0, y1)
                sx2, sy2 = min(native_w, x2), min(native_h, y2)
                if sx2 > sx1 and sy2 > sy1:
                    dx1, dy1 = sx1 - x1, sy1 - y1
                    canvas[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = frame[
                        sy1:sy2, sx1:sx2
                    ]
                crops[a, p] = (
                    torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
                )

        return crops


class ETHDatasetSocialGAN:
    """Removed: pickle generation is disabled.

    The ETH/UCY pipeline now reads raw TU-Clausthal files exclusively. This
    stub exists only so old imports fail with an actionable message.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ETHDatasetSocialGAN is disabled. The ETH/UCY pipeline no longer "
            "creates or reads .pkl files; use ETHDataset with raw data."
        )
