import os
from glob import glob
import pickle
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
from einops import rearrange

from utils.normalization import normalize_min_max
from torch.nn.utils.rnn import pad_sequence

# Video encoder imports
try:
    from video_encoder.moflow_adapter import FrameFeatureLookup
except ImportError:
    FrameFeatureLookup = None  # fallback if video_encoder not available


def rotate_traj(past_rel, future_rel, past_abs, rotate_time_frame=0):
    """
    @params past_rel: [N, A, P, 2]
    @params future_rel: [N, A, F, 2]
    @params past_abs: [N, A, P, 2]
    @params rotate_time_frame: int
    """

    A = past_rel.size(1)
    past_rel = rearrange(past_rel, 'b a p d -> (b a) p d')
    past_abs = rearrange(past_abs, 'b a p d -> (b a) p d')
    future_rel = rearrange(future_rel, 'b a f d -> (b a) f d')

    past_diff = past_rel[:, rotate_time_frame]
    # past_diff = past[:, rotate_time_frame] - past[:, rotate_time_frame-1]

    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-5))
    past_theta = torch.where((past_diff[:, 0] < 0), past_theta + math.pi, past_theta)

    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = -torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)          # [N, P, 2]
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(1, 2)      # [N, F, 2]
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(1, 2)            # [N, P, 2]

    past_after = rearrange(past_after, '(b a) p d -> b a p d', a=A)
    future_after = rearrange(future_after, '(b a) f d -> b a f d', a=A)
    past_abs_after = rearrange(past_abs_after, '(b a) p d -> b a p d', a=A)

    return past_after, future_after, past_abs_after


def seq_collate_sdd(batch):
    (index, past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, z_video, agent_crops) = zip(*batch)
    indexes = torch.stack(index, dim=0)
    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)

    # Handle video features: stack if they are real tensors, else None
    if z_video[0].numel() > 1:  # Not a dummy scalar
        z_video_stack = torch.stack(z_video, dim=0)  # [B, T_obs, D_raw]
    else:
        z_video_stack = None

    # Agent crops: placeholder for now; treat similarly
    if agent_crops[0].numel() > 1:
        agent_crops_stack = torch.stack(agent_crops, dim=0)  # [B, A, P, 3, h, w]
    else:
        agent_crops_stack = None

    batch_size = torch.tensor(pre_motion_3D.shape[0]) ### bt
    data = {
        'indexes': indexes,
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,
    }
    if z_video_stack is not None:
        data['z_video_global'] = z_video_stack
    if agent_crops_stack is not None:
        data['agent_crops'] = agent_crops_stack
    return data


def seq_collate_imle_train(batch):
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, y_t, y_pred_data) = zip(*batch)

    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    y_t = torch.stack(y_t, dim=0)
    y_pred_data = torch.stack(y_pred_data,dim=0)

    batch_size = torch.tensor(pre_motion_3D.shape[0]) ### bt
    data = {
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,
        'y_t': y_t,
        'y_pred_data': y_pred_data
    }

    return data


class SDDDataset(Dataset):
    def __init__(self, cfg, data_dir,
                 training=True, overfit=False, rotate_time_frame=0, imle=False, subset=None):
        super(SDDDataset, self).__init__()

        """init"""
        self.cfg = cfg
        self.training = training
        self.overfit = overfit
        self.rotate_time_frame = rotate_time_frame
        self.imle = imle
        self.held_out_scene = subset  # None means use all scenes; if a scene name, it's the held-out test scene

        self.past_frames = cfg.past_frames
        self.future_frames = cfg.future_frames
        self.seq_len = self.past_frames + self.future_frames
        self.max_agents_per_scene = 0
        assert self.seq_len == 20 and self.past_frames == 8, "Sanity check on frame length failed!"

        # Define all possible scenes
        self.all_scenes = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']

        # Determine which scenes to use based on held_out_scene and overfit
        if self.overfit:
            # When overfitting, use all scenes for both train and test
            self.scenes_to_use = self.all_scenes
        else:
            if self.held_out_scene is None or self.held_out_scene == 'all':
                # Use all scenes
                self.scenes_to_use = self.all_scenes
            else:
                # held_out_scene is the scene to hold out for testing
                if self.training:
                    # Training set: all scenes except the held-out one
                    self.scenes_to_use = [s for s in self.all_scenes if s != self.held_out_scene]
                else:
                    # Test set: only the held-out scene
                    self.scenes_to_use = [self.held_out_scene]

        # Print info
        print(f"SDDDataset: Training={training}, Overfit={overfit}")
        print(f"  Held-out scene: {self.held_out_scene}")
        print(f"  Using scenes: {self.scenes_to_use}")

        # Load annotations and build dataset
        self._build_dataset(data_dir)

        # After building dataset, set up the data in the format expected by the rest of the code
        self._prepare_tensors()

        # Video settings (similar to original code)
        self.use_video = bool(getattr(cfg.MODEL.CONTEXT_ENCODER, 'USE_VIDEO', False))
        self.use_agent_video = bool(getattr(cfg.MODEL.CONTEXT_ENCODER, 'USE_AGENT_VIDEO', False))
        self.agent_crop_size = getattr(cfg.MODEL.CONTEXT_ENCODER, 'AGENT_CROP_SIZE', [64, 64])
        self.z_video_global = None
        if self.use_video:
            video_dim_raw = int(getattr(cfg.MODEL.CONTEXT_ENCODER, 'VIDEO_DIM_RAW', 512))
            # Load actual video features if available, otherwise create placeholder
            features_root = getattr(cfg.MODEL.CONTEXT_ENCODER, 'VIDEO_FEATURES_ROOT',
                                    'features/resnet18')
            # For SDD, we treat the entire dataset as one "scene" for video features
            scene_name = 'sdd'
            features_path = os.path.join(features_root, f"{scene_name}.npy")
            manifest_path = os.path.join(features_root, f"{scene_name}.manifest.parquet")

            if os.path.exists(features_path) and os.path.exists(manifest_path):
                # Load actual features
                try:
                    import pandas as pd
                    import numpy as np
                    from video_encoder.moflow_adapter import FrameFeatureLookup

                    lookup = FrameFeatureLookup.from_root(features_root, scene=scene_name)
                    T_obs = int(self.cfg.past_frames)
                    D = lookup.features.shape[1]
                    stride = int(getattr(cfg.MODEL.CONTEXT_ENCODER, 'VIDEO_STRIDE', 12))  # Default to 12 for SDD

                    # We need to map each sample to a start frame ID
                    # We stored the start frame in each sample during _build_dataset
                    start_fids = np.array([s[3] for s in self.samples_with_frames], dtype=np.int64)
                    z = np.zeros((len(self.samples_with_frames), T_obs, D), dtype=np.float32)
                    for i, fid in enumerate(start_fids):
                        z[i] = lookup.window(int(fid), n_frames=T_obs, stride=stride,
                                             policy='nearest')
                    self.z_video_global = torch.from_numpy(z)
                except Exception as e:
                    print(f"Warning: Failed to load video features ({e}). Using zero placeholders.")
                    self.z_video_global = torch.zeros((len(self.samples_with_frames), int(self.cfg.past_frames), video_dim_raw), dtype=torch.float32)
                    print(f"Warning: Using zero placeholder for video features. Shape: {self.z_video_global.shape}")
                    print(f"         Expected features at: {features_path}")
            else:
                # Create placeholder zero features
                self.z_video_global = torch.zeros((len(self.samples_with_frames), int(self.cfg.past_frames), video_dim_raw), dtype=torch.float32)
                print(f"Warning: Using zero placeholder for video features. Shape: {self.z_video_global.shape}")
                print(f"         Expected features at: {features_path}")

    def _get_scenes_to_use(self, all_scenes, held_out_scene, overfit, training):
        """Determine which scenes to use based on split and overfit flag."""
        if overfit:
            return all_scenes
        if held_out_scene is None or held_out_scene == 'all':
            return all_scenes
        else:
            if training:
                return [s for s in all_scenes if s != held_out_scene]
            else:
                return [held_out_scene]

    def _build_dataset(self, data_dir):
        """Build the dataset from annotation files."""
        print(f"Loading annotations from {data_dir} for scenes: {self.scenes_to_use}")

        # We'll collect:
        #   all_windows: list of windows, each window is a list of 20 points (x, y, lost, occluded, generated, frame)
        #   but we will filter by lost/occluded later
        all_windows = []  # each element will be a dict: {'points': list of 20 (x,y), 'start_frame': int, 'valid': bool}
        # Actually, we'll first collect raw windows that are consecutive in frame, then filter by lost/occluded

        # Step 1: Build tracks per scene/video/track_id
        tracks_dict = defaultdict(list)  # key: (scene, video_dir, track_id) -> list of (frame, x, y, lost, occluded, generated)

        for scene in self.scenes_to_use:
            scene_anno_dir = os.path.join(data_dir, 'annotations', scene)
            if not os.path.isdir(scene_anno_dir):
                print(f"Warning: Annotation directory not found: {scene_anno_dir}")
                continue
            video_dirs = [d for d in os.listdir(scene_anno_dir)
                         if os.path.isdir(os.path.join(scene_anno_dir, d)) and d.startswith('video')]
            for video_dir in video_dirs:
                anno_file = os.path.join(scene_anno_dir, video_dir, 'annotations.txt')
                if not os.path.isfile(anno_file):
                    print(f"Warning: Annotation file not found: {anno_file}")
                    continue
                try:
                    with open(anno_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) < 10:
                                continue
                            try:
                                track_id = int(parts[0])
                                xmin = float(parts[1])
                                ymin = float(parts[2])
                                xmax = float(parts[3])
                                ymax = float(parts[4])
                                frame = int(parts[5])
                                lost = int(parts[6])
                                occluded = int(parts[7])
                                generated = int(parts[8])
                                label = parts[9]
                                # Remove quotes if present
                                if label.startswith('"') and label.endswith('"'):
                                    label = label[1:-1]
                            except (ValueError, IndexError):
                                continue

                            # We only want pedestrians
                            if label != 'Pedestrian':
                                continue

                            # Compute center of bounding box
                            x = (xmin + xmax) / 2.0
                            y = (ymin + ymax) / 2.0
                            # Store: (frame, x, y, lost, occluded, generated)
                            tracks_dict[(scene, video_dir, track_id)].append((frame, x, y, lost, occluded, generated))
                except Exception as e:
                    print(f"Error reading {anno_file}: {e}")

        print(f"Found {len(tracks_dict)} tracks across all scenes/videos.")

        # Step 2: For each track, sort by frame and slide windows
        all_raw_windows = []  # each element: (list of 20 (x,y), start_frame)
        for key, points in tracks_dict.items():
            # Sort by frame
            points.sort(key=lambda p: p[0])  # sort by frame

            # Slide window of 20 consecutive frames
            for i in range(len(points) - 19):
                window = points[i:i+20]
                # Check if frames are consecutive
                frames = [p[0] for p in window]
                if all(frames[j] + 1 == frames[j+1] for j in range(19)):
                    # Check that none of the points in the window is lost or occluded
                    if all(p[4] == 0 and p[5] == 0 for p in window):  # lost index 4, occluded index 5
                        # Extract the (x,y) for the 20 points
                        xy_points = [(p[1], p[2]) for p in window]
                        start_frame = frames[0]
                        all_raw_windows.append((xy_points, start_frame))

        print(f"Found {len(all_raw_windows)} raw windows (consecutive frames, no lost/occluded).")

        if len(all_raw_windows) == 0:
            print("Warning: No valid windows found. Dataset will be empty.")
            self.samples_with_frames = []  # (past_norm, future_norm, third, start_frame)
            return

        # Step 3: Collect all points to compute global min and max for normalization
        all_points = []
        for window_points, _ in all_raw_windows:
            all_points.extend(window_points)

        if len(all_points) == 0:
            print("Warning: No points in windows. Dataset will be empty.")
            self.samples_with_frames = []
            return

        all_points_np = np.array(all_points, dtype=np.float32)
        x_min, y_min = np.min(all_points_np[:, 0]), np.min(all_points_np[:, 1])
        x_max, y_max = np.max(all_points_np[:, 0]), np.max(all_points_np[:, 1])

        # Avoid division by zero
        if x_max - x_min < 1e-6:
            x_max = x_min + 1.0
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0

        print(f"Global X range: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Global Y range: [{y_min:.2f}, {y_max:.2f}]")

        # Step 4: Normalize windows and create samples
        self.samples_with_frames = []  # each element: (past_norm_np, future_norm_np, third_np, start_frame)
        for window_points, start_frame in all_raw_windows:
            # Normalize each point
            normalized_points = []
            for (x, y) in window_points:
                x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
                y_norm = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
                normalized_points.append((x_norm, y_norm))

            # Split into past (first 8) and future (last 12)
            past_points = normalized_points[:8]
            future_points = normalized_points[8:20]

            # Convert to numpy arrays
            past_np = np.array(past_points, dtype=np.float32)  # shape (8,2)
            future_np = np.array(future_points, dtype=np.float32) # shape (12,2)

            # Create dummy third element (matching original shape)
            third_np = np.empty((20, 0, 2), dtype=np.float32)  # shape (20,0,2)

            self.samples_with_frames.append((past_np, future_np, third_np, start_frame))

        print(f"Created {len(self.samples_with_frames)} samples.")

        # Store min/max for potential use in normalization (though we already normalized)
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max

    def _prepare_tensors(self):
        """Convert the list of samples into tensors in the format expected by the rest of the code."""
        if len(self.samples_with_frames) == 0:
            # Create empty tensors with expected shapes
            self.past_traj = torch.empty((0, 1, self.past_frames, 2))
            self.future_traj = torch.empty((0, 1, self.future_frames, 2))
            self.past_traj_original_scale = self.past_traj.clone()
            self.future_traj_original_scale = self.future_traj.clone()
            return

        # Extract components from samples_with_frames
        past_list = [s[0] for s in self.samples_with_frames]   # list of (8,2) arrays
        future_list = [s[1] for s in self.samples_with_frames] # list of (12,2) arrays
        # We don't use the third element in the rest of the code, but we'll keep it as dummy
        # third_list = [s[2] for s in self.samples_with_frames]  # list of (20,0,2) arrays

        # Stack into numpy arrays
        past_np = np.array(past_list)   # [N, 8, 2]
        future_np = np.array(future_list) # [N, 12, 2]

        # Add the agent dimension (dimension 1) to match the expected format [N, 1, T, 2]
        past_np = np.expand_dims(past_np, axis=1)   # [N, 1, 8, 2]
        future_np = np.expand_dims(future_np, axis=1) # [N, 1, 12, 2]

        # Convert to tensors
        self.past_traj = torch.from_numpy(past_np)
        self.future_traj = torch.from_numpy(future_np)

        # For consistency with original code, set the original scale tensors to the same as normalized
        # (since we already normalized to [-1,1])
        self.past_traj_original_scale = self.past_traj.clone()
        self.future_traj_original_scale = self.future_traj.clone()

        # Also set the min/max in config to -1,1 so that the normalization step is a no-op
        self.cfg.past_traj_min = -1.0
        self.cfg.past_traj_max = 1.0
        self.cfg.fut_traj_min = -1.0
        self.cfg.fut_traj_max = 1.0

    def __len__(self):
        return len(self.past_traj)

    def __getitem__(self, item):
        if self.imle:
            # We don't have IMLE data in this implementation, so return dummy values
            # This is a placeholder - implement if needed
            past_traj = torch.zeros(1)
            fut_traj = torch.zeros(1)
            past_traj_original_scale = torch.zeros(1)
            fut_traj_original_scale = torch.zeros(1)
            fut_traj_vel = torch.zeros(1)
            y_t = torch.zeros(1)
            y_pred_data = torch.zeros(1)
            z_video = torch.zeros(1) if self.z_video_global is not None else torch.zeros(1)
            agent_crops = torch.zeros(1)
            return [
                past_traj,
                fut_traj,
                past_traj_original_scale,
                fut_traj_original_scale,
                fut_traj_vel,
                y_t,
                y_pred_data,
                z_video,
                agent_crops,
            ]
        else:
            # Return the sample in the format expected by the original code
            # The original code returned a tuple of 8 elements:
            # [index, past_traj_norm_scale, fut_traj_norm_scale, past_traj_original_scale,
            #  fut_traj_original_scale, fut_traj_vel, z_video, agent_crops]
            # We don't have vel or agent_crops yet, so we'll set them to zeros.
            # We also don't have an index in the original sense, but we'll use the item as index.

            past_traj = self.past_traj[item]          # [1, 8, 2]
            fut_traj = self.future_traj[item]       # [1, 12, 2]
            past_traj_original_scale = self.past_traj_original_scale[item]
            fut_traj_original_scale = self.future_traj_original_scale[item]

            # Compute velocity from past_traj_relative (as in original code)
            # We need to compute past_traj_relative first
            initial_pos = past_traj[:, -1:, :]  # [1, 1, 2] - last position of past trajectory
            past_traj_rel = past_traj - initial_pos  # [1, 8, 2]
            # Velocity: difference between consecutive positions
            past_traj_vel = torch.cat([past_traj_rel[:, 1:] - past_traj_rel[:, :-1],
                                       torch.zeros_like(past_traj_rel[:, :1])], dim=1)  # [1, 8, 2]

            # For future_traj_vel, we don't have future ground truth in the same way?
            # In the original code, they computed future_traj_vel from future_traj_rel
            # but we don't have future_traj_relative?
            # We'll set it to zero for now.
            fut_traj_vel = torch.zeros_like(fut_traj)  # [1, 12, 2]

            # Video features
            if self.z_video_global is not None:
                z_video = self.z_video_global[item]  # [T_obs, D]
            else:
                z_video = torch.zeros((self.cfg.past_frames, 512))  # default D=512

            # Agent crops - dummy
            agent_crops = torch.zeros(1)

            # Return in the order expected by seq_collate_sdd
            return [
                torch.tensor([item], dtype=torch.int32),  # index
                past_traj,  # [1, 8, 2]
                fut_traj,   # [1, 12, 2]
                past_traj_original_scale,
                fut_traj_original_scale,
                fut_traj_vel,
                z_video,
                agent_crops,
            ]


if __name__ == "__main__":
    # Simple test
    import sys
    sys.path.insert(0, '/home/dhya/Repos/Studies/MoFlow')
    from utils.config import Config
    cfg = Config('cfg/sdd/cor_fm.yml', 'test')
    dataset = SDDDataset(cfg, './data/sdd', training=True, overfit=False, held_out_scene='bookstore')
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample types: {[type(s) for s in sample]}")
        print(f"Past train shape: {sample[1].shape}")
        print(f"Future train shape: {sample[2].shape}")