import os
import pickle
import glob
import numpy as np
import math
from einops import rearrange
import torch
import matplotlib.pyplot as plt
from utils.normalization import normalize_min_max
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from video_encoder.moflow_adapter import FrameFeatureLookup


def seq_collate_eth(batch):
    (index, past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, z_video, agent_crops) = zip(*batch)
    pre_motion_3D = torch.stack(past_traj, dim=0)
    fut_motion_3D = torch.stack(fut_traj, dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel_stack = torch.stack(traj_vel, dim=0)

    # z_video may be a degenerate `torch.zeros(1)` when USE_VIDEO=False — detect.
    if z_video[0].dim() == 3:
        z_video_stack = torch.stack(z_video, dim=0)        # [B, T_obs, D_raw]
    else:
        z_video_stack = None

    # agent_crops may be a degenerate `torch.zeros(1)` when USE_AGENT_VIDEO=False — detect.
    if agent_crops[0].dim() == 4:  # [A, P, 3, h, w] -> 4 dimensions
        agent_crops_stack = torch.stack(agent_crops, dim=0)  # [B, A, P, 3, h, w]
    else:
        agent_crops_stack = None

    data_dict = {
        'batch_size': torch.tensor(pre_motion_3D.shape[0]),
        'index': torch.cat(index, dim=0),
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel_stack,
    }
    if z_video_stack is not None:
        data_dict['z_video_global'] = z_video_stack
    if agent_crops_stack is not None:
        data_dict['agent_crops'] = agent_crops_stack
    return data_dict


def seq_collate_imle_train(batch):
    # Removed video_frames unpacking to reflect the optimized lightweight pipeline
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, y_t, y_pred_data, agent_crops) = zip(*batch)

    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    y_t = torch.stack(y_t, dim=0)
    y_pred_data = torch.stack(y_pred_data,dim=0)

    # agent_crops may be a degenerate `torch.zeros(1)` when USE_AGENT_VIDEO=False — detect.
    if agent_crops[0].dim() == 4:  # [A, P, 3, h, w] -> 4 dimensions
        agent_crops_stack = torch.stack(agent_crops, dim=0)  # [B, A, P, 3, h, w]
    else:
        agent_crops_stack = None

    batch_size = torch.tensor(pre_motion_3D.shape[0]) ### bt
    data = {
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,
        'y_t': y_t,
        'y_pred_data': y_pred_data,
    }
    if agent_crops_stack is not None:
        data['agent_crops'] = agent_crops_stack

    return data


def rotate_traj(past_rel, future_rel, past_abs, agents=2, rotate_time_frame=0, subset='eth'):
    past_rel = rearrange(past_rel, 'b a p d -> (b a) p d')
    past_abs = rearrange(past_abs, 'b a p d -> (b a) p d')
    future_rel = rearrange(future_rel, 'b a f d -> (b a) f d')
    past_diff = past_rel[:, rotate_time_frame]
    
    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
    past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)
    
    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(1, 2)
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(1, 2)
    past_after = rearrange(past_after, '(b a) p d -> b a p d', a=agents)
    future_after = rearrange(future_after, '(b a) f d -> b a f d', a=agents)
    past_abs_after = rearrange(past_abs_after, '(b a) p d -> b a p d', a=agents)

    return past_after, future_after, past_abs_after


class ETHDataset(object):
    def __init__(self, cfg, training=True, data_dir = None, subset = None, rotate_time_frame=0, imle=False, type='original'):
        self.type = type
        self.imle = imle
        self.frame_ids = []
        
        ### LED version of preprocessed data
        if self.type == 'LED':
            data_file_path = os.path.join(data_dir, self.type, '{:s}_data_{:s}.npy'.format(subset, 'train' if training else 'test'))
            num_file_path = os.path.join(data_dir, self.type, '{:s}_num_{:s}.npy'.format(subset, 'train' if training else 'test'))

            all_data = np.load(data_file_path)
            all_num = np.load(num_file_path)
            
            # === SET self.frame_ids FOR LED ===
            self.frame_ids = all_num[:, 0, 0] 
            
            self.all_data = torch.Tensor(all_data)
            self.all_num = torch.Tensor(all_num) 
        elif self.type == 'original':
            ### Original version of preprocessed data
            data_file_path = os.path.join(data_dir, self.type, subset, f'{subset}_{"train" if training else "test"}.pkl')
            
            with open(data_file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # --- CRITICAL FIX: Extract scalar count ---
            all_data = data_dict['traj']
            num_peds = all_data.shape[0] 
            
            # Metadata for temporal synchronization
            frame_list = np.array(data_dict['frame_list'])
            seq_start_end = data_dict['seq_start_end']
            
            # Create a 1D FLAT array
            full_frame_ids = np.zeros(num_peds, dtype=int)
            for i, (start, end) in enumerate(seq_start_end):
                full_frame_ids[start:end] = frame_list[i]
                
            self.frame_ids = full_frame_ids 
            # ----------------------------------------------------------------

            # Load the trajectory coordinates as tensors
            self.all_data = torch.Tensor(all_data)
            self.all_data = self.all_data[:, None, :, :] # Shape: [A, 1, T, 2]

        else:
            raise ValueError('Invalid type')


        self.cfg = cfg
        self.rotate_time_frame = rotate_time_frame
        self.imle = imle
        
        ### set the agent_num in the cfg
        cfg.agents = self.all_data.shape[1]
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = cfg.agents
        
        ### compute past and future trajectories
        past_traj_abs = self.all_data[:,:,:cfg.past_frames]
        initial_pos = past_traj_abs[:, :, -1:]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj = (self.all_data[:,:,cfg.past_frames:] - initial_pos).contiguous()
        if cfg.rotate:
            past_traj_rel, fut_traj, past_traj_abs = rotate_traj(past_traj_rel, fut_traj, past_traj_abs, cfg.agents, rotate_time_frame, subset)
        past_traj_vel = torch.cat((past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1], torch.zeros_like(past_traj_rel[:,:, -1:])), dim=2)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat((fut_traj[:, :, 1:] - fut_traj[:,:, :-1], torch.zeros_like(fut_traj[:, :, -1:])), dim=2)

        self.rotate_aug = cfg.rotate and training

        if training:
            cfg.fut_traj_max = fut_traj.max()
            cfg.fut_traj_min = fut_traj.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()
        
        ### record the original to avoid numerical errors
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj

        self.use_video = bool(getattr(cfg.MODEL.CONTEXT_ENCODER, 'USE_VIDEO', False))
        self.use_agent_video = bool(getattr(cfg.MODEL.CONTEXT_ENCODER, 'USE_AGENT_VIDEO', False))
        self.agent_crop_size = getattr(cfg.MODEL.CONTEXT_ENCODER, 'AGENT_CROP_SIZE', [64, 64])
        self.z_video_global = None
        if self.use_video:
            split = 'train' if training else 'test'
            frame_idx_path = os.path.join(
                data_dir, 'original', subset,
                f'{subset}_{split}_frame_index.pkl'
            )
            if not os.path.exists(frame_idx_path):
                raise FileNotFoundError(
                    f"USE_VIDEO=True but {frame_idx_path} is missing. "
                    f"Run: python -m video_encoder.scripts.build_frame_index "
                    f"--data-root <raw_root> --pkl-dir {os.path.join(data_dir, 'original')} "
                    f"--scene {subset} --split {split}"
                )
            with open(frame_idx_path, 'rb') as f:
                frame_index = pickle.load(f)
            
            start_fids = np.asarray(frame_index['start_frame_id'], dtype=np.int64)
            stride = int(frame_index.get('stride', 10))
            assert start_fids.shape[0] == self.all_data.shape[0], (
                f"frame_index has {start_fids.shape[0]} entries but pickle has "
                f"{self.all_data.shape[0]} samples. Rebuild the index."
            )
            features_root = getattr(cfg.MODEL.CONTEXT_ENCODER, 'VIDEO_FEATURES_ROOT',
                                    'features/resnet18')
            scene_name = frame_index.get('scene', subset)
            lookup = FrameFeatureLookup.from_root(features_root, scene=scene_name)
            T_obs = int(cfg.past_frames)
            D = lookup.features.shape[1]
            z = np.zeros((self.all_data.shape[0], T_obs, D), dtype=np.float32)
            for i, fid in enumerate(start_fids):
                z[i] = lookup.window(int(fid), n_frames=T_obs, stride=stride,
                                     policy='nearest')
            self.z_video_global = torch.from_numpy(z)
   
        ### min-max linear normalization
        if cfg.data_norm == 'min_max':
            self.past_traj = normalize_min_max(past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1).contiguous()
            self.fut_traj = normalize_min_max(fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1).contiguous()
        elif cfg.data_norm == 'original':
            self.past_traj = past_traj
            self.fut_traj = fut_traj


        """load distillation target"""
        if imle:
            os.makedirs(os.path.join(data_dir, f'imle/{subset}'), exist_ok=True)
            pkl_ls = sorted(glob.glob(os.path.join(data_dir, f'imle/{subset}/*train*.pkl')))

            keys_ls = ['past_traj', 'fut_traj', 'past_traj_original_scale', 'fut_traj_original_scale', 'fut_traj_vel', 'y_t', 'y_pred_data', 'start_frame']
            imle_data_dict = {}
            total_scenes_loaded_ = 0
            for i_pkl, cur_pkl in enumerate(pkl_ls):
                data = pickle.load(open(cur_pkl, 'rb'))

                if i_pkl == 0:
                    self.imle_meta_data = data['meta_data']
                
                for key in keys_ls:
                    if key not in imle_data_dict:
                        imle_data_dict[key] = []
                    if key == 'y_t':
                        imle_data_dict[key].append(data[key][:, -1])
                    else:
                        imle_data_dict[key].append(data[key])

                total_scenes_loaded_ += data['past_traj'].shape[0]

                if total_scenes_loaded_ >= len(self.past_traj):
                    break

            # concat the data
            for key in keys_ls:
                imle_data_dict[key] = torch.from_numpy(np.concatenate(imle_data_dict[key], axis=0))[:len(self.past_traj)]

            self.imle_data_dict = imle_data_dict
        
        self.subset = subset

        self.SCENE_RESOLUTIONS = {
            'eth':   (640, 480),
            'hotel': (720, 576),
            'univ':  (720, 576),
            'zara1': (720, 576),
            'zara2': (720, 576),
        }
        
        self.SCENE_WORLD_BOUNDS = {
            'eth':   (-7.69, 13.89,  -1.81, 12.67),             
            'hotel': (-10.31, 4.31, -2.77,  4.04),
            'univ':  (-0.46, 15.47,  -0.32, 13.89),
            'zara1': (-0.14, 15.48,  -0.37, 12.39),
            'zara2': (-0.36, 15.56,  -0.19, 13.48),
        }
        
        self.SCENE_FLIP = {
            'eth':   (False, False),  
            'hotel': (False, False),    
            'univ':  (True,  False),  
            'zara1': (False, False),  
            'zara2': (False, False),  
        }
        
        self.SCENE_SWAP_XY = {'hotel': True}
        self.SCENE_FLIP_X = {'hotel': True}

        # Load scene-specific homography matrix
        h_path = os.path.join(data_dir, 'homography', f'{subset}_H.txt')
        if os.path.exists(h_path):
            self.H = np.loadtxt(h_path)
        else:
            print(f"Warning: Homography for {subset} not found at {h_path}. Using identity.")
            self.H = np.eye(3) 
        self.H_inv = np.linalg.inv(self.H)
        self.orig_res = self.SCENE_RESOLUTIONS.get(subset, (720, 576))
        
    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, item):
        if self.imle:
            out = [
                    self.imle_data_dict['past_traj'][item], 
                    self.imle_data_dict['fut_traj'][item],
                    self.imle_data_dict['past_traj_original_scale'][item],
                    self.imle_data_dict['fut_traj_original_scale'][item],
                    self.imle_data_dict['fut_traj_vel'][item],
                    self.imle_data_dict['y_t'][item],
                    self.imle_data_dict['y_pred_data'][item]
                ]
            start_frame = int(self.imle_data_dict['start_frame'][item])
        else:
            past_traj_norm_scale = self.past_traj[item]                             # [A, P, 6]
            fut_traj_norm_scale = self.fut_traj[item]                               # [A, F, 2] 
            past_traj_original_scale = self.past_traj_original_scale[item]          # [A, P, 6]
            fut_traj_original_scale = self.fut_traj_original_scale[item]            # [A, F, 2]
            fut_traj_vel = self.fut_traj_vel[item]                                  # [A, F, 2]   

            if self.rotate_aug:
                A = past_traj_norm_scale.size(0)
                rot_angle = torch.rand(A) * 2 * math.pi

                past_traj_abs_o, past_traj_rel_o, past_traj_vel_o = past_traj_original_scale.chunk(3, dim=-1)   # [A, P, 2] * 3

                rotate_matrix = torch.zeros((rot_angle.size(0), 2, 2)).to(past_traj_norm_scale.device)          # [A, 2, 2]
                rotate_matrix[:, 0, 0] = torch.cos(rot_angle)
                rotate_matrix[:, 0, 1] = torch.sin(rot_angle)
                rotate_matrix[:, 1, 0] = - torch.sin(rot_angle)
                rotate_matrix[:, 1, 1] = torch.cos(rot_angle)

                past_traj_abs_o_rot = torch.matmul(rotate_matrix, past_traj_abs_o.transpose(1, 2)).transpose(1, 2)
                past_traj_rel_o_rot = torch.matmul(rotate_matrix, past_traj_rel_o.transpose(1, 2)).transpose(1, 2)
                fut_traj_o_rot = torch.matmul(rotate_matrix, fut_traj_original_scale.transpose(1, 2)).transpose(1, 2)

                past_traj_vel_o_rot = torch.cat((past_traj_rel_o_rot[:, :, 1:] - past_traj_rel_o_rot[:, :, :-1], torch.zeros_like(past_traj_rel_o_rot[:,:, -1:])), dim=2)
                past_traj_rot = torch.cat((past_traj_abs_o_rot, past_traj_rel_o_rot, past_traj_vel_o_rot), dim=-1)
                
                fut_traj_vel_o = torch.cat((fut_traj_o_rot[:, :, 1:] - fut_traj_o_rot[:,:, :-1], torch.zeros_like(fut_traj_o_rot[:, :, -1:])), dim=2)

                
                past_traj_norm_scale = normalize_min_max(past_traj_rot, self.cfg.past_traj_min, self.cfg.past_traj_max, -1, 1).contiguous()
                fut_traj_norm_scale = normalize_min_max(fut_traj_o_rot, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1).contiguous()
                past_traj_original_scale = past_traj_rot
                fut_traj_original_scale = fut_traj_o_rot
                fut_traj_vel = fut_traj_vel_o

            # Extract agent-centric video crops if enabled
            agent_crops = None
            if self.use_agent_video:
                agent_crops = self._extract_agent_crops(past_traj_original_scale)

            # Extract agent-centric video crops if enabled
            agent_crops = None
            if self.use_agent_video:
                agent_crops = self._extract_agent_crops(past_traj_original_scale)

            z_video = (self.z_video_global[item]
                   if self.z_video_global is not None
                   else torch.zeros(1))
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

    def world_to_pixel_backup(self, traj_pts, target_res=(224, 224)):
        traj_w  = np.array(traj_pts, dtype=np.float64)
        orig_w, orig_h = self.orig_res
        bounds = self.SCENE_WORLD_BOUNDS.get(self.subset)

        # Scenes with valid homography
        if bounds is None:
            h = np.hstack((traj_w, np.ones((len(traj_w), 1)))).T
            c = self.H_inv @ h
            img_pts = (c / c[2]).T[:, :2].copy()
        else:
            x_min, x_max, y_min, y_max = bounds
            pad = 10
            img_pts = np.zeros_like(traj_w)
            img_pts[:, 0] = (traj_w[:, 0] - x_min) / (x_max - x_min) * (orig_w - 2*pad) + pad
            img_pts[:, 1] = (traj_w[:, 1] - y_min) / (y_max - y_min) * (orig_h - 2*pad) + pad

        # Flip y — world y-axis points opposite to image y-axis in all ETH-UCY scenes
        img_pts[:, 1] = orig_h - img_pts[:, 1]

        img_pts[:, 0] = img_pts[:, 0] / orig_w * target_res[0]
        img_pts[:, 1] = img_pts[:, 1] / orig_h * target_res[1]
        return img_pts
        
    def world_to_pixel(self, traj_pts, target_res=(224, 224)):
        traj_w  = np.array(traj_pts, dtype=np.float64)
        orig_w, orig_h = self.orig_res

        x_min, x_max, y_min, y_max = self.SCENE_WORLD_BOUNDS[self.subset]
        pad = 10
        img_pts = np.zeros_like(traj_w)
        img_pts[:, 0] = (traj_w[:, 0] - x_min) / (x_max - x_min) * (orig_w - 2*pad) + pad
        img_pts[:, 1] = (traj_w[:, 1] - y_min) / (y_max - y_min) * (orig_h - 2*pad) + pad
        
        # Y-axis is inverted relative to image coordinates in all ETH-UCY scenes
        img_pts[:, 1] = orig_h - img_pts[:, 1]

        img_pts[:, 0] = img_pts[:, 0] / orig_w * target_res[0]
        img_pts[:, 1] = img_pts[:, 1] / orig_h * target_res[1]
        return img_pts

    def _extract_agent_crops(self, past_traj_original_scale):
        """
        Extract agent-centric video crops for each agent in each observed frame.

        Args:
            past_traj_original_scale: [A, P, 6] tensor containing [abs_xy, rel_xy, vel_xy]
                                    where abs_xy are world coordinates

        Returns:
            Tensor of shape [A, P, 3, h, w] containing RGB crops for each agent/frame,
            or None if video is not enabled
        """
        if not self.use_agent_video:
            return None

        A, P, _ = past_traj_original_scale.shape
        h, w = self.agent_crop_size

        # Initialize crops tensor
        crops = torch.zeros(A, P, 3, h, w, dtype=torch.float32)

        # Get original image dimensions for the scene
        orig_w, orig_h = self.orig_res

        # Process each agent and frame
        for a in range(A):
            for p in range(P):
                # Extract world coordinates (x, y) - first two channels
                world_xy = past_traj_original_scale[a, p, :2].numpy()  # [x, y]

                # Convert to pixel coordinates using existing homography method
                # world_to_pixel expects [N, 2] array
                pixel_xy = self.world_to_pixel(world_xy.reshape(1, 2))[0]  # [x, y] in pixel space
                pixel_x, pixel_y = int(round(pixel_xy[0])), int(round(pixel_xy[1]))

                # Calculate crop boundaries
                x1 = pixel_x - w // 2
                y1 = pixel_y - h // 2
                x2 = x1 + w
                y2 = y1 + h

                # Handle padding if crop goes outside image boundaries
                x1_pad = max(0, -x1)
                y1_pad = max(0, -y1)
                x2_pad = max(0, x2 - orig_w)
                y2_pad = max(0, y2 - orig_h)

                # Adjust crop boundaries to be within image
                x1_clip = max(0, x1)
                y1_clip = max(0, y1)
                x2_clip = min(orig_w, x2)
                y2_clip = min(orig_h, y2)

                # Extract the actual image region
                if x2_clip > x1_clip and y2_clip > y1_clip:
                    # TODO: Actually load the image frame and extract the crop
                    # For now, we'll return zeros as placeholder
                    # In a real implementation, we would:
                    # 1. Determine which frame this is (need frame ID)
                    # 2. Load the corresponding image
                    # 3. Extract the crop [y1_clip:y2_clip, x1_clip:x2_clip]
                    # 4. Apply padding if needed
                    pass

                # For now, return zeros tensor with correct shape
                # This will be replaced with actual image loading logic
                crops[a, p] = torch.zeros(3, h, w, dtype=torch.float32)

        return crops

class ETHDatasetSocialGAN:
    '''
    For the purpose of saving pickle files for train/val/test data for ETH-UCY dataset
    '''
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t', subset='eth'):
        super(ETHDatasetSocialGAN, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.cur_frame_no = 0

        all_files_train = sorted(os.listdir(self.data_dir + '/train/'))
        all_files_test = sorted(os.listdir(self.data_dir + '/test/'))
        all_files_val = sorted(os.listdir(self.data_dir + '/val/'))
        all_files_train = [os.path.join(self.data_dir + '/train/', path) for path in all_files_train]
        all_files_test = [os.path.join(self.data_dir + '/test/', path) for path in all_files_test]
        all_files_val = [os.path.join(self.data_dir + '/val/', path) for path in all_files_val]
        data_dict = {'train': all_files_train, 'test': all_files_test, 'val': all_files_val}
        
        
        for type, all_files in data_dict.items():

            ### init the frame, seq_list and num_peds_in_seq while storing the train/val/test data
            num_peds_in_seq = []
            seq_list = []
            frame_list = []
            for path in all_files:    
                data = self.read_file(path, delim)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        num_peds_considered += 1
                    if num_peds_considered > min_ped:
                        num_peds_in_seq.append(num_peds_considered)
                        seq_list.append(curr_seq[:num_peds_considered])
                        frame_list.append(frames[idx])

            self.num_seq = len(seq_list)
            self.trajs = np.concatenate(seq_list, axis=0).transpose(0, 2, 1)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            self.frame_list = np.array(frame_list, dtype=np.int32)
            traj_data = {
                'traj': self.trajs,
                'frame_list': self.frame_list,
                'num_peds_in_seq': num_peds_in_seq,
                'seq_start_end': self.seq_start_end
            }
            pickle.dump(traj_data, open(self.data_dir + '/{:s}_{:s}.pkl'.format(subset, type), 'wb'))


    def read_file(self, _path, delim):
        delim = delim if delim else self.delim
        data = []
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)
