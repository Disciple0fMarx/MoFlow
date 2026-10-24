#!/usr/bin/env python3
"""
SDD Data Preparation Script
Following the same pattern as ETH/UCY data preparation for video features.

This script will:
1. Extract global video features from SDD videos using custom_video.py with stride=12
2. Create frame index files mapping trajectory samples to video frames
3. Prepare data for use with the SDD data loader
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/dhya/Repos/Studies/MoFlow')

def extract_sdd_features(data_root, features_root, scene_list, stride=12):
    """
    Extract features from SDD videos using custom_video.py

    Args:
        data_root: Path to raw SDD videos (expects structure: data_root/scene/video_id/video.mov)
        features_root: Path to save extracted features
        scene_list: List of SDD scene names
        stride: Frame stride for feature extraction (default: 12 to match ETH/UCY)
    """
    from video_encoder.custom_video import encode_custom_video

    print(f"Extracting features for SDD scenes with stride={stride}")

    for scene in scene_list:
        print(f"\nProcessing scene: {scene}")

        # Look for videos in the nested structure: data_root/scene/video_id/video.mov
        scene_video_dir = os.path.join(data_root, scene)
        if not os.path.isdir(scene_video_dir):
            print(f"WARNING: Scene directory not found: {scene_video_dir}")
            continue

        # Find video subdirectories (video0, video1, etc.)
        video_subdirs = [d for d in os.listdir(scene_video_dir)
                        if os.path.isdir(os.path.join(scene_video_dir, d)) and d.startswith('video')]
        if not video_subdirs:
            print(f"WARNING: No video subdirectories found in {scene_video_dir}")
            continue

        # Use the first video (typically video0) for feature extraction
        # This assumes all videos for a scene cover similar content or you want to process one representative video
        video_subdir = sorted(video_subdirs)[0]  # e.g., 'video0'
        video_path = os.path.join(scene_video_dir, video_subdir, "video.mov")

        if not os.path.exists(video_path):
            # Try .mp4 extension
            video_path = os.path.join(scene_video_dir, video_subdir, "video.mp4")
        if not os.path.exists(video_path):
            print(f"WARNING: Video file not found for scene {scene} in {video_path}")
            continue

        output_dir = os.path.join(features_root, "sdd", scene)
        os.makedirs(output_dir, exist_ok=True)

        print(f"  Input video: {video_path}")
        print(f"  Output dir:  {output_dir}")

        try:
            result = encode_custom_video(
                video=video_path,
                out_dir=output_dir,
                backbone="resnet18",
                stride=stride,
                device=None,  # Auto-detect CUDA/CPU
                batch_size=32
            )
            print(f"  SUCCESS: Features saved to {result['features']}")
        except Exception as e:
            print(f"  ERROR processing {scene}: {e}")

def create_sdd_frame_index(data_root, pkl_dir, scene, split, stride=12):
    """
    Create frame index file for SDD scene, similar to ETH/UCY build_frame_index.py

    Args:
        data_root: Path to raw data (for determining stride from actual video)
        pkl_dir: Directory containing SDD pickle files
        scene: SDD scene name
        split: 'train' or 'test'
        stride: Frame stride (default: 12)
    """
    import subprocess

    print(f"Creating frame index for SDD scene {scene} split {split}")

    # Path to SDD pickle file
    pkl_path = os.path.join(pkl_dir, "original", f"{scene}_{split}.pkl")
    if not os.path.exists(pkl_path):
        print(f"ERROR: SDD pickle file not found: {pkl_path}")
        return False

    # Load the pickle to get trajectory data
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    n_samples = len(pkl_data["traj"])  # Shape: [N, T_total, 2]
    print(f"  Found {n_samples} samples in {pkl_path}")

    # For SDD, try to get video information to create a more meaningful frame index
    scene_video_dir = os.path.join(data_root, scene)
    video_path = None

    if os.path.isdir(scene_video_dir):
        # Find video subdirectories (video0, video1, etc.)
        video_subdirs = [d for d in os.listdir(scene_video_dir)
                        if os.path.isdir(os.path.join(scene_video_dir, d)) and d.startswith('video')]
        if video_subdirs:
            # Use the first video (typically video0)
            video_subdir = sorted(video_subdirs)[0]  # e.g., 'video0'
            video_path = os.path.join(scene_video_dir, video_subdir, "video.mov")
            if not os.path.exists(video_path):
                # Try .mp4 extension
                video_path = os.path.join(scene_video_dir, video_subdir, "video.mp4")

    start_frame_ids = None
    actual_stride = stride

    if video_path and os.path.exists(video_path):
        try:
            # Try to get video frame rate using OpenCV if available
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if fps > 0:
                    # Estimate how the trajectory data was sampled
                    # Assuming the original SDD processing used a similar approach to ETH/UCY
                    # ETH/UCY: 25 fps original -> stride 10 -> 2.5 Hz sampling
                    # If SDD is ~30 fps, and we want similar temporal resolution:
                    target_hz = 2.5  # Matches ETH/UCY effective sampling rate
                    estimated_stride = max(1, int(round(fps / target_hz)))

                    print(f"  Video FPS: {fps:.1f}, Frames: {frame_count}")
                    print(f"  Estimated stride for {target_hz}Hz sampling: {estimated_stride}")

                    # Use the estimated stride if it's reasonable
                    if 1 <= estimated_stride <= 30:
                        actual_stride = estimated_stride
                    else:
                        print(f"  Warning: Estimated stride {estimated_stride} seems unreasonable, using default {stride}")

                    # Generate frame IDs assuming uniform sampling from start
                    # This is an approximation - real alignment would require knowing
                    # how the original dataset was created
                    # We'll space samples evenly across the video duration
                    if n_samples > 1:
                        # Distribute n_samples across the video duration
                        max_frame_idx = frame_count - 1
                        if max_frame_idx > 0:
                            step = max_frame_idx / (n_samples - 1)
                            start_frame_ids = np.array([int(i * step) for i in range(n_samples)], dtype=np.int64)
                        else:
                            start_frame_ids = np.zeros(n_samples, dtype=np.int64)
                    else:
                        start_frame_ids = np.zeros(n_samples, dtype=np.int6f)

                    print(f"  Generated approximate frame IDs for {n_samples} samples")
                else:
                    print(f"  Warning: Could not determine FPS for {video_path}")
            else:
                print(f"  Warning: Could not open video {video_path}")
        except ImportError:
            print(f"  Warning: OpenCV not available for video analysis, using dummy frame IDs")
        except Exception as e:
            print(f"  Warning: Error analyzing video {video_path}: {e}")

    # Fallback to dummy frame indices if video analysis failed
    if start_frame_ids is None:
        print(f"  Warning: Using zero placeholders for frame IDs (temporal alignment approximate)")
        start_frame_ids = np.zeros(n_samples, dtype=np.int64)

    # Create index dictionary
    index_dict = {
        "scene": scene,
        "stride": int(actual_stride),
        "start_frame_id": start_frame_ids,
    }

    # Save frame index
    out_dir = os.path.join(pkl_dir, "original", scene)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}_{split}_frame_index.pkl")

    with open(out_path, 'wb') as f:
        pickle.dump(index_dict, f)

    print(f"  Frame index saved to: {out_path} (stride={actual_stride})")
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare SDD data for video-conditioned MoFlow")
    parser.add_argument("--data-root", required=True,
                        help="Path to raw SDD videos (e.g., /path/to/sdd/videos)")
    parser.add_argument("--features-root", default="./features",
                        help="Root directory for saving extracted features")
    parser.add_argument("--pkl-dir", default="./data/sdd",
                        help="Directory containing SDD pickle files")
    parser.add_argument("--stride", type=int, default=12,
                        help="Frame stride for feature extraction (default: 12)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction, only create frame indices")
    parser.add_argument("--skip-indexing", action="store_true",
                        help="Skip frame index creation, only extract features")

    args = parser.parse_args()

    # SDD scenes from the video_encoder/api.py
    sdd_scenes = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']

    print("SDD Data Preparation")
    print("====================")
    print(f"Data root: {args.data_root}")
    print(f"Features root: {args.features_root}")
    print(f"PKL dir: {args.pkl_dir}")
    print(f"Stride: {args.stride}")
    print(f"Scenes: {', '.join(sdd_scenes)}")
    print()

    # Step 1: Extract features from videos
    if not args.skip_extraction:
        print("Step 1: Extracting video features")
        print("-" * 40)
        extract_sdd_features(args.data_root, args.features_root, sdd_scenes, args.stride)
        print()

    # Step 2: Create frame index files
    if not args.skip_indexing:
        print("Step 2: Creating frame index files")
        print("-" * 40)
        for scene in sdd_scenes:
            for split in ['train', 'test']:
                create_sdd_frame_index(
                    args.data_root,
                    args.pkl_dir,
                    scene,
                    split,
                    args.stride
                )
        print()

    print("SDD data preparation complete!")
    print("\nNext steps:")
    print("1. Update cfg/sdd/cor_fm.yml to set USE_VIDEO: True")
    print("2. Update video_encoder/api.py with proper SDD scene resolutions and world bounds")
    print("3. Run training: python fm_sdd.py --cfg cfg/sdd/cor_fm.yml --subset bookstore")

if __name__ == "__main__":
    main()