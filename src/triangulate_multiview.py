#!/usr/bin/env python3
"""
Per-frame triangulation for multi-view video datasets.

This script generates per-frame 3D point clouds from synchronized multi-view images
using RoMa for dense feature matching and triangulation with known camera poses.

Input: Multi-view dataset with intri.yml and extri.yml calibration files
Output: points3d_frameXXXXXX.npy and colors_frameXXXXXX.npy files

Usage:
    python scripts/triangulate_multiview.py \
        --data-dir /path/to/dataset \
        --output-dir /path/to/output \
        --frame-start 0 --frame-end 30
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_opencv_yaml(yaml_path: str) -> Dict:
    """Parse OpenCV YAML file with matrix data."""
    with open(yaml_path, 'r') as f:
        content = f.read()

    # Remove YAML header
    content = re.sub(r'%YAML.*\n---\n', '', content)

    result = {}

    # Parse names list
    names_match = re.search(r'names:\s*\n((?:\s+-\s*"[^"]+"\n)+)', content)
    if names_match:
        names = re.findall(r'"([^"]+)"', names_match.group(1))
        result['names'] = names

    # Parse matrices and scalars
    # Matrix pattern: KEY: !!opencv-matrix\n  rows: N\n  cols: M\n  dt: d\n  data: [...]
    matrix_pattern = r'(\w+):\s*!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)\s*\n\s*dt:\s*\w\s*\n\s*data:\s*\[([\d\s,.\-e+]+)\]'
    for match in re.finditer(matrix_pattern, content):
        key = match.group(1)
        rows = int(match.group(2))
        cols = int(match.group(3))
        data = [float(x.strip()) for x in match.group(4).split(',')]
        result[key] = np.array(data).reshape(rows, cols)

    # Parse scalar values
    scalar_pattern = r'^(\w+):\s*([-\d.]+)\s*$'
    for match in re.finditer(scalar_pattern, content, re.MULTILINE):
        key = match.group(1)
        if key not in result:
            result[key] = float(match.group(2))

    return result


def load_cameras(data_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load camera intrinsics and extrinsics from yml files."""
    intri_path = os.path.join(data_dir, 'intri.yml')
    extri_path = os.path.join(data_dir, 'extri.yml')

    intri = parse_opencv_yaml(intri_path)
    extri = parse_opencv_yaml(extri_path)

    cam_names = intri.get('names', extri.get('names', []))

    intrinsics = {}
    extrinsics = {}

    for cam in cam_names:
        # Intrinsics
        K = intri.get(f'K_{cam}')
        if K is not None:
            intrinsics[cam] = K

        # Extrinsics: Rot (3x3) and T (3x1)
        Rot = extri.get(f'Rot_{cam}')
        T = extri.get(f'T_{cam}')
        if Rot is not None and T is not None:
            # Build 4x4 extrinsic matrix [R|t]
            extrinsics[cam] = (Rot, T.flatten())

    return intrinsics, extrinsics, cam_names


def triangulate_points_dlt(
    pts1: np.ndarray, pts2: np.ndarray,
    P1: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """
    Triangulate 3D points from 2D correspondences using DLT.

    Args:
        pts1: [N, 2] points in image 1
        pts2: [N, 2] points in image 2
        P1: [3, 4] projection matrix for camera 1
        P2: [3, 4] projection matrix for camera 2

    Returns:
        points3d: [N, 3] triangulated 3D points
    """
    N = pts1.shape[0]
    points3d = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points3d[i] = X[:3] / X[3]

    return points3d


def triangulate_multiview(
    points_2d: Dict[str, np.ndarray],
    proj_matrices: Dict[str, np.ndarray],
    min_views: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate points visible in multiple views.

    Args:
        points_2d: Dict[cam_name, (N, 2) array of 2D points]
        proj_matrices: Dict[cam_name, (3, 4) projection matrix]
        min_views: Minimum number of views required

    Returns:
        points3d: Triangulated 3D points
        valid_mask: Boolean mask of valid points
    """
    cam_names = list(points_2d.keys())
    if len(cam_names) < 2:
        return np.array([]), np.array([])

    # Use first two cameras for initial triangulation
    cam1, cam2 = cam_names[0], cam_names[1]
    pts1 = points_2d[cam1]
    pts2 = points_2d[cam2]
    P1 = proj_matrices[cam1]
    P2 = proj_matrices[cam2]

    points3d = triangulate_points_dlt(pts1, pts2, P1, P2)

    return points3d


def match_features_roma(
    img1: np.ndarray, img2: np.ndarray,
    roma_model,
    device: str = "cuda",
    max_matches: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match features between two images using RoMa.

    Returns:
        kpts1: [N, 2] keypoints in image 1
        kpts2: [N, 2] keypoints in image 2
        confidence: [N] match confidence scores
    """
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    # RoMa expects RGB images
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Convert to torch tensors
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0

    # Get dense warp and certainty
    warp, certainty = roma_model.match(img1_tensor, img2_tensor, device=device)

    # Sample matches
    matches, certainty_sampled = roma_model.sample(warp, certainty, num=max_matches)

    # Convert to pixel coordinates
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H1, W1, H2, W2)

    kpts1 = kpts1.cpu().numpy()
    kpts2 = kpts2.cpu().numpy()
    confidence = certainty_sampled.cpu().numpy()

    return kpts1, kpts2, confidence


def match_features_sift(
    img1: np.ndarray, img2: np.ndarray,
    max_matches: int = 5000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match features between two images using SIFT (fallback).

    Returns:
        kpts1: [N, 2] keypoints in image 1
        kpts2: [N, 2] keypoints in image 2
        confidence: [N] match confidence scores
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_matches)

    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return np.array([]), np.array([]), np.array([])

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return np.array([]), np.array([]), np.array([])

    kpts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    kpts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    confidence = np.array([1.0 - m.distance / 500.0 for m in good_matches])
    confidence = np.clip(confidence, 0, 1)

    return kpts1, kpts2, confidence


def process_frame(
    frame_idx: int,
    data_dir: str,
    output_dir: str,
    cam_names: List[str],
    intrinsics: Dict[str, np.ndarray],
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]],
    roma_model,
    use_roma: bool = True,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    max_matches_per_pair: int = 5000,
    reprojection_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame: match features across views and triangulate.

    Returns:
        points3d: [N, 3] 3D points
        colors: [N, 3] RGB colors (0-255)
    """
    frame_name = f"{frame_idx:06d}.jpg"

    # Load all camera images for this frame
    images = {}
    for cam in cam_names:
        img_path = os.path.join(data_dir, 'images', cam, frame_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(images) < 2:
        print(f"Frame {frame_idx}: Not enough images found")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    available_cams = list(images.keys())

    # Build projection matrices
    proj_matrices = {}
    for cam in available_cams:
        K = intrinsics[cam]
        R, t = extrinsics[cam]
        # P = K @ [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        proj_matrices[cam] = P

    all_points3d = []
    all_colors = []

    # Match and triangulate between camera pairs
    n_cams = len(available_cams)
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            cam1, cam2 = available_cams[i], available_cams[j]
            img1, img2 = images[cam1], images[cam2]

            # Match features
            if use_roma and roma_model is not None:
                try:
                    kpts1, kpts2, conf = match_features_roma(
                        img1, img2, roma_model, device, max_matches_per_pair
                    )
                except Exception as e:
                    print(f"RoMa failed for {cam1}-{cam2}: {e}, falling back to SIFT")
                    kpts1, kpts2, conf = match_features_sift(img1, img2, max_matches_per_pair)
            else:
                kpts1, kpts2, conf = match_features_sift(img1, img2, max_matches_per_pair)

            if len(kpts1) < 10:
                continue

            # Filter by confidence
            mask = conf >= confidence_threshold
            kpts1 = kpts1[mask]
            kpts2 = kpts2[mask]

            if len(kpts1) < 10:
                continue

            # Triangulate
            P1, P2 = proj_matrices[cam1], proj_matrices[cam2]
            points3d = triangulate_points_dlt(kpts1, kpts2, P1, P2)

            # Filter by reprojection error
            # Reproject to both cameras
            pts3d_h = np.hstack([points3d, np.ones((len(points3d), 1))])

            proj1 = (P1 @ pts3d_h.T).T
            proj1 = proj1[:, :2] / proj1[:, 2:3]
            err1 = np.linalg.norm(proj1 - kpts1, axis=1)

            proj2 = (P2 @ pts3d_h.T).T
            proj2 = proj2[:, :2] / proj2[:, 2:3]
            err2 = np.linalg.norm(proj2 - kpts2, axis=1)

            valid = (err1 < reprojection_threshold) & (err2 < reprojection_threshold)

            # Filter points behind camera
            R1, t1 = extrinsics[cam1]
            cam1_pos = -R1.T @ t1
            depths = np.linalg.norm(points3d - cam1_pos, axis=1)
            valid &= (depths > 0.1) & (depths < 100)

            points3d = points3d[valid]
            kpts1 = kpts1[valid]

            if len(points3d) == 0:
                continue

            # Get colors from image 1
            kpts1_int = kpts1.astype(int)
            kpts1_int[:, 0] = np.clip(kpts1_int[:, 0], 0, img1.shape[1] - 1)
            kpts1_int[:, 1] = np.clip(kpts1_int[:, 1], 0, img1.shape[0] - 1)
            colors = img1[kpts1_int[:, 1], kpts1_int[:, 0]]

            all_points3d.append(points3d)
            all_colors.append(colors)

    if len(all_points3d) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    all_points3d = np.vstack(all_points3d)
    all_colors = np.vstack(all_colors)

    # Remove duplicate points (within 0.01 units)
    if len(all_points3d) > 0:
        # Simple deduplication using voxel grid
        voxel_size = 0.005
        voxel_coords = (all_points3d / voxel_size).astype(int)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        all_points3d = all_points3d[unique_idx]
        all_colors = all_colors[unique_idx]

    return all_points3d, all_colors


def main():
    parser = argparse.ArgumentParser(description="Generate per-frame point clouds from multi-view video")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset with intri.yml and extri.yml")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for point clouds")
    parser.add_argument("--frame-start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frame-end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--use-roma", action="store_true", default=True, help="Use RoMa for matching (default)")
    parser.add_argument("--use-sift", action="store_true", help="Use SIFT instead of RoMa")
    parser.add_argument("--device", type=str, default="cuda", help="Device for RoMa")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Confidence threshold for matches")
    parser.add_argument("--max-matches", type=int, default=5000, help="Max matches per camera pair")
    parser.add_argument("--reprojection-threshold", type=float, default=5.0, help="Max reprojection error in pixels")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load cameras
    print("Loading camera calibration...")
    intrinsics, extrinsics, cam_names = load_cameras(args.data_dir)
    print(f"Loaded {len(cam_names)} cameras: {cam_names}")

    # Determine frame range
    sample_cam = cam_names[0]
    sample_dir = os.path.join(args.data_dir, 'images', sample_cam)
    frame_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.jpg')])
    n_frames = len(frame_files)

    frame_start = args.frame_start
    frame_end = args.frame_end if args.frame_end is not None else n_frames
    print(f"Processing frames {frame_start} to {frame_end - 1}")

    # Initialize RoMa if requested
    roma_model = None
    use_roma = args.use_roma and not args.use_sift

    if use_roma:
        try:
            from romatch import roma_outdoor
            print("Loading RoMa model...")
            roma_model = roma_outdoor(device=args.device)
            print("RoMa loaded successfully")
        except ImportError:
            print("RoMa not installed, falling back to SIFT")
            print("Install with: pip install romatch")
            use_roma = False
        except Exception as e:
            print(f"Failed to load RoMa: {e}, falling back to SIFT")
            use_roma = False

    # Process frames
    for frame_idx in tqdm(range(frame_start, frame_end), desc="Processing frames"):
        points3d, colors = process_frame(
            frame_idx=frame_idx,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cam_names=cam_names,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            roma_model=roma_model,
            use_roma=use_roma,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            max_matches_per_pair=args.max_matches,
            reprojection_threshold=args.reprojection_threshold
        )

        # Save point cloud
        points_path = os.path.join(args.output_dir, f"points3d_frame{frame_idx:06d}.npy")
        colors_path = os.path.join(args.output_dir, f"colors_frame{frame_idx:06d}.npy")

        np.save(points_path, points3d.astype(np.float32))
        np.save(colors_path, colors.astype(np.float32))

        if frame_idx % 5 == 0:
            print(f"Frame {frame_idx}: {len(points3d)} points")

    print(f"\nDone! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
