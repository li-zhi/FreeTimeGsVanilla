#!/usr/bin/env python3
"""
Convert OpenCV calibration YAML files (intri.yml, extri.yml) to COLMAP sparse format.

This script creates COLMAP-compatible cameras.bin, images.bin, and points3D.bin files
from pre-calibrated multi-view camera setups.

Usage:
    python convert_calibration_to_colmap.py \
        --input-dir /path/to/dataset \
        --output-dir /path/to/output
"""

import argparse
import struct
import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


def parse_opencv_yaml(yaml_path: str) -> dict:
    """Parse OpenCV YAML file manually (handles !!opencv-matrix format)."""
    data = {}
    current_key = None
    current_matrix = None
    matrix_data = []

    with open(yaml_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip YAML header and empty lines
        if line.startswith('%YAML') or line == '---' or not line:
            i += 1
            continue

        # Parse names list
        if line == 'names:':
            names = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                name = lines[i].strip()[2:].strip('"').strip("'")
                names.append(name)
                i += 1
            data['names'] = names
            continue

        # Parse key-value pairs
        if ':' in line and not line.endswith(':'):
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()

            if value == '!!opencv-matrix':
                # Start reading matrix
                current_key = key
                matrix_data = []
                rows = cols = 0
                i += 1
                while i < len(lines):
                    mline = lines[i].strip()
                    if mline.startswith('rows:'):
                        rows = int(mline.split(':')[1].strip())
                    elif mline.startswith('cols:'):
                        cols = int(mline.split(':')[1].strip())
                    elif mline.startswith('dt:'):
                        pass  # data type, assume double
                    elif mline.startswith('data:'):
                        # Parse data array
                        data_str = mline.split(':', 1)[1].strip()
                        data_str = data_str.strip('[]')
                        values = [float(x.strip()) for x in data_str.split(',') if x.strip()]
                        matrix_data = values
                        data[current_key] = np.array(matrix_data).reshape(rows, cols)
                        i += 1
                        break
                    i += 1
                continue
            else:
                # Simple value
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value

        i += 1

    return data


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    # Using scipy-like conversion
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def write_cameras_binary(cameras: dict, path: str):
    """
    Write cameras.bin in COLMAP binary format.

    Format per camera:
    - camera_id: uint32
    - model_id: int32 (1 = PINHOLE, 4 = OPENCV)
    - width: uint64
    - height: uint64
    - params: double[] (4 for PINHOLE: fx, fy, cx, cy; 8 for OPENCV: fx, fy, cx, cy, k1, k2, p1, p2)
    """
    with open(path, 'wb') as f:
        # Number of cameras
        f.write(struct.pack('<Q', len(cameras)))

        for cam_id, cam in cameras.items():
            f.write(struct.pack('<I', cam_id))  # camera_id
            f.write(struct.pack('<i', cam['model_id']))  # model_id
            f.write(struct.pack('<Q', cam['width']))  # width
            f.write(struct.pack('<Q', cam['height']))  # height

            for param in cam['params']:
                f.write(struct.pack('<d', param))


def write_images_binary(images: dict, path: str):
    """
    Write images.bin in COLMAP binary format.

    Format per image:
    - image_id: uint32
    - qw, qx, qy, qz: double[4] (quaternion)
    - tx, ty, tz: double[3] (translation)
    - camera_id: uint32
    - name: string (null-terminated)
    - num_points2D: uint64
    - points2D: (x, y, point3D_id)[] - we write empty since no triangulated points yet
    """
    with open(path, 'wb') as f:
        # Number of images
        f.write(struct.pack('<Q', len(images)))

        for img_id, img in images.items():
            f.write(struct.pack('<I', img_id))  # image_id

            # Quaternion (w, x, y, z)
            for q in img['qvec']:
                f.write(struct.pack('<d', q))

            # Translation
            for t in img['tvec']:
                f.write(struct.pack('<d', t))

            f.write(struct.pack('<I', img['camera_id']))  # camera_id

            # Image name (null-terminated)
            name_bytes = img['name'].encode('utf-8') + b'\x00'
            f.write(name_bytes)

            # Number of 2D points (0 for now - no features extracted)
            f.write(struct.pack('<Q', 0))


def write_points3d_binary(path: str):
    """Write empty points3D.bin file."""
    with open(path, 'wb') as f:
        # Number of points
        f.write(struct.pack('<Q', 0))


def main():
    parser = argparse.ArgumentParser(description='Convert OpenCV calibration to COLMAP sparse format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing intri.yml, extri.yml, and images/')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for COLMAP format data')
    parser.add_argument('--single-frame', type=int, default=None,
                        help='If specified, only process this single frame index')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Parse calibration files
    print("Parsing intrinsic calibration...")
    intri = parse_opencv_yaml(str(input_dir / 'intri.yml'))

    print("Parsing extrinsic calibration...")
    extri = parse_opencv_yaml(str(input_dir / 'extri.yml'))

    camera_names = intri['names']
    print(f"Found {len(camera_names)} cameras: {camera_names}")

    # Get image dimensions from first image
    images_dir = input_dir / 'images'
    first_cam_dir = images_dir / camera_names[0]
    first_image = list(first_cam_dir.glob('*.jpg'))[0]
    sample_img = cv2.imread(str(first_image))
    img_height, img_width = sample_img.shape[:2]
    print(f"Image dimensions: {img_width} x {img_height}")

    # Get list of frames
    frame_files = sorted([f.name for f in first_cam_dir.glob('*.jpg')])
    frame_indices = [int(f.split('.')[0]) for f in frame_files]
    print(f"Found {len(frame_indices)} frames: {frame_indices[0]} to {frame_indices[-1]}")

    if args.single_frame is not None:
        frame_indices = [args.single_frame]
        print(f"Processing single frame: {args.single_frame}")

    # Create output structure
    # For FreeTimeGS, we need: output_dir/images/ and output_dir/sparse/0/
    colmap_images_dir = output_dir / 'images'
    sparse_dir = output_dir / 'sparse' / '0'
    colmap_images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Build cameras dictionary (one camera model per physical camera)
    # Using OPENCV model (model_id=4) to include distortion
    cameras = {}
    for cam_idx, cam_name in enumerate(camera_names):
        cam_id = cam_idx + 1  # COLMAP uses 1-indexed

        K = intri.get(f'K_{cam_name}')
        D = intri.get(f'D_{cam_name}')

        if K is None:
            print(f"Warning: Missing intrinsics for camera {cam_name}")
            continue

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Distortion coefficients (k1, k2, p1, p2)
        if D is not None and D.size >= 4:
            k1, k2, p1, p2 = D.flatten()[:4]
        else:
            k1, k2, p1, p2 = 0, 0, 0, 0

        cameras[cam_id] = {
            'model_id': 4,  # OPENCV
            'width': img_width,
            'height': img_height,
            'params': [fx, fy, cx, cy, k1, k2, p1, p2]
        }

    print(f"Created {len(cameras)} camera models")

    # Build images dictionary
    # FreeTimeParser expects ONE image per physical camera (at reference frame)
    # with nested naming: cam{XX}/{frame}.jpg
    # The parser will then list all frames from each camera folder
    images = {}
    image_id = 1

    # Use first frame as reference for COLMAP poses
    reference_frame = frame_indices[0]
    reference_frame_name = f'{reference_frame:06d}.jpg'

    for cam_idx, cam_name in enumerate(camera_names):
        cam_id = cam_idx + 1

        # Get rotation and translation
        R = extri.get(f'Rot_{cam_name}')
        T = extri.get(f'T_{cam_name}')

        if R is None or T is None:
            print(f"Warning: Missing extrinsics for camera {cam_name}")
            continue

        # COLMAP uses world-to-camera transformation.
        # The OpenCV extri.yml Rot_XX / T_XX are already world-to-camera
        # (same convention as triangulate_multiview.py: P = K @ [R | t]).
        R_w2c = R
        t_w2c = T.flatten()

        # Convert to quaternion
        qvec = rotation_matrix_to_quaternion(R_w2c)

        # Image name format: cam{cam_name}/{reference_frame:06d}.jpg (nested)
        colmap_img_name = f'cam{cam_name}/{reference_frame:06d}.jpg'

        images[image_id] = {
            'qvec': qvec,
            'tvec': t_w2c,
            'camera_id': cam_id,
            'name': colmap_img_name
        }

        image_id += 1

    print(f"Created {len(images)} image entries (one per camera)")

    # Create nested image directory structure with symlinks to ALL frames
    print("Creating nested image directory structure...")
    for cam_name in camera_names:
        cam_dir = colmap_images_dir / f'cam{cam_name}'
        cam_dir.mkdir(exist_ok=True)

        for frame_idx in frame_indices:
            frame_name = f'{frame_idx:06d}.jpg'
            src_path = images_dir / cam_name / frame_name
            dst_path = cam_dir / frame_name

            if not dst_path.exists():
                if src_path.exists():
                    dst_path.symlink_to(src_path.resolve())
                else:
                    print(f"Warning: Source image not found: {src_path}")

    print(f"Linked {len(frame_indices)} frames for {len(camera_names)} cameras")

    # Write binary files
    print("Writing cameras.bin...")
    write_cameras_binary(cameras, str(sparse_dir / 'cameras.bin'))

    print("Writing images.bin...")
    write_images_binary(images, str(sparse_dir / 'images.bin'))

    print("Writing points3D.bin (empty)...")
    write_points3d_binary(str(sparse_dir / 'points3D.bin'))

    print(f"\nCOLMAP sparse reconstruction written to: {sparse_dir}")
    print(f"Images linked to: {colmap_images_dir}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── cam00/")
    print(f"  │   │   ├── 000000.jpg")
    print(f"  │   │   └── ...")
    print(f"  │   └── cam{{XX}}/")
    print(f"  └── sparse/0/")
    print(f"      ├── cameras.bin  (one model per camera)")
    print(f"      ├── images.bin   (one entry per camera at reference frame)")
    print(f"      └── points3D.bin")


if __name__ == '__main__':
    main()
