#!/usr/bin/env python3
"""
Tests for convert_calibration_to_colmap.py

Uses sample data from tests/resources/actor1_4_subseq/
"""

import os
import sys
import struct
import tempfile
import shutil
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from convert_calibration_to_colmap import (
    parse_opencv_yaml,
    rotation_matrix_to_quaternion,
    write_cameras_binary,
    write_images_binary,
    write_points3d_binary,
)

# Path to test resources
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources', 'actor1_4_subseq')


class TestParseOpencvYaml:
    """Tests for parse_opencv_yaml function."""

    def test_parse_intri_yaml(self):
        """Test parsing intrinsics YAML file."""
        intri_path = os.path.join(RESOURCES_DIR, 'intri.yml')
        result = parse_opencv_yaml(intri_path)

        # Check names list is parsed
        assert 'names' in result
        assert len(result['names']) == 18  # 18 cameras (00-17)
        assert result['names'][0] == '00'
        assert result['names'][-1] == '17'

        # Check K matrix is parsed correctly
        assert 'K_00' in result
        K = result['K_00']
        assert K.shape == (3, 3)
        assert np.isclose(K[0, 0], 2157.684, rtol=1e-3)  # fx
        assert np.isclose(K[1, 1], 2157.928, rtol=1e-3)  # fy
        assert np.isclose(K[0, 2], 967.95, rtol=1e-3)    # cx
        assert np.isclose(K[1, 2], 533.784, rtol=1e-3)   # cy

        # Check distortion coefficients
        assert 'D_00' in result
        D = result['D_00']
        assert D.shape == (5, 1)

    def test_parse_extri_yaml(self):
        """Test parsing extrinsics YAML file."""
        extri_path = os.path.join(RESOURCES_DIR, 'extri.yml')
        result = parse_opencv_yaml(extri_path)

        # Check rotation matrix is parsed
        assert 'Rot_00' in result
        Rot = result['Rot_00']
        assert Rot.shape == (3, 3)

        # Check translation vector is parsed
        assert 'T_00' in result
        T = result['T_00']
        assert T.shape == (3, 1)

    def test_parse_scalar_values(self):
        """Test parsing scalar values from YAML."""
        extri_path = os.path.join(RESOURCES_DIR, 'extri.yml')
        result = parse_opencv_yaml(extri_path)

        # Check scalar values like t_00, n_00, f_00
        assert 't_00' in result
        assert result['t_00'] == 0.0
        assert 'n_00' in result
        assert result['n_00'] == 4.0
        assert 'f_00' in result
        assert result['f_00'] == 9.0

    def test_parse_all_cameras(self):
        """Test that all 18 cameras are parsed correctly."""
        intri_path = os.path.join(RESOURCES_DIR, 'intri.yml')
        extri_path = os.path.join(RESOURCES_DIR, 'extri.yml')

        intri = parse_opencv_yaml(intri_path)
        extri = parse_opencv_yaml(extri_path)

        for i in range(18):
            cam_name = f'{i:02d}'
            assert f'K_{cam_name}' in intri, f"Missing K_{cam_name}"
            assert f'D_{cam_name}' in intri, f"Missing D_{cam_name}"
            assert f'Rot_{cam_name}' in extri, f"Missing Rot_{cam_name}"
            assert f'T_{cam_name}' in extri, f"Missing T_{cam_name}"


class TestRotationMatrixToQuaternion:
    """Tests for rotation_matrix_to_quaternion function."""

    def test_identity_rotation(self):
        """Test identity rotation matrix gives identity quaternion."""
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)

        # Identity quaternion is [1, 0, 0, 0]
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_quaternion_is_normalized(self):
        """Test that output quaternion has unit norm."""
        # Random rotation matrix (orthogonal)
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        q = rotation_matrix_to_quaternion(R)

        norm = np.linalg.norm(q)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_90_degree_rotation_x(self):
        """Test 90 degree rotation around X axis."""
        # R_x(90) = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        q = rotation_matrix_to_quaternion(R)

        # q for 90 deg around x: [cos(45), sin(45), 0, 0] = [sqrt(2)/2, sqrt(2)/2, 0, 0]
        expected = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0])
        np.testing.assert_allclose(np.abs(q), np.abs(expected), atol=1e-10)

    def test_90_degree_rotation_y(self):
        """Test 90 degree rotation around Y axis."""
        # R_y(90) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ])
        q = rotation_matrix_to_quaternion(R)

        # q for 90 deg around y: [sqrt(2)/2, 0, sqrt(2)/2, 0]
        expected = np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0])
        np.testing.assert_allclose(np.abs(q), np.abs(expected), atol=1e-10)

    def test_90_degree_rotation_z(self):
        """Test 90 degree rotation around Z axis."""
        # R_z(90) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        q = rotation_matrix_to_quaternion(R)

        # q for 90 deg around z: [sqrt(2)/2, 0, 0, sqrt(2)/2]
        expected = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])
        np.testing.assert_allclose(np.abs(q), np.abs(expected), atol=1e-10)

    def test_real_camera_rotation(self):
        """Test with actual camera rotation from dataset."""
        extri_path = os.path.join(RESOURCES_DIR, 'extri.yml')
        extri = parse_opencv_yaml(extri_path)

        R = extri['Rot_00']
        q = rotation_matrix_to_quaternion(R)

        # Check quaternion is unit norm
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-10)

        # Verify rotation matrix can be reconstructed from quaternion
        w, x, y, z = q
        R_reconstructed = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        np.testing.assert_allclose(R_reconstructed, R, atol=1e-6)


class TestWriteCamerasBinary:
    """Tests for write_cameras_binary function."""

    def test_write_single_camera(self):
        """Test writing a single camera to binary file."""
        cameras = {
            1: {
                'model_id': 4,  # OPENCV
                'width': 1920,
                'height': 1080,
                'params': [1000.0, 1000.0, 960.0, 540.0, 0.1, 0.01, 0.001, 0.0001]
            }
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            path = f.name

        try:
            write_cameras_binary(cameras, path)

            # Read and verify
            with open(path, 'rb') as f:
                num_cameras = struct.unpack('<Q', f.read(8))[0]
                assert num_cameras == 1

                cam_id = struct.unpack('<I', f.read(4))[0]
                assert cam_id == 1

                model_id = struct.unpack('<i', f.read(4))[0]
                assert model_id == 4

                width = struct.unpack('<Q', f.read(8))[0]
                assert width == 1920

                height = struct.unpack('<Q', f.read(8))[0]
                assert height == 1080

                params = [struct.unpack('<d', f.read(8))[0] for _ in range(8)]
                np.testing.assert_allclose(params, cameras[1]['params'])
        finally:
            os.unlink(path)

    def test_write_multiple_cameras(self):
        """Test writing multiple cameras to binary file."""
        cameras = {
            1: {'model_id': 4, 'width': 1920, 'height': 1080, 'params': [1000.0]*8},
            2: {'model_id': 4, 'width': 1920, 'height': 1080, 'params': [2000.0]*8},
            3: {'model_id': 4, 'width': 1920, 'height': 1080, 'params': [3000.0]*8},
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            path = f.name

        try:
            write_cameras_binary(cameras, path)

            with open(path, 'rb') as f:
                num_cameras = struct.unpack('<Q', f.read(8))[0]
                assert num_cameras == 3
        finally:
            os.unlink(path)


class TestWriteImagesBinary:
    """Tests for write_images_binary function."""

    def test_write_single_image(self):
        """Test writing a single image to binary file."""
        images = {
            1: {
                'qvec': np.array([1.0, 0.0, 0.0, 0.0]),
                'tvec': np.array([1.0, 2.0, 3.0]),
                'camera_id': 1,
                'name': 'test_image.jpg'
            }
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            path = f.name

        try:
            write_images_binary(images, path)

            with open(path, 'rb') as f:
                num_images = struct.unpack('<Q', f.read(8))[0]
                assert num_images == 1

                img_id = struct.unpack('<I', f.read(4))[0]
                assert img_id == 1

                qvec = [struct.unpack('<d', f.read(8))[0] for _ in range(4)]
                np.testing.assert_allclose(qvec, [1.0, 0.0, 0.0, 0.0])

                tvec = [struct.unpack('<d', f.read(8))[0] for _ in range(3)]
                np.testing.assert_allclose(tvec, [1.0, 2.0, 3.0])

                camera_id = struct.unpack('<I', f.read(4))[0]
                assert camera_id == 1

                # Read null-terminated string
                name_bytes = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name_bytes += char
                assert name_bytes.decode('utf-8') == 'test_image.jpg'

                num_points2d = struct.unpack('<Q', f.read(8))[0]
                assert num_points2d == 0
        finally:
            os.unlink(path)

    def test_write_multiple_images(self):
        """Test writing multiple images to binary file."""
        images = {
            1: {'qvec': np.array([1.0, 0.0, 0.0, 0.0]), 'tvec': np.zeros(3), 'camera_id': 1, 'name': 'img1.jpg'},
            2: {'qvec': np.array([1.0, 0.0, 0.0, 0.0]), 'tvec': np.zeros(3), 'camera_id': 1, 'name': 'img2.jpg'},
            3: {'qvec': np.array([1.0, 0.0, 0.0, 0.0]), 'tvec': np.zeros(3), 'camera_id': 2, 'name': 'img3.jpg'},
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            path = f.name

        try:
            write_images_binary(images, path)

            with open(path, 'rb') as f:
                num_images = struct.unpack('<Q', f.read(8))[0]
                assert num_images == 3
        finally:
            os.unlink(path)


class TestWritePoints3dBinary:
    """Tests for write_points3d_binary function."""

    def test_write_empty_points(self):
        """Test writing empty points3D.bin file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            path = f.name

        try:
            write_points3d_binary(path)

            with open(path, 'rb') as f:
                num_points = struct.unpack('<Q', f.read(8))[0]
                assert num_points == 0

                # File should only contain the count
                remaining = f.read()
                assert len(remaining) == 0
        finally:
            os.unlink(path)


class TestEndToEndConversion:
    """End-to-end tests for the full conversion pipeline."""

    @pytest.fixture
    def output_dir(self):
        """Create a temporary output directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_full_conversion_with_pycolmap(self, output_dir):
        """Test full conversion and verify with pycolmap."""
        pytest.importorskip('pycolmap')
        pytest.importorskip('cv2')
        import pycolmap
        import cv2

        # Parse calibration
        intri = parse_opencv_yaml(os.path.join(RESOURCES_DIR, 'intri.yml'))
        extri = parse_opencv_yaml(os.path.join(RESOURCES_DIR, 'extri.yml'))

        camera_names = intri['names']

        # Get image dimensions
        images_dir = os.path.join(RESOURCES_DIR, 'images')
        first_cam_dir = os.path.join(images_dir, camera_names[0])
        first_image = sorted([f for f in os.listdir(first_cam_dir) if f.endswith('.jpg')])[0]
        sample_img = cv2.imread(os.path.join(first_cam_dir, first_image))
        img_height, img_width = sample_img.shape[:2]

        # Build cameras dict
        cameras = {}
        for cam_idx, cam_name in enumerate(camera_names):
            cam_id = cam_idx + 1
            K = intri[f'K_{cam_name}']
            D = intri.get(f'D_{cam_name}')

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            if D is not None and D.size >= 4:
                k1, k2, p1, p2 = D.flatten()[:4]
            else:
                k1, k2, p1, p2 = 0, 0, 0, 0

            cameras[cam_id] = {
                'model_id': 4,
                'width': img_width,
                'height': img_height,
                'params': [fx, fy, cx, cy, k1, k2, p1, p2]
            }

        # Build images dict (single frame for test)
        images = {}
        image_id = 1
        frame_idx = 0

        for cam_idx, cam_name in enumerate(camera_names):
            cam_id = cam_idx + 1
            R = extri[f'Rot_{cam_name}']
            T = extri[f'T_{cam_name}']

            R_w2c = R.T
            t_w2c = -R_w2c @ T.flatten()
            qvec = rotation_matrix_to_quaternion(R_w2c)

            images[image_id] = {
                'qvec': qvec,
                'tvec': t_w2c,
                'camera_id': cam_id,
                'name': f'cam{cam_name}_frame{frame_idx:06d}.jpg'
            }
            image_id += 1

        # Write files
        sparse_dir = os.path.join(output_dir, 'sparse', '0')
        os.makedirs(sparse_dir)

        write_cameras_binary(cameras, os.path.join(sparse_dir, 'cameras.bin'))
        write_images_binary(images, os.path.join(sparse_dir, 'images.bin'))
        write_points3d_binary(os.path.join(sparse_dir, 'points3D.bin'))

        # Verify with pycolmap — support both the new Reconstruction API
        # (pycolmap >= 3.10) and the older SceneManager API (the JasonLSC fork
        # pinned in pyproject.toml only provides SceneManager).
        K_expected = intri['K_00']
        if hasattr(pycolmap, 'Reconstruction'):
            rec = pycolmap.Reconstruction(sparse_dir)
            cameras_out = rec.cameras
            images_out = rec.images
            points_out = rec.points3D
            cam = cameras_out[1]
            cam_model_name = cam.model_name
            cam_width, cam_height = cam.width, cam.height
            cam_fx, cam_fy, cam_cx, cam_cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        else:
            manager = pycolmap.SceneManager(sparse_dir)
            manager.load_cameras()
            manager.load_images()
            manager.load_points3D()
            cameras_out = manager.cameras
            images_out = manager.images
            points_out = manager.points3D
            cam = cameras_out[1]
            cam_model_name = pycolmap.Camera.GetNameFromType(cam.camera_type)
            cam_width, cam_height = cam.width, cam.height
            cam_fx, cam_fy, cam_cx, cam_cy = cam.fx, cam.fy, cam.cx, cam.cy

        assert len(cameras_out) == 18
        assert len(images_out) == 18
        assert len(points_out) == 0

        # Check camera model
        assert cam_model_name == 'OPENCV'
        assert cam_width == img_width
        assert cam_height == img_height

        # Check intrinsics match
        assert np.isclose(cam_fx, K_expected[0, 0], rtol=1e-5)
        assert np.isclose(cam_fy, K_expected[1, 1], rtol=1e-5)
        assert np.isclose(cam_cx, K_expected[0, 2], rtol=1e-5)
        assert np.isclose(cam_cy, K_expected[1, 2], rtol=1e-5)

    def test_conversion_creates_correct_directory_structure(self, output_dir):
        """Test that conversion creates the expected directory structure."""
        sparse_dir = os.path.join(output_dir, 'sparse', '0')
        os.makedirs(sparse_dir)

        cameras = {1: {'model_id': 4, 'width': 100, 'height': 100, 'params': [100.0]*8}}
        images = {1: {'qvec': np.array([1.0, 0, 0, 0]), 'tvec': np.zeros(3), 'camera_id': 1, 'name': 'test.jpg'}}

        write_cameras_binary(cameras, os.path.join(sparse_dir, 'cameras.bin'))
        write_images_binary(images, os.path.join(sparse_dir, 'images.bin'))
        write_points3d_binary(os.path.join(sparse_dir, 'points3D.bin'))

        assert os.path.exists(os.path.join(sparse_dir, 'cameras.bin'))
        assert os.path.exists(os.path.join(sparse_dir, 'images.bin'))
        assert os.path.exists(os.path.join(sparse_dir, 'points3D.bin'))


class TestCLIIntegration:
    """Tests for command-line interface."""

    @pytest.fixture
    def output_dir(self):
        """Create a temporary output directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_cli_single_frame(self, output_dir):
        """Test CLI with --single-frame option."""
        import subprocess

        script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'convert_calibration_to_colmap.py')

        result = subprocess.run([
            sys.executable, script_path,
            '--input-dir', RESOURCES_DIR,
            '--output-dir', output_dir,
            '--single-frame', '0'
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify output files exist
        assert os.path.exists(os.path.join(output_dir, 'sparse', '0', 'cameras.bin'))
        assert os.path.exists(os.path.join(output_dir, 'sparse', '0', 'images.bin'))
        assert os.path.exists(os.path.join(output_dir, 'sparse', '0', 'points3D.bin'))

    def test_cli_full_conversion(self, output_dir):
        """Test CLI full conversion."""
        import subprocess

        script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'convert_calibration_to_colmap.py')

        result = subprocess.run([
            sys.executable, script_path,
            '--input-dir', RESOURCES_DIR,
            '--output-dir', output_dir,
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Check output contains expected messages
        assert 'Found 18 cameras' in result.stdout
        assert 'COLMAP sparse reconstruction written to' in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
