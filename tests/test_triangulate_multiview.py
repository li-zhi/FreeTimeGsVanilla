#!/usr/bin/env python3
"""
Tests for triangulate_multiview.py

Uses sample data from tests/resources/actor1_4_subseq/
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from triangulate_multiview import (
    parse_opencv_yaml,
    load_cameras,
    triangulate_points_dlt,
    match_features_sift,
    process_frame,
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


class TestLoadCameras:
    """Tests for load_cameras function."""

    def test_load_cameras_returns_correct_structure(self):
        """Test that load_cameras returns intrinsics, extrinsics, and names."""
        intrinsics, extrinsics, cam_names = load_cameras(RESOURCES_DIR)

        assert isinstance(intrinsics, dict)
        assert isinstance(extrinsics, dict)
        assert isinstance(cam_names, list)

    def test_load_cameras_count(self):
        """Test correct number of cameras loaded."""
        intrinsics, extrinsics, cam_names = load_cameras(RESOURCES_DIR)

        assert len(cam_names) == 18
        assert len(intrinsics) == 18
        assert len(extrinsics) == 18

    def test_intrinsics_shape(self):
        """Test intrinsic matrices have correct shape."""
        intrinsics, _, _ = load_cameras(RESOURCES_DIR)

        for cam, K in intrinsics.items():
            assert K.shape == (3, 3), f"Camera {cam} K matrix has wrong shape"

    def test_extrinsics_structure(self):
        """Test extrinsics are (R, t) tuples with correct shapes."""
        _, extrinsics, _ = load_cameras(RESOURCES_DIR)

        for cam, (R, t) in extrinsics.items():
            assert R.shape == (3, 3), f"Camera {cam} R matrix has wrong shape"
            assert t.shape == (3,), f"Camera {cam} t vector has wrong shape"


class TestTriangulatePointsDlt:
    """Tests for triangulate_points_dlt function."""

    def test_triangulate_simple_case(self):
        """Test triangulation with simple known geometry."""
        # Create two cameras looking at origin
        # Camera 1 at (0, 0, 5) looking at origin
        K = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        R1 = np.eye(3)
        t1 = np.array([0, 0, 5])
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])

        # Camera 2 at (2, 0, 5) looking at origin
        R2 = np.eye(3)
        t2 = np.array([2, 0, 5])
        P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

        # Project point (0, 0, 0) to both cameras
        world_point = np.array([0, 0, 0, 1])
        proj1 = P1 @ world_point
        proj1 = proj1[:2] / proj1[2]
        proj2 = P2 @ world_point
        proj2 = proj2[:2] / proj2[2]

        pts1 = proj1.reshape(1, 2)
        pts2 = proj2.reshape(1, 2)

        # Triangulate
        result = triangulate_points_dlt(pts1, pts2, P1, P2)

        assert result.shape == (1, 3)
        # Should recover approximately (0, 0, 0)
        assert np.allclose(result[0], [0, 0, 0], atol=0.1)

    def test_triangulate_multiple_points(self):
        """Test triangulation with multiple points."""
        K = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        R1 = np.eye(3)
        t1 = np.array([0, 0, 10])
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])

        R2 = np.eye(3)
        t2 = np.array([2, 0, 10])
        P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

        # Multiple world points
        world_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])

        # Project to both cameras
        pts1_list = []
        pts2_list = []
        for wp in world_points:
            wp_h = np.append(wp, 1)
            p1 = P1 @ wp_h
            p1 = p1[:2] / p1[2]
            p2 = P2 @ wp_h
            p2 = p2[:2] / p2[2]
            pts1_list.append(p1)
            pts2_list.append(p2)

        pts1 = np.array(pts1_list)
        pts2 = np.array(pts2_list)

        result = triangulate_points_dlt(pts1, pts2, P1, P2)

        assert result.shape == (3, 3)
        for i in range(3):
            assert np.allclose(result[i], world_points[i], atol=0.1)


class TestMatchFeaturesSift:
    """Tests for match_features_sift function."""

    def test_match_same_image(self):
        """Test matching an image with itself returns many matches."""
        img_path = os.path.join(RESOURCES_DIR, 'images', '00', '000000.jpg')
        if not os.path.exists(img_path):
            pytest.skip("Test image not found")

        import cv2
        img = cv2.imread(img_path)

        kpts1, kpts2, conf = match_features_sift(img, img, max_matches=1000)

        # Matching same image should give many matches
        assert len(kpts1) > 100
        assert len(kpts1) == len(kpts2)
        assert len(conf) == len(kpts1)

        # Points should be nearly identical
        assert np.allclose(kpts1, kpts2, atol=1.0)

    def test_match_different_views(self):
        """Test matching between different camera views."""
        img1_path = os.path.join(RESOURCES_DIR, 'images', '00', '000000.jpg')
        img2_path = os.path.join(RESOURCES_DIR, 'images', '01', '000000.jpg')

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            pytest.skip("Test images not found")

        import cv2
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        kpts1, kpts2, conf = match_features_sift(img1, img2, max_matches=5000)

        # Should find some matches between views
        assert len(kpts1) > 10
        assert kpts1.shape[1] == 2
        assert kpts2.shape[1] == 2
        assert all(0 <= c <= 1 for c in conf)

    def test_match_returns_empty_for_blank_image(self):
        """Test that blank images return empty matches."""
        blank = np.zeros((100, 100, 3), dtype=np.uint8)

        kpts1, kpts2, conf = match_features_sift(blank, blank)

        assert len(kpts1) == 0 or len(kpts1) < 10


class TestProcessFrame:
    """Tests for process_frame function."""

    def test_process_frame_generates_points(self):
        """Test that process_frame generates 3D points."""
        intrinsics, extrinsics, cam_names = load_cameras(RESOURCES_DIR)

        # Use only cameras that have images
        available_cams = ['00', '01', '02']

        with tempfile.TemporaryDirectory() as tmpdir:
            points3d, colors = process_frame(
                frame_idx=0,
                data_dir=RESOURCES_DIR,
                output_dir=tmpdir,
                cam_names=available_cams,
                intrinsics={k: intrinsics[k] for k in available_cams},
                extrinsics={k: extrinsics[k] for k in available_cams},
                roma_model=None,
                use_roma=False,
                device="cpu",
                confidence_threshold=0.3,
                max_matches_per_pair=1000,
                reprojection_threshold=10.0
            )

            # Should generate some points
            assert len(points3d) > 0
            assert points3d.shape[1] == 3
            assert colors.shape[1] == 3

    def test_process_frame_output_shapes(self):
        """Test that points and colors have matching shapes."""
        intrinsics, extrinsics, cam_names = load_cameras(RESOURCES_DIR)
        available_cams = ['00', '01', '02']

        with tempfile.TemporaryDirectory() as tmpdir:
            points3d, colors = process_frame(
                frame_idx=0,
                data_dir=RESOURCES_DIR,
                output_dir=tmpdir,
                cam_names=available_cams,
                intrinsics={k: intrinsics[k] for k in available_cams},
                extrinsics={k: extrinsics[k] for k in available_cams},
                roma_model=None,
                use_roma=False,
                device="cpu",
                confidence_threshold=0.3,
                max_matches_per_pair=1000,
                reprojection_threshold=10.0
            )

            assert points3d.shape[0] == colors.shape[0]

    def test_process_frame_colors_range(self):
        """Test that colors are in valid range [0, 255]."""
        intrinsics, extrinsics, cam_names = load_cameras(RESOURCES_DIR)
        available_cams = ['00', '01', '02']

        with tempfile.TemporaryDirectory() as tmpdir:
            points3d, colors = process_frame(
                frame_idx=0,
                data_dir=RESOURCES_DIR,
                output_dir=tmpdir,
                cam_names=available_cams,
                intrinsics={k: intrinsics[k] for k in available_cams},
                extrinsics={k: extrinsics[k] for k in available_cams},
                roma_model=None,
                use_roma=False,
                device="cpu",
                confidence_threshold=0.3,
                max_matches_per_pair=1000,
                reprojection_threshold=10.0
            )

            if len(colors) > 0:
                assert colors.min() >= 0
                assert colors.max() <= 255


class TestOutputFormat:
    """Tests for verifying output file format compatibility."""

    def test_output_matches_expected_format(self):
        """Test that generated output matches expected NPY format."""
        # Load the pre-generated output
        points_path = os.path.join(RESOURCES_DIR, 'points3d_frame000000.npy')
        colors_path = os.path.join(RESOURCES_DIR, 'colors_frame000000.npy')

        if not os.path.exists(points_path):
            pytest.skip("Pre-generated output not found")

        points = np.load(points_path)
        colors = np.load(colors_path)

        # Check shapes
        assert points.ndim == 2
        assert points.shape[1] == 3
        assert colors.ndim == 2
        assert colors.shape[1] == 3
        assert points.shape[0] == colors.shape[0]

        # Check dtypes
        assert points.dtype == np.float32
        assert colors.dtype == np.float32

    def test_output_has_reasonable_values(self):
        """Test that output values are reasonable."""
        points_path = os.path.join(RESOURCES_DIR, 'points3d_frame000000.npy')
        colors_path = os.path.join(RESOURCES_DIR, 'colors_frame000000.npy')

        if not os.path.exists(points_path):
            pytest.skip("Pre-generated output not found")

        points = np.load(points_path)
        colors = np.load(colors_path)

        # Points should be finite
        assert np.all(np.isfinite(points))

        # Colors should be in [0, 255]
        assert colors.min() >= 0
        assert colors.max() <= 255

        # Should have reasonable number of points
        assert len(points) > 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
