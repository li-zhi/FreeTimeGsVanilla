#!/usr/bin/env python3
"""
Tests for combine_frames_fast_keyframes.py

Uses sample data from tests/resources/triangulate_data/ (frames 0, 1, 2, 5, 6, 10).
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from combine_frames_fast_keyframes import (
    load_frame_data,
    compute_velocity_knn,
    estimate_scene_scale,
    smart_density_velocity_sampling,
)


# Path to test data (triangulated point clouds: frames 0, 1, 2, 5, 6, 10)
TRIANGULATE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'resources', 'triangulate_data')


@pytest.fixture(scope="module")
def frame_dir():
    """Return the path to the triangulate data directory."""
    if not os.path.exists(TRIANGULATE_DATA_DIR):
        pytest.skip("Test data not available in tests/resources/triangulate_data")
    # Check if at least frame 0 exists
    if not os.path.exists(os.path.join(TRIANGULATE_DATA_DIR, 'points3d_frame000000.npy')):
        pytest.skip("Frame 0 not found in tests/resources/triangulate_data")
    return TRIANGULATE_DATA_DIR


class TestLoadFrameData:
    """Tests for load_frame_data function."""

    def test_load_existing_frame(self, frame_dir):
        """Test loading an existing frame returns positions and colors."""
        from pathlib import Path
        positions, colors = load_frame_data(Path(frame_dir), 0)

        assert positions is not None
        assert colors is not None
        assert positions.ndim == 2
        assert positions.shape[1] == 3
        assert colors.shape == positions.shape

    def test_load_nonexistent_frame(self, frame_dir):
        """Test loading a non-existent frame returns None."""
        from pathlib import Path
        positions, colors = load_frame_data(Path(frame_dir), 9999)

        assert positions is None
        assert colors is None

    def test_load_frame_dtypes(self, frame_dir):
        """Test that loaded data has correct dtype."""
        from pathlib import Path
        positions, colors = load_frame_data(Path(frame_dir), 0)

        assert positions.dtype == np.float32
        assert colors.dtype == np.float32

    def test_load_frame_valid_values(self, frame_dir):
        """Test that loaded data has finite values."""
        from pathlib import Path
        positions, colors = load_frame_data(Path(frame_dir), 0)

        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(colors))


class TestComputeVelocityKnn:
    """Tests for compute_velocity_knn function."""

    def test_velocity_same_points(self):
        """Test velocity is zero when points don't move."""
        pos_t = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        pos_t1 = pos_t.copy()

        velocities, valid_mask = compute_velocity_knn(pos_t, pos_t1, max_distance=0.5)

        assert velocities.shape == pos_t.shape
        assert valid_mask.shape == (len(pos_t),)
        assert np.all(valid_mask)  # All should be valid
        assert np.allclose(velocities, 0)  # No movement

    def test_velocity_uniform_motion(self):
        """Test velocity computation with known uniform motion."""
        pos_t = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        pos_t1 = pos_t + np.array([0.1, 0, 0])  # Move all points by 0.1 in x

        velocities, valid_mask = compute_velocity_knn(pos_t, pos_t1, max_distance=0.5)

        assert np.all(valid_mask)
        expected_vel = np.array([[0.1, 0, 0]] * 3)
        assert np.allclose(velocities, expected_vel, atol=1e-5)

    def test_velocity_max_distance_filter(self):
        """Test that points beyond max_distance are filtered out."""
        pos_t = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pos_t1 = np.array([[0.1, 0, 0], [5, 0, 0]], dtype=np.float32)  # Second point moved far

        velocities, valid_mask = compute_velocity_knn(pos_t, pos_t1, max_distance=0.5)

        assert valid_mask[0] == True   # First point is valid
        assert valid_mask[1] == False  # Second point moved too far

    def test_velocity_empty_input(self):
        """Test velocity computation with empty arrays."""
        pos_t = np.array([], dtype=np.float32).reshape(0, 3)
        pos_t1 = np.array([], dtype=np.float32).reshape(0, 3)

        velocities, valid_mask = compute_velocity_knn(pos_t, pos_t1, max_distance=0.5)

        assert velocities.shape == (0, 3)
        assert valid_mask.shape == (0,)

    def test_velocity_real_data(self, frame_dir):
        """Test velocity computation on real triangulated data."""
        from pathlib import Path
        pos_t, _ = load_frame_data(Path(frame_dir), 0)
        pos_t1, _ = load_frame_data(Path(frame_dir), 1)

        if pos_t is None or pos_t1 is None:
            pytest.skip("Required frames not available")

        velocities, valid_mask = compute_velocity_knn(pos_t, pos_t1, max_distance=0.5, k=1)

        assert velocities.shape == pos_t.shape
        assert valid_mask.dtype == bool
        # Most points should have valid velocity matches
        assert valid_mask.sum() > len(pos_t) * 0.5  # At least 50% should match

    def test_velocity_k_neighbors(self):
        """Test velocity computation with k > 1."""
        pos_t = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pos_t1 = np.array([[0.1, 0, 0], [1.1, 0, 0], [0.15, 0, 0]], dtype=np.float32)

        velocities, valid_mask = compute_velocity_knn(pos_t, pos_t1, max_distance=0.5, k=2)

        # Should still work with k > 1
        assert velocities.shape == pos_t.shape
        assert valid_mask.shape == (len(pos_t),)


class TestEstimateSceneScale:
    """Tests for estimate_scene_scale function."""

    def test_estimate_scale_unit_cube(self):
        """Test scene scale estimation on a unit cube."""
        # Points in a unit cube [0, 1]^3
        np.random.seed(42)
        positions = np.random.uniform(0, 1, size=(1000, 3)).astype(np.float32)

        stats = estimate_scene_scale(positions)

        assert 'bbox_diagonal' in stats
        assert 'median_nn_dist' in stats
        assert 'suggested_voxel_size' in stats

        # Diagonal of unit cube is sqrt(3) ≈ 1.73
        assert np.isclose(stats['bbox_diagonal'], np.sqrt(3), rtol=0.1)

    def test_estimate_scale_large_scene(self):
        """Test scene scale estimation on a larger scene."""
        np.random.seed(42)
        positions = np.random.uniform(-50, 50, size=(5000, 3)).astype(np.float32)

        stats = estimate_scene_scale(positions)

        # Diagonal should be around sqrt(3) * 100 ≈ 173
        assert stats['bbox_diagonal'] > 100
        assert stats['suggested_voxel_size'] > 0.01

    def test_estimate_scale_empty(self):
        """Test scene scale estimation with empty input."""
        positions = np.array([], dtype=np.float32).reshape(0, 3)

        stats = estimate_scene_scale(positions)

        # Should return default values
        assert stats['bbox_diagonal'] == 1.0
        assert stats['suggested_voxel_size'] == 0.1

    def test_estimate_scale_clamping(self):
        """Test that voxel size is clamped to [0.01, 1.0]."""
        # Very dense points (should clamp voxel size to minimum)
        np.random.seed(42)
        positions = np.random.uniform(0, 0.001, size=(1000, 3)).astype(np.float32)

        stats = estimate_scene_scale(positions)

        assert stats['suggested_voxel_size'] >= 0.01
        assert stats['suggested_voxel_size'] <= 1.0

    def test_estimate_scale_real_data(self, frame_dir):
        """Test scene scale estimation on real data."""
        from pathlib import Path
        positions, _ = load_frame_data(Path(frame_dir), 0)

        if positions is None:
            pytest.skip("Frame 0 not available")

        stats = estimate_scene_scale(positions)

        assert stats['bbox_diagonal'] > 0
        assert stats['median_nn_dist'] > 0
        assert 0.01 <= stats['suggested_voxel_size'] <= 1.0


class TestSmartDensityVelocitySampling:
    """Tests for smart_density_velocity_sampling function."""

    def test_sampling_returns_correct_count(self):
        """Test that sampling returns the requested number of points."""
        np.random.seed(42)
        n_points = 10000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32) * 0.1
        colors = np.random.randint(0, 255, (n_points, 3)).astype(np.float32)

        target_count = 1000
        pos_s, vel_s, col_s, idx = smart_density_velocity_sampling(
            positions, velocities, colors, target_count, seed=42
        )

        assert len(pos_s) == target_count
        assert len(vel_s) == target_count
        assert len(col_s) == target_count
        assert len(idx) == target_count

    def test_sampling_preserves_data(self):
        """Test that sampled data comes from original arrays."""
        np.random.seed(42)
        n_points = 1000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32) * 0.1
        colors = np.random.randint(0, 255, (n_points, 3)).astype(np.float32)

        target_count = 100
        pos_s, vel_s, col_s, idx = smart_density_velocity_sampling(
            positions, velocities, colors, target_count, seed=42
        )

        # Check that sampled values match original at indices
        assert np.allclose(pos_s, positions[idx])
        assert np.allclose(vel_s, velocities[idx])
        assert np.allclose(col_s, colors[idx])

    def test_sampling_unique_indices(self):
        """Test that sampled indices are unique."""
        np.random.seed(42)
        n_points = 1000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32) * 0.1
        colors = np.random.randint(0, 255, (n_points, 3)).astype(np.float32)

        target_count = 500
        _, _, _, idx = smart_density_velocity_sampling(
            positions, velocities, colors, target_count, seed=42
        )

        assert len(np.unique(idx)) == target_count

    def test_sampling_returns_all_when_under_budget(self):
        """Test that all points are returned when count < budget."""
        n_points = 100
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (n_points, 3)).astype(np.float32)

        target_count = 1000  # More than we have
        pos_s, vel_s, col_s, idx = smart_density_velocity_sampling(
            positions, velocities, colors, target_count
        )

        assert len(pos_s) == n_points  # Returns all
        assert np.array_equal(idx, np.arange(n_points))

    def test_sampling_velocity_bias(self):
        """Test that high-velocity points are more likely to be sampled."""
        np.random.seed(42)
        n_points = 10000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.zeros((n_points, 3), dtype=np.float32)
        colors = np.random.randint(0, 255, (n_points, 3)).astype(np.float32)

        # Give first 100 points high velocity
        velocities[:100] = np.array([1.0, 0, 0])

        target_count = 500
        _, vel_s, _, idx = smart_density_velocity_sampling(
            positions, velocities, colors, target_count,
            velocity_weight=10.0, seed=42
        )

        # High velocity points should be over-represented
        high_vel_sampled = np.sum(idx < 100)
        expected_random = target_count * (100 / n_points)  # 5 expected randomly

        assert high_vel_sampled > expected_random * 2  # At least 2x oversampled

    def test_sampling_reproducibility(self):
        """Test that same seed produces same results."""
        n_points = 1000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (n_points, 3)).astype(np.float32)

        target_count = 100

        _, _, _, idx1 = smart_density_velocity_sampling(
            positions, velocities, colors, target_count, seed=123
        )
        _, _, _, idx2 = smart_density_velocity_sampling(
            positions, velocities, colors, target_count, seed=123
        )

        assert np.array_equal(idx1, idx2)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_basic(self, frame_dir):
        """Test running the full script with basic settings."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, 'test_keyframes.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '10',
                '--keyframe-step', '5',
            ], capture_output=True, text=True, timeout=120)

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert os.path.exists(output_path)

            # Load and verify output
            data = np.load(output_path)
            assert 'positions' in data
            assert 'velocities' in data
            assert 'colors' in data
            assert 'times' in data
            assert 'durations' in data
            assert 'has_velocity' in data

    def test_full_pipeline_with_sampling(self, frame_dir):
        """Test running the full script with smart sampling."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, 'test_keyframes_sampled.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '10',
                '--keyframe-step', '5',
                '--use-smart-sampling',
                '--total-budget', '10000',
            ], capture_output=True, text=True, timeout=120)

            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert os.path.exists(output_path)

            # Load and verify output
            data = np.load(output_path)
            # With budget of 10000, should have at most that many points
            assert len(data['positions']) <= 10000

    def test_output_npz_format(self, frame_dir):
        """Test that output NPZ has all expected fields with correct shapes."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, 'test_format.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '10',
                '--keyframe-step', '5',
            ], capture_output=True, text=True, timeout=120)

            assert result.returncode == 0, f"Script failed: {result.stderr}"

            data = np.load(output_path)
            n_points = len(data['positions'])

            # Check shapes
            assert data['positions'].shape == (n_points, 3)
            assert data['velocities'].shape == (n_points, 3)
            assert data['colors'].shape == (n_points, 3)
            assert data['times'].shape == (n_points, 1)
            assert data['durations'].shape == (n_points, 1)
            assert data['has_velocity'].shape == (n_points,)

            # Check dtypes
            assert data['positions'].dtype == np.float32
            assert data['velocities'].dtype == np.float32
            assert data['colors'].dtype == np.float32
            assert data['times'].dtype == np.float32
            assert data['durations'].dtype == np.float32
            assert data['has_velocity'].dtype == bool

    def test_output_time_values(self, frame_dir):
        """Test that output time values are correctly normalized."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, 'test_times.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '10',
                '--keyframe-step', '5',
            ], capture_output=True, text=True, timeout=120)

            assert result.returncode == 0, f"Script failed: {result.stderr}"

            data = np.load(output_path)
            times = data['times']

            # Times should be in [0, 1] range
            assert times.min() >= 0.0
            assert times.max() <= 1.0

            # With keyframe_step=5 and frame_end=10, we should have keyframes at 0, 5, 10
            # Normalized times: 0/11, 5/11, 10/11
            unique_times = np.unique(times)
            assert len(unique_times) >= 2  # At least 2 keyframes

    def test_output_colors_normalized(self, frame_dir):
        """Test that output colors are normalized to [0, 1]."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, 'test_colors.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '10',
                '--keyframe-step', '5',
            ], capture_output=True, text=True, timeout=120)

            assert result.returncode == 0, f"Script failed: {result.stderr}"

            data = np.load(output_path)
            colors = data['colors']

            # Colors should be normalized to [0, 1]
            assert colors.min() >= 0.0
            assert colors.max() <= 1.0

    def test_metadata_saved(self, frame_dir):
        """Test that metadata is saved in the NPZ file."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, 'test_metadata.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '10',
                '--keyframe-step', '5',
            ], capture_output=True, text=True, timeout=120)

            assert result.returncode == 0, f"Script failed: {result.stderr}"

            data = np.load(output_path)

            # Check metadata fields
            assert 'frame_start' in data
            assert 'frame_end' in data
            assert 'keyframe_step' in data
            assert int(data['frame_start']) == 0
            assert int(data['frame_end']) == 10
            assert int(data['keyframe_step']) == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_intermediate_frame(self, frame_dir):
        """Test handling when an intermediate frame is missing."""
        from pathlib import Path

        # Frame 3 is not in our test set, so keyframes 0, 5, 10 should still work
        # but velocity computation for frame 5 might use frame 6
        pos, _ = load_frame_data(Path(frame_dir), 5)
        assert pos is not None  # Frame 5 exists

        pos_next, _ = load_frame_data(Path(frame_dir), 6)
        assert pos_next is not None  # Frame 6 exists for velocity

    def test_single_frame(self, frame_dir):
        """Test with only a single frame (no velocity possible)."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            # Copy only frame 0
            single_frame_dir = os.path.join(output_dir, 'single')
            os.makedirs(single_frame_dir)
            shutil.copy(
                os.path.join(frame_dir, 'points3d_frame000000.npy'),
                single_frame_dir
            )
            shutil.copy(
                os.path.join(frame_dir, 'colors_frame000000.npy'),
                single_frame_dir
            )

            output_path = os.path.join(output_dir, 'single_frame.npz')

            result = subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', single_frame_dir,
                '--output-path', output_path,
                '--frame-start', '0',
                '--frame-end', '0',
                '--keyframe-step', '1',
            ], capture_output=True, text=True, timeout=60)

            assert result.returncode == 0
            assert os.path.exists(output_path)

            data = np.load(output_path)
            # Velocities should be zero (no next frame)
            assert np.allclose(data['velocities'], 0)
            # No valid velocity
            assert np.sum(data['has_velocity']) == 0

    def test_sample_ratio(self, frame_dir):
        """Test that sample_ratio parameter reduces point count."""
        import subprocess

        with tempfile.TemporaryDirectory() as output_dir:
            # Full points
            output_full = os.path.join(output_dir, 'full.npz')
            subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_full,
                '--frame-start', '0',
                '--frame-end', '5',
                '--keyframe-step', '5',
                '--sample-ratio', '1.0',
            ], capture_output=True, text=True, timeout=60)

            # Half points
            output_half = os.path.join(output_dir, 'half.npz')
            subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'combine_frames_fast_keyframes.py'),
                '--input-dir', frame_dir,
                '--output-path', output_half,
                '--frame-start', '0',
                '--frame-end', '5',
                '--keyframe-step', '5',
                '--sample-ratio', '0.5',
            ], capture_output=True, text=True, timeout=60)

            data_full = np.load(output_full)
            data_half = np.load(output_half)

            # Half should have approximately half the points
            ratio = len(data_half['positions']) / len(data_full['positions'])
            assert 0.4 < ratio < 0.6  # Allow some variance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
