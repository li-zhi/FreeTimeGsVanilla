#!/usr/bin/env python3
"""
Tests for simple_trainer_freetime_4d_pure_relocation.py

Uses sample data from tests/resources/ for testing:
- tests/resources/actor1_4_subseq/ - calibration data
- tests/resources/triangulate_data/ - triangulated point cloud data
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_trainer_freetime_4d_pure_relocation import (
    Config,
    load_init_npz,
    load_init_npz_stratified,
    estimate_voxel_size,
    smart_sample_points,
    create_splats_with_optimizers_4d,
    generate_interpolated_path,
    _normalize,
    _viewmatrix,
)

# Path to test resources
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')
TRIANGULATE_DATA_DIR = os.path.join(RESOURCES_DIR, 'triangulate_data')
ACTOR_DATA_DIR = os.path.join(RESOURCES_DIR, 'actor1_4_subseq')


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_npz_path(tmp_path):
    """Create a sample NPZ file for testing."""
    n_points = 1000
    np.random.seed(42)

    # Create sample data
    positions = np.random.randn(n_points, 3).astype(np.float32) * 5
    velocities = np.random.randn(n_points, 3).astype(np.float32) * 0.1
    colors = np.random.rand(n_points, 3).astype(np.float32) * 255
    times = np.linspace(0, 1, n_points).astype(np.float32)
    durations = np.ones(n_points, dtype=np.float32) * 0.1

    npz_path = tmp_path / "test_init.npz"
    np.savez(
        npz_path,
        positions=positions,
        velocities=velocities,
        colors=colors,
        times=times,
        durations=durations,
        frame_start=0,
        frame_end=30,
    )
    return str(npz_path)


@pytest.fixture
def sample_npz_with_keyframes(tmp_path):
    """Create a sample NPZ file with keyframe structure."""
    np.random.seed(42)

    # 5 keyframes with 200 points each
    keyframe_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    points_per_keyframe = 200

    all_positions = []
    all_velocities = []
    all_colors = []
    all_times = []
    all_durations = []

    for t in keyframe_times:
        positions = np.random.randn(points_per_keyframe, 3).astype(np.float32) * 5
        velocities = np.random.randn(points_per_keyframe, 3).astype(np.float32) * 0.1
        colors = np.random.rand(points_per_keyframe, 3).astype(np.float32) * 255
        times = np.full(points_per_keyframe, t, dtype=np.float32)
        durations = np.ones(points_per_keyframe, dtype=np.float32) * 0.2

        all_positions.append(positions)
        all_velocities.append(velocities)
        all_colors.append(colors)
        all_times.append(times)
        all_durations.append(durations)

    positions = np.vstack(all_positions)
    velocities = np.vstack(all_velocities)
    colors = np.vstack(all_colors)
    times = np.concatenate(all_times)
    durations = np.concatenate(all_durations)

    npz_path = tmp_path / "test_keyframe.npz"
    np.savez(
        npz_path,
        positions=positions,
        velocities=velocities,
        colors=colors,
        times=times,
        durations=durations,
        frame_start=0,
        frame_end=30,
        keyframe_step=5,
    )
    return str(npz_path)


@pytest.fixture
def real_triangulate_npz(tmp_path):
    """Create NPZ from real triangulated data if available."""
    if not os.path.exists(TRIANGULATE_DATA_DIR):
        pytest.skip("Triangulate test data not available")

    # Load a few frames of triangulated data
    all_positions = []
    all_colors = []
    all_times = []

    for frame_idx in [0, 5, 10]:
        points_file = os.path.join(TRIANGULATE_DATA_DIR, f'points3d_frame{frame_idx:06d}.npy')
        colors_file = os.path.join(TRIANGULATE_DATA_DIR, f'colors_frame{frame_idx:06d}.npy')

        if os.path.exists(points_file) and os.path.exists(colors_file):
            points = np.load(points_file)
            colors = np.load(colors_file)
            times = np.full(len(points), frame_idx / 30.0, dtype=np.float32)

            all_positions.append(points)
            all_colors.append(colors)
            all_times.append(times)

    if not all_positions:
        pytest.skip("No triangulate data files found")

    positions = np.vstack(all_positions).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.float32)
    times = np.concatenate(all_times)

    # Estimate velocities as zero (no motion between keyframes in test)
    velocities = np.zeros_like(positions)
    durations = np.ones(len(positions), dtype=np.float32) * 0.2

    npz_path = tmp_path / "real_triangulate.npz"
    np.savez(
        npz_path,
        positions=positions,
        velocities=velocities,
        colors=colors,
        times=times,
        durations=durations,
        frame_start=0,
        frame_end=30,
    )
    return str(npz_path)


# ============================================================
# Tests for Config dataclass
# ============================================================

class TestConfig:
    """Tests for Config dataclass."""

    def test_default_config(self):
        """Test default config values."""
        cfg = Config()

        assert cfg.max_steps == 70_000
        assert cfg.batch_size == 1
        assert cfg.sh_degree == 3
        assert cfg.init_opacity == 0.5
        assert cfg.lambda_img == 0.8
        assert cfg.lambda_ssim == 0.2

    def test_adjust_steps(self):
        """Test step adjustment with factor."""
        cfg = Config()
        original_max_steps = cfg.max_steps
        original_eval_steps = cfg.eval_steps.copy()

        cfg.adjust_steps(0.5)

        assert cfg.max_steps == original_max_steps // 2
        assert cfg.eval_steps == [s // 2 for s in original_eval_steps]

    def test_adjust_steps_scaling(self):
        """Test that all step-related params scale correctly."""
        cfg = Config()
        cfg.densification_start_step = 1000
        cfg.reg_4d_start_step = 500
        cfg.relocation_stop_iter = 50000

        cfg.adjust_steps(2.0)

        assert cfg.densification_start_step == 2000
        assert cfg.reg_4d_start_step == 1000
        assert cfg.relocation_stop_iter == 100000

    def test_config_with_custom_values(self):
        """Test config with custom values."""
        cfg = Config(
            max_steps=10000,
            batch_size=4,
            init_duration=0.3,
            velocity_lr_start=1e-3,
        )

        assert cfg.max_steps == 10000
        assert cfg.batch_size == 4
        assert cfg.init_duration == 0.3
        assert cfg.velocity_lr_start == 1e-3


# ============================================================
# Tests for load_init_npz
# ============================================================

class TestLoadInitNpz:
    """Tests for load_init_npz function."""

    def test_load_basic(self, sample_npz_path):
        """Test basic NPZ loading."""
        result = load_init_npz(
            sample_npz_path,
            max_samples=500,
            frame_start=0,
            frame_end=30,
        )

        assert 'positions' in result
        assert 'velocities' in result
        assert 'colors' in result
        assert 'times' in result
        assert 'durations' in result

        assert isinstance(result['positions'], torch.Tensor)
        assert result['positions'].shape[1] == 3
        assert result['velocities'].shape[1] == 3
        assert result['colors'].shape[1] == 3

    def test_sampling_reduces_points(self, sample_npz_path):
        """Test that sampling reduces point count."""
        result = load_init_npz(
            sample_npz_path,
            max_samples=100,
            frame_start=0,
            frame_end=30,
        )

        assert len(result['positions']) <= 100

    def test_no_sampling_keeps_all_points(self, sample_npz_path):
        """Test that max_samples=0 keeps all points."""
        result = load_init_npz(
            sample_npz_path,
            max_samples=0,  # Disable sampling
            frame_start=0,
            frame_end=30,
        )

        # Should have all 1000 points
        assert len(result['positions']) == 1000

    def test_velocity_scaling(self, sample_npz_path):
        """Test that velocities are scaled by total_frames."""
        result = load_init_npz(
            sample_npz_path,
            max_samples=0,
            frame_start=0,
            frame_end=30,
        )

        # Velocities should be scaled by total_frames (30)
        # Original velocities were ~0.1, so scaled should be ~3.0
        vel_mags = torch.norm(result['velocities'], dim=1)
        assert vel_mags.max() > 1.0  # Should be scaled up

    def test_colors_normalized(self, sample_npz_path):
        """Test that colors are normalized to [0, 1]."""
        result = load_init_npz(
            sample_npz_path,
            max_samples=0,
            frame_start=0,
            frame_end=30,
        )

        assert result['colors'].max() <= 1.0
        assert result['colors'].min() >= 0.0

    def test_time_range(self, sample_npz_path):
        """Test that times are in [0, 1] range."""
        result = load_init_npz(
            sample_npz_path,
            max_samples=0,
            frame_start=0,
            frame_end=30,
        )

        assert result['times'].min() >= 0.0
        assert result['times'].max() <= 1.0

    def test_velocity_capping(self, tmp_path):
        """Test that large velocities are capped."""
        # Create NPZ with very large velocities
        n_points = 100
        np.random.seed(42)

        npz_path = tmp_path / "large_vel.npz"
        np.savez(
            npz_path,
            positions=np.random.randn(n_points, 3).astype(np.float32),
            velocities=np.random.randn(n_points, 3).astype(np.float32) * 100,  # Very large
            colors=np.random.rand(n_points, 3).astype(np.float32),
            times=np.linspace(0, 1, n_points).astype(np.float32),
            durations=np.ones(n_points, dtype=np.float32) * 0.1,
            frame_start=0,
            frame_end=30,
        )

        result = load_init_npz(
            str(npz_path),
            max_samples=0,
            frame_start=0,
            frame_end=30,
        )

        vel_mags = torch.norm(result['velocities'], dim=1)
        assert vel_mags.max() <= 10.0 + 1e-5  # Should be capped at ~10.0 (with fp tolerance)


# ============================================================
# Tests for load_init_npz_stratified
# ============================================================

class TestLoadInitNpzStratified:
    """Tests for load_init_npz_stratified function."""

    def test_stratified_sampling(self, sample_npz_with_keyframes):
        """Test stratified sampling across time windows."""
        result = load_init_npz_stratified(
            sample_npz_with_keyframes,
            max_samples=500,
            frame_start=0,
            frame_end=30,
        )

        assert 'positions' in result
        assert len(result['positions']) <= 500

        # Check that we have points from multiple time windows
        times = result['times'].squeeze().numpy()
        unique_times = np.unique(np.round(times, 2))
        assert len(unique_times) >= 3  # Should have multiple time windows

    def test_stratified_preserves_temporal_coverage(self, sample_npz_with_keyframes):
        """Test that stratified sampling preserves temporal coverage."""
        result = load_init_npz_stratified(
            sample_npz_with_keyframes,
            max_samples=200,
            frame_start=0,
            frame_end=30,
        )

        times = result['times'].squeeze().numpy()

        # Check that we have points at beginning, middle, and end
        assert times.min() < 0.1  # Has points near t=0
        assert times.max() > 0.9  # Has points near t=1


# ============================================================
# Tests for estimate_voxel_size
# ============================================================

class TestEstimateVoxelSize:
    """Tests for estimate_voxel_size function."""

    def test_basic_estimation(self):
        """Test basic voxel size estimation."""
        np.random.seed(42)
        positions = np.random.randn(1000, 3).astype(np.float32)

        voxel_size = estimate_voxel_size(positions)

        assert voxel_size > 0
        assert voxel_size <= 1.0  # Should be clamped

    def test_sparse_point_cloud(self):
        """Test with sparse point cloud."""
        # Very sparse points should result in larger voxel size
        positions = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
        ], dtype=np.float32)

        voxel_size = estimate_voxel_size(positions)

        # Should be relatively large due to sparse points
        assert voxel_size >= 0.01

    def test_dense_point_cloud(self):
        """Test with dense point cloud."""
        # Create a dense grid
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        z = np.linspace(0, 1, 20)
        xx, yy, zz = np.meshgrid(x, y, z)
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        voxel_size = estimate_voxel_size(positions)

        assert voxel_size > 0
        assert voxel_size <= 1.0

    def test_empty_point_cloud(self):
        """Test with empty point cloud."""
        positions = np.array([], dtype=np.float32).reshape(0, 3)

        voxel_size = estimate_voxel_size(positions)

        assert voxel_size == 0.1  # Default for empty


# ============================================================
# Tests for smart_sample_points
# ============================================================

class TestSmartSamplePoints:
    """Tests for smart_sample_points function."""

    def test_basic_sampling(self):
        """Test basic smart sampling."""
        np.random.seed(42)
        n_points = 1000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.rand(n_points, 3).astype(np.float32)

        indices = smart_sample_points(
            positions,
            velocities,
            colors,
            target_count=200,
        )

        assert len(indices) <= 200
        assert indices.max() < n_points
        assert indices.min() >= 0

    def test_no_sampling_needed(self):
        """Test when target exceeds point count."""
        np.random.seed(42)
        n_points = 100
        positions = np.random.randn(n_points, 3).astype(np.float32)
        velocities = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.rand(n_points, 3).astype(np.float32)

        indices = smart_sample_points(
            positions,
            velocities,
            colors,
            target_count=500,  # More than available
        )

        # Should return all indices when target > count
        assert len(indices) == n_points

    def test_velocity_boosting(self):
        """Test that high-velocity points get boosted."""
        np.random.seed(42)
        n_points = 1000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.rand(n_points, 3).astype(np.float32)

        # Create velocities where first 100 points have high velocity
        velocities = np.zeros((n_points, 3), dtype=np.float32)
        velocities[:100] = np.ones((100, 3)) * 10.0  # High velocity

        indices = smart_sample_points(
            positions,
            velocities,
            colors,
            target_count=200,
            velocity_weight=5.0,
        )

        # High velocity points should be well represented
        high_vel_sampled = np.sum(indices < 100)
        low_vel_sampled = np.sum(indices >= 100)

        # With velocity boosting, high-vel points should be over-represented
        assert high_vel_sampled > 0


# ============================================================
# Tests for create_splats_with_optimizers_4d
# ============================================================

class TestCreateSplatsWithOptimizers4D:
    """Tests for create_splats_with_optimizers_4d function."""

    @pytest.fixture
    def init_data(self):
        """Create sample init data."""
        n_points = 100
        torch.manual_seed(42)

        return {
            'positions': torch.randn(n_points, 3),
            'velocities': torch.randn(n_points, 3) * 0.1,
            'colors': torch.rand(n_points, 3),
            'times': torch.linspace(0, 1, n_points).unsqueeze(-1),
            'durations': torch.ones(n_points, 1) * 0.2,
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_splats_basic(self, init_data):
        """Test basic splat creation."""
        cfg = Config(init_duration=0.2)

        splats, optimizers = create_splats_with_optimizers_4d(
            cfg,
            init_data,
            scene_scale=1.0,
            device="cuda",
        )

        # Check all parameters exist
        assert 'means' in splats
        assert 'scales' in splats
        assert 'quats' in splats
        assert 'opacities' in splats
        assert 'sh0' in splats
        assert 'shN' in splats
        assert 'times' in splats
        assert 'durations' in splats
        assert 'velocities' in splats

        # Check shapes
        n_points = len(init_data['positions'])
        assert splats['means'].shape == (n_points, 3)
        assert splats['velocities'].shape == (n_points, 3)
        assert splats['times'].shape == (n_points, 1)
        assert splats['durations'].shape == (n_points, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_optimizers(self, init_data):
        """Test that optimizers are created for all params."""
        cfg = Config(init_duration=0.2)

        splats, optimizers = create_splats_with_optimizers_4d(
            cfg,
            init_data,
            scene_scale=1.0,
            device="cuda",
        )

        # Check optimizers for all params
        expected_params = ['means', 'scales', 'quats', 'opacities',
                          'sh0', 'shN', 'times', 'durations', 'velocities']

        for param_name in expected_params:
            assert param_name in optimizers, f"Missing optimizer for {param_name}"
            assert isinstance(optimizers[param_name], torch.optim.Adam)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_splats_on_device(self, init_data):
        """Test that splats are moved to correct device."""
        cfg = Config(init_duration=0.2)

        splats, _ = create_splats_with_optimizers_4d(
            cfg,
            init_data,
            scene_scale=1.0,
            device="cuda",
        )

        for name, param in splats.items():
            assert param.device.type == "cuda", f"{name} not on CUDA"


# ============================================================
# Tests for trajectory generation
# ============================================================

class TestTrajectoryGeneration:
    """Tests for trajectory generation functions."""

    def test_normalize(self):
        """Test vector normalization."""
        v = np.array([3.0, 4.0, 0.0])
        normalized = _normalize(v)

        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)
        np.testing.assert_almost_equal(normalized, [0.6, 0.8, 0.0])

    def test_viewmatrix(self):
        """Test view matrix construction."""
        position = np.array([0.0, 0.0, 5.0])
        lookdir = np.array([0.0, 0.0, -1.0])
        up = np.array([0.0, 1.0, 0.0])

        m = _viewmatrix(lookdir, up, position)

        assert m.shape == (3, 4)
        # Position should be in last column
        np.testing.assert_array_almost_equal(m[:, 3], position)

    def test_generate_interpolated_path(self):
        """Test path interpolation between camera poses."""
        # Create simple camera poses
        n_poses = 5
        poses = np.zeros((n_poses, 3, 4))
        for i in range(n_poses):
            angle = i * np.pi / (n_poses - 1)
            poses[i, :3, :3] = np.eye(3)
            poses[i, :, 3] = [np.sin(angle), 0, np.cos(angle)]

        n_interp = 10
        interp_poses = generate_interpolated_path(poses, n_interp)

        # Should have (n_interp * (n_poses - 1)) poses
        expected_count = n_interp * (n_poses - 1)
        assert interp_poses.shape == (expected_count, 3, 4)

    def test_interpolated_path_continuity(self):
        """Test that interpolated path is continuous."""
        # Create circular camera path
        n_poses = 4
        poses = np.zeros((n_poses, 3, 4))
        for i in range(n_poses):
            angle = i * 2 * np.pi / n_poses
            poses[i, :3, :3] = np.eye(3)
            poses[i, :, 3] = [np.cos(angle), np.sin(angle), 0]

        interp_poses = generate_interpolated_path(poses, n_interp=5)

        # Check positions change smoothly
        positions = interp_poses[:, :, 3]
        diffs = np.diff(positions, axis=0)
        diff_norms = np.linalg.norm(diffs, axis=1)

        # No sudden jumps
        assert diff_norms.max() < 1.0


# ============================================================
# Tests with real triangulate data
# ============================================================

class TestWithRealData:
    """Tests using real triangulated data from tests/resources/."""

    def test_load_real_triangulate_data(self, real_triangulate_npz):
        """Test loading NPZ created from real triangulated data."""
        result = load_init_npz(
            real_triangulate_npz,
            max_samples=500,
            frame_start=0,
            frame_end=30,
        )

        assert len(result['positions']) > 0
        assert result['positions'].shape[1] == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_splats_from_real_data(self, real_triangulate_npz):
        """Test creating splats from real triangulated data."""
        # Load data
        init_data = load_init_npz(
            real_triangulate_npz,
            max_samples=200,
            frame_start=0,
            frame_end=30,
        )

        cfg = Config(init_duration=0.2)

        splats, optimizers = create_splats_with_optimizers_4d(
            cfg,
            init_data,
            scene_scale=1.0,
            device="cuda",
        )

        assert len(splats['means']) == len(init_data['positions'])


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    """Integration tests for the training pipeline components."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_initialization_pipeline(self, sample_npz_path):
        """Test full initialization from NPZ to splats."""
        # 1. Load NPZ
        init_data = load_init_npz(
            sample_npz_path,
            max_samples=200,
            frame_start=0,
            frame_end=30,
        )

        # 2. Create splats
        cfg = Config(init_duration=0.2)
        splats, optimizers = create_splats_with_optimizers_4d(
            cfg,
            init_data,
            scene_scale=1.0,
            device="cuda",
        )

        # 3. Verify temporal computation
        t = 0.5
        times = splats['times']
        durations = torch.exp(splats['durations'])

        # Temporal opacity: exp(-0.5 * ((t - mu_t) / s)^2)
        time_diff = t - times
        temporal_opacity = torch.exp(-0.5 * (time_diff / durations) ** 2)

        assert temporal_opacity.shape == times.shape
        assert temporal_opacity.min() >= 0.0
        assert temporal_opacity.max() <= 1.0

        # 4. Verify position computation with velocity
        if cfg.use_velocity:
            velocities = splats['velocities']
            positions = splats['means']

            # Position at time t: x(t) = x + v * (t - t_canonical)
            positions_at_t = positions + velocities * time_diff

            assert positions_at_t.shape == positions.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
