# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreeTimeGSVanilla is a minimal implementation of **FreeTimeGS** (CVPR 2025) for 4D Gaussian Splatting of dynamic scenes. It uses [gsplat](https://github.com/nerfstudio-project/gsplat) as the rendering backend.

**Core concept**: Each Gaussian has temporal parameters:
- Position evolves as: `x(t) = x + v * (t - t_canonical)`
- Temporal opacity: `σ(t) = exp(-0.5 * ((t - t_canonical) / duration)²)`

## Common Commands

### Full Pipeline (Keyframe Extraction + Training)
```bash
bash run_pipeline.sh <input_dir> <data_dir> <result_dir> <start_frame> <end_frame> <keyframe_step> <gpu_id> [config]

# Example:
bash run_pipeline.sh /path/to/triangulation /path/to/colmap /path/to/results 0 61 5 0 default_keyframe_small
```

### Step 1: Combine Keyframes with Velocity
```bash
python src/combine_frames_fast_keyframes.py \
    --input-dir /path/to/triangulation/output \
    --output-path /path/to/keyframes.npz \
    --frame-start 0 --frame-end 60 --keyframe-step 5
```

### Step 2: Train 4D Gaussians
```bash
CUDA_VISIBLE_DEVICES=0 python src/simple_trainer_freetime_4d_pure_relocation.py <config> \
    --data-dir /path/to/colmap/data \
    --init-npz-path /path/to/keyframes.npz \
    --result-dir /path/to/results \
    --start-frame 0 --end-frame 61 --max-steps 30000
```

Available configs: `default_keyframe` (~15M points), `default_keyframe_small` (~4M points)

### Interactive 4D Viewer
```bash
python src/viewer_4d.py --ckpt results/ckpts/ckpt_30000.pt --port 8080 --total-frames 60
```

### Export from Checkpoint (No Training)
```bash
python src/simple_trainer_freetime_4d_pure_relocation.py <config> \
    --ckpt-path /path/to/ckpt.pt --export-only
```

## Architecture

### Two-Stage Pipeline

1. **Point Cloud Preparation** ([src/combine_frames_fast_keyframes.py](src/combine_frames_fast_keyframes.py))
   - Loads per-frame triangulated 3D points (`points3d_frameXXXXXX.npy`, `colors_frameXXXXXX.npy`)
   - Extracts keyframes at specified intervals
   - Computes velocity via k-NN matching between consecutive frames
   - Outputs NPZ with positions, velocities, colors, times, durations

2. **4D Gaussian Training** ([src/simple_trainer_freetime_4d_pure_relocation.py](src/simple_trainer_freetime_4d_pure_relocation.py))
   - Initializes 4D Gaussians from NPZ
   - Trains with temporal parameters using gsplat's MCMC or DefaultStrategy
   - Outputs checkpoints, PLY sequences, and trajectory videos

### Key Components

| File | Purpose |
|------|---------|
| `src/simple_trainer_freetime_4d_pure_relocation.py` | Main trainer with 4D Gaussian optimization |
| `src/combine_frames_fast_keyframes.py` | Keyframe extraction and velocity estimation |
| `src/viewer_4d.py` | Interactive viser/nerfview-based viewer |
| `src/utils.py` | Helpers (KNN, colormap, PLY loading, camera modules) |
| `datasets/FreeTime_dataset.py` | COLMAP parser and PyTorch dataset |
| `datasets/traj.py` | Camera trajectory generation (ellipse, spiral, arc, dolly) |
| `datasets/normalize.py` | Scene normalization utilities |

### Training Phases (Annealing Strategy)

1. **Settling Phase** (steps 0 to `densification_start_step`):
   - All 4D parameters enabled from step 0
   - High velocity LR to capture fast motion
   - No densification (let initialization settle)

2. **Refinement Phase** (after `densification_start_step`):
   - Velocity LR annealing
   - Densification/relocation enabled
   - 4D regularization active

**Critical**: Never freeze velocities—this destroys initialization.

### Input Data Format

**Per-Frame Point Clouds** (for `combine_frames_fast_keyframes.py`):
```
input_dir/
├── points3d_frame000000.npy   # [M, 3] float32
├── colors_frame000000.npy     # [M, 3] float32 (0-255)
└── ...
```

### Generating Per-Frame Point Clouds (RoMa + COLMAP)

The original FreeTimeGS paper uses RoMa for dense feature matching followed by COLMAP triangulation:

**Step 1: Install RoMa**
```bash
pip install romatch
```

**Step 2: Feature Matching with RoMa**
```python
from romatch import roma_outdoor
import numpy as np

roma_model = roma_outdoor(device="cuda")

# Match features between multi-view images for each frame
warp, certainty = roma_model.match(imA_path, imB_path, device="cuda")
matches, certainty = roma_model.sample(warp, certainty)
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
```

**Step 3: Triangulate with COLMAP**
```bash
# Extract features (or import RoMa matches)
colmap feature_extractor --database_path db.db --image_path images/

# Match features across views
colmap sequential_matcher --database_path db.db

# Triangulate with known camera poses
colmap point_triangulator \
    --database_path db.db \
    --image_path images/ \
    --input_path sparse/known_poses/ \
    --output_path sparse/triangulated/
```

**Step 4: Export to NPY format**
```python
import pycolmap

# Load triangulated points
reconstruction = pycolmap.Reconstruction("sparse/triangulated/")
points3d = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float32)
colors = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.float32)

# Save per-frame (after filtering by visibility)
np.save(f"points3d_frame{frame_idx:06d}.npy", points3d)
np.save(f"colors_frame{frame_idx:06d}.npy", colors)
```

**Sample Dataset**: [Neural 3D Video Dataset](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) - 21 synchronized GoPro videos with camera poses (CC-BY-NC 4.0)

**COLMAP Data** (for trainer):
```
data_dir/
├── images/
│   └── cam01_frame000000.jpg
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### NPZ Format

| Field | Shape | Description |
|-------|-------|-------------|
| `positions` | [N, 3] | 3D coordinates |
| `velocities` | [N, 3] | Velocity vectors (m/frame) |
| `colors` | [N, 3] | RGB normalized to [0, 1] |
| `times` | [N, 1] | Normalized timestamps [0, 1] |
| `durations` | [N, 1] | Temporal duration |
| `has_velocity` | [N] | Valid velocity mask |

### Checkpoint Format (.pt)

```python
{
    "splats": {
        "means": tensor[N, 3],       # Canonical positions
        "scales": tensor[N, 3],      # Log-scale
        "quats": tensor[N, 4],       # Rotation (wxyz)
        "opacities": tensor[N],      # Logit opacities
        "sh0": tensor[N, 1, 3],      # DC spherical harmonics
        "shN": tensor[N, K, 3],      # Higher-order SH
        "times": tensor[N, 1],       # Canonical time
        "durations": tensor[N, 1],   # Log temporal duration
        "velocities": tensor[N, 3],  # Linear velocity
    },
    "step": int,
}
```

## Key Dependencies

- `gsplat` - CUDA Gaussian splatting backend
- `torch` - PyTorch 2.0+
- `pycolmap` - COLMAP data loading
- `viser`, `nerfview` - Interactive viewer
- `tyro` - CLI configuration
- `fused_ssim` - Fast SSIM computation

## Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 60000 | Training iterations |
| `--init-duration` | 0.1 | Initial temporal duration |
| `--velocity-lr-start` | 5e-3 | Starting velocity LR |
| `--velocity-lr-end` | 1e-4 | Ending velocity LR |
| `--lambda-4d-reg` | 1e-3 | 4D regularization weight |
| `--keyframe-step` | 5 | Frames between keyframes |

## Important Notes
- In every response to me, always start with calling me DUDE
- When commiting, ignore untracked files
- When commiting, always save the full prompt messages since the last commit to the commit message
