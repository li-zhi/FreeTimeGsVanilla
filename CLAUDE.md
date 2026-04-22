# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreeTimeGSVanilla is a minimal implementation of **FreeTimeGS** (CVPR 2025) for 4D Gaussian Splatting of dynamic scenes. It uses [gsplat](https://github.com/nerfstudio-project/gsplat) as the rendering backend.

**Core concept**: Each Gaussian has temporal parameters:
- Position evolves as: `x(t) = x + v * (t - t_canonical)`
- Temporal opacity: `σ(t) = exp(-0.5 * ((t - t_canonical) / duration)²)`

## Environment Setup

```bash
# Create virtual environment with Python 3.12
uv venv .venv --python 3.12

# Activate the environment first
source .venv/bin/activate

# Install torch with CUDA 11.8 support first (required for build dependencies)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install numpy (required for building some packages)
uv pip install "numpy>=1.26,<2.0"

# Install all dependencies in editable mode
uv pip install -e . --no-build-isolation
```

**Important Notes:**
- gsplat is pinned to v1.5.3 in `pyproject.toml` for CUDA 11.8 compatibility
- The `--no-build-isolation` flag is required because gsplat and other CUDA packages need torch available during build
- If you encounter CUDA version mismatch errors, ensure torch is installed with the correct CUDA version first
- VSCode Python interpreter is configured in `.vscode/settings.json` to use `.venv/bin/python`

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
| `src/triangulate_multiview.py` | Multi-view triangulation for per-frame point clouds |
| `src/convert_calibration_to_colmap.py` | Convert OpenCV calibration to COLMAP sparse format |
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

### Generating Per-Frame Point Clouds

#### Using triangulate_multiview.py

This repository includes [src/triangulate_multiview.py](src/triangulate_multiview.py) for generating per-frame point clouds from multi-view video with known camera calibration.

```bash
python src/triangulate_multiview.py \
    --data-dir /path/to/dataset \
    --output-dir /path/to/output \
    --frame-start 0 --frame-end 30 \
    --device cuda \
    --confidence-threshold 0.3 \
    --max-matches 5000
```

**Input dataset structure:**
```
data_dir/
├── intri.yml          # Camera intrinsics (OpenCV YAML format)
├── extri.yml          # Camera extrinsics (Rot_XX, T_XX matrices)
└── images/
    ├── 00/            # Camera 00
    │   ├── 000000.jpg
    │   └── ...
    ├── 01/            # Camera 01
    └── ...
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | required | Path to dataset with intri.yml/extri.yml |
| `--output-dir` | required | Output directory for point clouds |
| `--frame-start` | 0 | Start frame index |
| `--frame-end` | None | End frame index (exclusive) |
| `--use-roma` | True | Use RoMa for matching (falls back to SIFT) |
| `--use-sift` | False | Force SIFT instead of RoMa |
| `--confidence-threshold` | 0.3 | Match confidence threshold |
| `--max-matches` | 5000 | Max matches per camera pair |
| `--reprojection-threshold` | 5.0 | Max reprojection error in pixels |

### Converting OpenCV Calibration to COLMAP Format

If you have a dataset with OpenCV-style calibration files (`intri.yml`, `extri.yml`), use [src/convert_calibration_to_colmap.py](src/convert_calibration_to_colmap.py) to generate COLMAP sparse reconstruction:

```bash
python src/convert_calibration_to_colmap.py \
    --input-dir /path/to/dataset \
    --output-dir /path/to/output
```

**Input dataset structure:**
```
input_dir/
├── intri.yml          # Camera intrinsics (K_XX, D_XX matrices)
├── extri.yml          # Camera extrinsics (Rot_XX, T_XX matrices)
└── images/
    ├── 00/            # Camera 00
    │   ├── 000000.jpg
    │   └── ...
    ├── 01/            # Camera 01
    └── ...
```

**Output structure (nested format for FreeTimeParser):**
```
output_dir/
├── images/
│   ├── cam00/                      # Nested camera folders
│   │   ├── 000000.jpg              # Symlinks to original images
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── cam01/
│   └── ...
└── sparse/0/
    ├── cameras.bin                 # OPENCV camera models (one per camera)
    ├── images.bin                  # Image poses (one entry per camera at reference frame)
    └── points3D.bin                # Empty (no triangulation)
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `--input-dir` | Directory with intri.yml, extri.yml, and images/ |
| `--output-dir` | Output directory for COLMAP format |
| `--single-frame` | Optional: process only this frame index |

**COLMAP Data** (for trainer - uses nested format):
```
data_dir/
├── images/
│   ├── cam00/
│   │   ├── 000000.jpg
│   │   └── ...
│   └── cam01/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

**Note:** The FreeTimeParser auto-detects nested vs flat image naming. Images in `images.bin` use the format `cam{XX}/{frame:06d}.jpg` to match the nested folder structure.

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
- When committing, ignore untracked files
- When committing, always save the full prompt messages since the last commit to the commit message
- When committing, run code review first
- When committing, run style check through `ruff check` (if needed, run style fix through `ruff check --fix`)
