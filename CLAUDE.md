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

## Pipeline Flow

```
Multi-view dataset (intri.yml, extri.yml, images/)
          │
          ├── Preprocessing A  →  per-frame point clouds (.npy)
          │   triangulate_multiview.py
          │
          └── Preprocessing B  →  COLMAP (SfM) sparse reconstruction
              convert_calibration_to_colmap.py

per-frame .npy  ──►  Step 1  ──►  keyframes.npz
                     combine_frames_fast_keyframes.py

COLMAP + keyframes.npz  ──►  Step 2  ──►  checkpoints, PLY sequences, trajectory videos
                             simple_trainer_freetime_4d_pure_relocation.py
```

`run_pipeline.sh` fuses Step 1 and Step 2 into a single invocation.

## Pipeline Steps

### Preprocessing A: Generate Per-Frame Point Clouds

Triangulate per-frame 3D points from multi-view video with known camera calibration. Consumes a [Multi-View Dataset Layout](#multi-view-dataset-layout), produces a [Per-Frame Point Cloud directory](#per-frame-point-cloud-files).

```bash
python src/triangulate_multiview.py \
    --data-dir /path/to/dataset \
    --output-dir /path/to/triangulation/output \
    --frame-start 0 --frame-end 30 \
    --device cuda \
    --confidence-threshold 0.3 \
    --max-matches 5000
```

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

### Preprocessing B: Convert OpenCV Calibration to COLMAP (SfM) Format

Convert OpenCV-style calibration (`intri.yml`, `extri.yml`) into a COLMAP sparse reconstruction consumable by the trainer's `--data-dir`. Consumes a [Multi-View Dataset Layout](#multi-view-dataset-layout), produces a [COLMAP (SfM) Nested Layout](#colmap-sfm-nested-layout).

```bash
python src/convert_calibration_to_colmap.py \
    --input-dir /path/to/dataset \
    --output-dir /path/to/colmap/data
```

| Parameter | Description |
|-----------|-------------|
| `--input-dir` | Directory with intri.yml, extri.yml, and images/ |
| `--output-dir` | Output directory for COLMAP (SfM) format |
| `--single-frame` | Optional: process only this frame index |

### Step 1: Combine Keyframes with Velocity

Load per-frame triangulated points, extract keyframes at a regular interval, compute per-point velocity via k-NN matching between consecutive frames, and emit a single NPZ (see [NPZ Format](#npz-format)).

```bash
python src/combine_frames_fast_keyframes.py \
    --input-dir /path/to/triangulation/output \
    --output-path /path/to/keyframes.npz \
    --frame-start 0 --frame-end 60 --keyframe-step 5
```

### Step 2: Train 4D Gaussians

Initialize 4D Gaussians from the keyframe NPZ, train with gsplat's MCMC/DefaultStrategy, and produce checkpoints, PLY sequences, and trajectory videos. Training follows a two-phase annealing schedule — see [Training Phases](#training-phases-annealing-strategy) and [Key Training Parameters](#key-training-parameters).

```bash
CUDA_VISIBLE_DEVICES=0 python src/simple_trainer_freetime_4d_pure_relocation.py <config> \
    --data-dir /path/to/colmap/data \
    --init-npz-path /path/to/keyframes.npz \
    --result-dir /path/to/results \
    --start-frame 0 --end-frame 61 --max-steps 30000
```

Available configs: `default_keyframe` (~15M points), `default_keyframe_small` (~4M points).

### Evaluation During Training

At each step in `--eval-steps` (default `[15000, 30000, 45000, 60000]`), the trainer runs a sampled validation pass — not a full val render — so training stays fast.

**Validation split.** Every `--test-every`-th camera (default `8`) is held out from training and used only for validation. With 18 cameras, cameras `0, 8, 16` become the val set; the rest train. The split is computed once in `FreeTime4DRunner.__init__` from the COLMAP camera list.

**Timing.** Eval triggers on the step immediately *before* each value in `--eval-steps` (at step 14999 if 15000 is listed), then training resumes.

**Per-eval procedure** (`FreeTime4DRunner.eval` in `src/simple_trainer_freetime_4d_pure_relocation.py`):
1. Iterate the val `DataLoader` (batch size 1, no shuffle). Each item is one `(camera, frame)` pair.
2. Subsample: skip all but every `--eval-sample-every`-th frame (default `60`). E.g., 300 frames × 3 val cameras = 900 items → ~15 actually rendered.
3. For each sampled item, rasterize Gaussians at that camera pose and time `t` (using the currently-active SH degree), clamp colors to `[0, 1]`.
4. Save a side-by-side `groundtruth | rendered` PNG to `result_dir/renders/val_step{step}_{i:04d}.png`.
5. Compute PSNR, SSIM, and LPIPS (network = `--lpips-net`, default `alex`); record per-image GPU wall-clock time.
6. Aggregate mean PSNR / SSIM / LPIPS, average render time per image (`ellipse_time`), and current Gaussian count (`num_GS`).
7. Write `result_dir/stats/val_step{step:04d}.json`, push all scalars to TensorBoard under the `val/` prefix, and print a one-line summary.

`eval()` is wrapped in `@torch.no_grad()`.

**Trajectory videos and PLY exports are separate.** Full trajectory videos (`render_traj`) — and PLY sequence exports (`export_ply_sequence`) when `--export-ply` is set — happen at `--save-steps`, not `--eval-steps`, and are not subsampled.

### Combined Step 1&2 (Keyframe Extraction + Training)

Runs Step 1 and Step 2 back-to-back via a helper script:

```bash
bash run_pipeline.sh <input_dir> <data_dir> <result_dir> <start_frame> <end_frame> <keyframe_step> <gpu_id> [config]

# Example:
bash run_pipeline.sh /path/to/triangulation /path/to/colmap /path/to/results 0 61 5 0 default_keyframe_small
```

## Other Tools

### Interactive 4D Viewer

```bash
python src/viewer_4d.py --ckpt results/ckpts/ckpt_30000.pt --port 8080 --total-frames 60
```

### Export from Checkpoint (No Training)

```bash
python src/simple_trainer_freetime_4d_pure_relocation.py <config> \
    --ckpt-path /path/to/ckpt.pt --export-only
```

## Training Phases (Annealing Strategy)

1. **Settling Phase** (steps 0 to `densification_start_step`):
   - All 4D parameters enabled from step 0
   - High velocity LR to capture fast motion
   - No densification (let initialization settle)

2. **Refinement Phase** (after `densification_start_step`):
   - Velocity LR annealing
   - Densification/relocation enabled
   - 4D regularization active

**Critical**: Never freeze velocities—this destroys initialization.

## Data Formats

### Multi-View Dataset Layout

Input to Preprocessing A and B. OpenCV-style calibration with per-camera image folders:

```
data_dir/
├── intri.yml          # Camera intrinsics (K_XX, D_XX — OpenCV YAML format)
├── extri.yml          # Camera extrinsics (Rot_XX, T_XX matrices)
└── images/
    ├── 00/            # Camera 00
    │   ├── 000000.jpg
    │   └── ...
    ├── 01/            # Camera 01
    └── ...
```

### Per-Frame Point Cloud Files

Output of Preprocessing A, input to Step 1:

```
triangulation_output/
├── points3d_frame000000.npy   # [M, 3] float32
├── colors_frame000000.npy     # [M, 3] float32 (0-255)
└── ...
```

### COLMAP (SfM) Nested Layout

Output of Preprocessing B, consumed by Step 2's `--data-dir`:

```
colmap_data/
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

**Note:** The FreeTimeParser auto-detects nested vs flat image naming. Images in `images.bin` use the format `cam{XX}/{frame:06d}.jpg` to match the nested folder structure.

### NPZ Format

Output of Step 1, consumed by Step 2's `--init-npz-path`:

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

## File Map

| File | Purpose |
|------|---------|
| `src/simple_trainer_freetime_4d_pure_relocation.py` | Main trainer with 4D Gaussian optimization |
| `src/combine_frames_fast_keyframes.py` | Keyframe extraction and velocity estimation |
| `src/triangulate_multiview.py` | Multi-view triangulation for per-frame point clouds |
| `src/convert_calibration_to_colmap.py` | Convert OpenCV calibration to COLMAP (SfM) sparse format |
| `src/viewer_4d.py` | Interactive viser/nerfview-based viewer |
| `src/utils.py` | Helpers (KNN, colormap, PLY loading, camera modules) |
| `datasets/FreeTime_dataset.py` | COLMAP (SfM) parser and PyTorch dataset |
| `datasets/traj.py` | Camera trajectory generation (ellipse, spiral, arc, dolly) |
| `datasets/normalize.py` | Scene normalization utilities |

## Key Dependencies

- `gsplat` - CUDA Gaussian splatting backend
- `torch` - PyTorch 2.0+
- `pycolmap` - COLMAP (SfM) data loading
- `viser`, `nerfview` - Interactive viewer
- `tyro` - CLI configuration
- `fused_ssim` - Fast SSIM computation

## Key Training Parameters

Defaults below are the bare `Config` dataclass defaults from `src/simple_trainer_freetime_4d_pure_relocation.py`. Preset configs (`default_keyframe`, `default_keyframe_small`) override several of these — the preset values take effect when you pass the config name as the first positional argument.

### Core Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `data/4d_scene` | Dataset directory (images + COLMAP sparse) |
| `--result-dir` | `results/freetime_4d` | Output directory for ckpts, videos, PLYs, TB logs |
| `--init-npz-path` | `None` | NPZ from Step 1 with positions, velocities, colors, times |
| `--max-steps` | `70000` | Total training iterations |
| `--batch-size` | `1` | Images per iteration |
| `--steps-scaler` | `1.0` | Scale all step-counted parameters (quick experiments) |
| `--data-factor` | `1` | Image downsample factor (1 = full resolution) |
| `--test-every` | `8` | Use every Nth camera for validation |
| `--eval-steps` | `[15000, 30000, 45000, 60000]` | Validation steps |
| `--save-steps` | `[15000, 30000, 45000, 60000]` | Checkpoint-save steps |

### Frame Range
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start-frame` | `0` | Start frame index (inclusive, t=0 maps here) |
| `--end-frame` | `300` | End frame index (exclusive, t=1 maps here) |
| `--frame-step` | `1` | Step between frames when loading |

### Initialization & Sampling
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-samples` | `2000000` | Max Gaussians to initialize from NPZ |
| `--no-sampling` | `False` | Use ALL NPZ points (overrides max-samples) |
| `--use-smart-sampling` | `True` | Density-, velocity-, center-weighted downsampling |
| `--smart-sampling-voxel-size` | `-1.0` | `-1` = auto-estimate from scene scale |
| `--smart-sampling-velocity-weight` | `5.0` | Moving points boosted up to `(1+w)×` |
| `--smart-sampling-center-weight` | `2.0` | Points near scene center boosted |
| `--use-stratified-sampling` | `False` | Sample equally across all frames |
| `--use-keyframe-sampling` | `False` | Sample densely from keyframes only |
| `--keyframe-step` | `-1` | `-1` = auto-read from NPZ metadata |
| `--sample-high-velocity-ratio` | `0.0` | High values (0.8) tend to select noisy outliers |

### Temporal (Duration & Velocity)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--init-duration` | `-1.0` | `-1` = auto from NPZ keyframe_step |
| `--auto-init-duration` | `True` | Compute duration from NPZ (`(kf_step/total) × multiplier`) |
| `--init-duration-multiplier` | `2.0` | Coverage factor over keyframe gap |
| `--use-velocity` | `True` | Enable velocity-based motion |
| `--velocity-lr-start` | `1e-2` | Starting velocity LR (preset configs use `5e-3`) |
| `--velocity-lr-end` | `1e-4` | Ending velocity LR (annealed) |

### Loss Weights & Regularization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda-img` | `0.8` | L1 image reconstruction weight |
| `--lambda-ssim` | `0.2` | SSIM structural similarity weight |
| `--lambda-perc` | `0.01` | LPIPS perceptual loss weight |
| `--lambda-4d-reg` | `1e-3` | 4D regularization weight (paper: `1e-2`) |
| `--lambda-duration-reg` | `1e-3` | Duration regularization weight |

### Training Phases
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--densification-start-step` | `1000` | Start densification/relocation/pruning after this step |
| `--reg-4d-start-step` | `0` | Start 4D regularization from step 0 |

### Densification: Relocation (MCMC-style)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-relocation` | `True` | Relocate low-opacity Gaussians to high-gradient regions |
| `--relocation-every` | `100` | Relocate every N iters (after densification-start-step) |
| `--relocation-stop-iter` | `50000` | Stop relocation after this iter |
| `--relocation-opacity-threshold` | `0.005` | Gaussians below this are candidates |
| `--relocation-max-ratio` | `0.015` | Max fraction relocated per step |
| `--relocation-lambda-grad` | `0.5` | Gradient weight in relocation score |
| `--relocation-lambda-opa` | `0.5` | Opacity weight in relocation score |

### Densification: Pruning (off by default)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-pruning` | `False` | Custom pruning (MCMCStrategy already handles this) |
| `--prune-every` | `500` | Prune every N iters |
| `--prune-stop-iter` | `20000` | Stop pruning after this iter |
| `--prune-opacity-threshold` | `0.005` | Opacity floor |
| `--use-budget-pruning` | `False` | Manual budget pruning (pure-relocation mode) |
| `--budget-prune-every` | `500` | Budget-prune every N iters |
| `--budget-prune-threshold` | `0.001` | Aggressive opacity floor for budget pruning |

### Learning Rates
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--position-lr` | `1.6e-4` | Gaussian positions |
| `--scales-lr` | `5e-3` | Gaussian scales |
| `--quats-lr` | `1e-3` | Gaussian rotations |
| `--opacities-lr` | `5e-2` | Gaussian opacities |
| `--sh0-lr` | `2.5e-3` | DC SH (base color) |
| `--shN-lr` | `2.5e-3 / 20` | Higher-order SH |
| `--times-lr` | `1e-3` | Canonical times |
| `--durations-lr` | `5e-3` | Temporal durations |

### Rendering & Trajectory
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--render-traj-path` | `arc` | `arc` / `interp` / `dolly` / `fixed` / `ellipse_z` / `ellipse_y` |
| `--render-traj-arc-degrees` | `30.0` | Total arc sweep for `arc` path |
| `--render-traj-dolly-amount` | `0.2` | Dolly distance as fraction of scene scale |
| `--render-traj-n-frames` | `120` | Video frame count |
| `--render-traj-fps` | `30` | Video FPS |
| `--render-traj-time-frames` | `None` | `None` = auto from frame range, `0` = same as n-frames |
| `--render-traj-camera-loops` | `1` | Camera passes through trajectory while time progresses |
| `--near-plane` | `0.01` | Near clipping plane |
| `--far-plane` | `1e10` | Far clipping plane |
| `--packed` | `False` | Packed rasterization (needed for 8M+ Gaussians) |
| `--antialiased` | `False` | Antialiased rasterization |

### PLY Export
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--export-ply` | `False` | Export PLY sequence at end of training |
| `--export-ply-steps` | `None` | Steps at which to export (`None` = use save-steps) |
| `--export-ply-format` | `ply` | `ply` / `splat` / `ply_compressed` |
| `--export-ply-opacity-threshold` | `0.01` | Skip Gaussians below this combined opacity |
| `--export-ply-compact` | `True` | Factored per-frame export (~15× smaller) |
| `--export-ply-frame-step` | `1` | Export every Nth frame |

### Model & Misc
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sh-degree` | `3` | Max SH degree |
| `--sh-degree-interval` | `1000` | Steps between SH degree increments |
| `--init-opacity` | `0.5` | Initial opacity (pre-sigmoid) |
| `--init-scale` | `1.0` | Scale factor for initial Gaussian sizes |
| `--global-scale` | `1.0` | Scene global scale factor |
| `--random-bkgd` | `False` | Randomize background (interferes with 4D temporal opacity) |
| `--disable-viewer` | `True` | Disable Viser viewer during training |
| `--lpips-net` | `alex` | `alex` (fast) or `vgg` (slightly more accurate) |
| `--tb-every` | `50` | Log scalars to TensorBoard every N steps |
| `--tb-image-every` | `200` | Log images to TensorBoard every N steps |

### Checkpoint / Resume / Export-Only
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ckpt-path` | `None` | Path to `.pt` checkpoint to resume or export from |
| `--export-only` | `False` | With `--ckpt-path`, skip training and just export |

## Important Notes
- In every response to me, always start with calling me DUDE
- When committing, ignore untracked files
- When committing, always save the full prompt messages since the last commit to the commit message
- When committing, run code review first
- When committing, run style check through `ruff check` (if needed, run style fix through `ruff check --fix`)
