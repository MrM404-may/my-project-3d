# Render.py Usage Guide

This guide explains how to use the new `render.py` file and its accompanying scripts.

## Overview

The `render.py` file contains a `Renderer` class with three main functionalities:

1. **render_init()**: Initializes the renderer, loads the model, cameras.json, and cache.json
2. **render_new(position, rotation, output_path)**: Renders a **custom viewpoint** given position (xyz) and rotation (quaternion)
3. **render_training_camera(camera_id, output_path)**: Renders a specific training camera by ID from cameras.json

## Files Included

- [`render.py`](file:///workspace/render.py) - Main renderer class
- [`render_custom.py`](file:///workspace/render_custom.py) - Command-line script for custom viewpoints
- [`render_training.py`](file:///workspace/render_training.py) - Command-line script to render training cameras
- [`render_example_views.py`](file:///workspace/render_example_views.py) - Example script to render cameras 22, 23, 24
- [`test_render.py`](file:///workspace/test_render.py) - Test script demonstrating all functionality

---

## Quick Start: Custom Viewpoints

### Render a Single Custom Viewpoint

```bash
python render_custom.py \
    --model-path /root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422 \
    --source-path /root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422 \
    --position 0.0 0.0 0.0 \
    --rotation 1.0 0.0 0.0 0.0
```

Parameters:
- `--position`: Camera position in 3D space [x, y, z]
- `--rotation`: Camera orientation as quaternion [w, x, y, z]
- `--output`: Path to save the rendered image

### Run Custom Viewpoint Demo

```bash
python render_custom.py \
    --model-path /root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422 \
    --source-path /root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422 \
    --demo
```

This renders 4 different custom viewpoints and saves them to `./outputs/custom_demo/`.

---

## Quick Start: Training Viewpoints

### Option 1: Using the Example Script

The easiest way is to use the pre-configured example script:

```bash
python render_example_views.py
```

This will render cameras 22, 23, 24 from your cameras.json data and save them to `./outputs/example_views/`.

### Option 2: Using the Command-Line Script

To render specific training cameras:

```bash
python render_training.py \
    --model-path /root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422 \
    --source-path /root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422 \
    --camera-ids 22 23 24
```

To render all training cameras:

```bash
python render_training.py \
    --model-path /root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422 \
    --source-path /root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422 \
    --all
```

---

## Using the Renderer Class Directly in Python

You can also use the Renderer class in your own Python code:

```python
from render import Renderer

# Initialize renderer
renderer = Renderer(
    model_path="/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422",
    source_path="/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422"
)

# Load the model and initialize
renderer.render_init()

# --- Option 1: Render a custom viewpoint ---
position = [0.0, 0.0, 0.0]  # xyz
rotation = [1.0, 0.0, 0.0, 0.0]  # quaternion [w, x, y, z]
renderer.render_new(position, rotation, "./custom_view.png")

# --- Option 2: Render a training camera ---
renderer.render_training_camera(22, "./training_camera_22.png")
```

---

## Quaternion Reference

Common quaternions for rotations:

- Identity (no rotation): `[1.0, 0.0, 0.0, 0.0]`
- 90° around X-axis: `[0.7071, 0.7071, 0.0, 0.0]`
- 90° around Y-axis: `[0.7071, 0.0, 0.7071, 0.0]`
- 90° around Z-axis: `[0.7071, 0.0, 0.0, 0.7071]`
- 45° around Y-axis: `[0.9239, 0.0, 0.3827, 0.0]`

---

## cameras.json Format

The code expects cameras.json in the following format:

```json
[
  {
    "id": 22,
    "img_name": "000056",
    "width": 1280,
    "height": 720,
    "position": [-14.703123712639574, 0.0800230657697498, 1.7612999939136067],
    "rotation": [[0.9996860397409149, -0.019123955160086126, -0.01618938807248506],
                [0.018832329601556085, 0.9996610072483189, -0.01797815199222925],
                [0.016527713359834762, 0.017667623674743515, 0.9997073020463455]],
    "fy": 861.21484941105,
    "fx": 867.3526294468791
  }
]
```

---

## Notes

- Custom viewpoints automatically find the most similar training camera
- The renderer uses the region and APE code from the most similar training camera
- The code supports both quaternion (for custom views) and 3x3 rotation matrix (for training views) formats
- All rendered images are clamped to [0, 1] before saving
