#!/usr/bin/env python3
"""
Test script for the new render.py file
Demonstrates how to use render_init, render_new, and render_training_camera functions
"""

from render import Renderer
import os

# Initialize renderer
renderer = Renderer(
    model_path="/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422",
    source_path="/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422"
)

# Initialize the renderer (loads model, cameras, etc.)
print("Initializing renderer...")
renderer.render_init()
print("Initialization completed.")

# Create output directories
os.makedirs("./outputs/custom", exist_ok=True)
os.makedirs("./outputs/training", exist_ok=True)

# Example 1: Render from custom viewpoint
print("\nRendering example 1: Custom viewpoint")
position1 = [0.0, 0.0, 0.0]
rotation1 = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion [w, x, y, z]
output1 = "./outputs/custom/render_origin.png"
renderer.render_new(position1, rotation1, output1)

# Example 2: Render from another custom viewpoint
print("\nRendering example 2: Another custom viewpoint")
position2 = [1.0, 0.0, 0.0]  # 1 meter along x-axis
rotation2 = [0.7071, 0.0, 0.7071, 0.0]  # 90 degrees around y-axis
output2 = "./outputs/custom/render_x1.png"
renderer.render_new(position2, rotation2, output2)

# Example 3: Render from training cameras
print("\nRendering example 3: Training cameras")
training_camera_ids = [22, 23, 24]  # From your shared cameras.json data
for cam_id in training_camera_ids:
    output_path = f"./outputs/training/render_camera_{cam_id}.png"
    print(f"\nRendering training camera {cam_id}...")
    renderer.render_training_camera(cam_id, output_path)

print("\nAll renders completed!")
