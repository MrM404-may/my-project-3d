#!/usr/bin/env python3
"""
Test script for the new render.py file
Demonstrates how to use render_init and render_new functions
"""

from render import Renderer

# Initialize renderer
renderer = Renderer(
    model_path="/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422",
    source_path="/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422"
)

# Initialize the renderer (loads model, cameras, etc.)
print("Initializing renderer...")
renderer.render_init()
print("Initialization completed.")

# Example 1: Render from origin with identity rotation
print("\nRendering example 1: Origin with identity rotation")
position1 = [0.0, 0.0, 0.0]
rotation1 = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion [w, x, y, z]
output1 = "./outputs/render_origin.png"
renderer.render_new(position1, rotation1, output1)

# Example 2: Render from a different position
print("\nRendering example 2: Different position")
position2 = [1.0, 0.0, 0.0]  # 1 meter along x-axis
rotation2 = [0.7071, 0.0, 0.7071, 0.0]  # 90 degrees around y-axis
output2 = "./outputs/render_x1.png"
renderer.render_new(position2, rotation2, output2)

# Example 3: Render from another position
print("\nRendering example 3: Another position")
position3 = [0.0, 1.0, 0.0]  # 1 meter along y-axis
rotation3 = [0.7071, 0.7071, 0.0, 0.0]  # 90 degrees around x-axis
output3 = "./outputs/render_y1.png"
renderer.render_new(position3, rotation3, output3)

print("\nAll renders completed!")
