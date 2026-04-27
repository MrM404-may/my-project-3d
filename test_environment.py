#!/usr/bin/env python3
"""
Test script to verify the environment and code structure
"""

import os
import sys

print("Testing environment...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check if we can import the Renderer class
try:
    from render import Renderer
    print("✓ Successfully imported Renderer class")
except Exception as e:
    print(f"✗ Error importing Renderer: {e}")
    print("\nPlease activate the conda environment first:")
    print("  conda activate octree-gs")
    sys.exit(1)

print("\nCode structure test passed!")
print("\nTo run the render scripts:")
print("1. First activate the conda environment:")
print("   conda activate octree-gs")
print("\n2. Then run the desired script:")
print("   # Render training camera 22")
print("   python render_training.py \\")
print("       --model-path /root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422 \\")
print("       --source-path /root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422 \\")
print("       --camera-ids 22")
print("\n   # Render custom viewpoint")
print("   python render_custom.py \\")
print("       --model-path /root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422 \\")
print("       --source-path /root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422 \\")
print("       --position 0.0 0.0 0.0 \\")
print("       --rotation 1.0 0.0 0.0 0.0")
