#!/usr/bin/env python3
"""
Example script to render specific training viewpoints (22, 23, 24) from your cameras.json data
"""

from render import Renderer
import os

def main():
    # Initialize renderer
    print("Initializing renderer...")
    renderer = Renderer(
        model_path="/root/autodl-tmp/Octree-GS/Octree-GS/output/Ma0422",
        source_path="/root/autodl-tmp/Octree-GS/Octree-GS/data/Ma0422"
    )
    renderer.render_init()
    print("Initialization completed.")
    
    # Create output directory
    output_dir = "./outputs/example_views"
    os.makedirs(output_dir, exist_ok=True)
    
    # Render the specific camera IDs from your cameras.json data
    camera_ids = [22, 23, 24]
    
    for cam_id in camera_ids:
        try:
            output_path = os.path.join(output_dir, f"render_camera_{cam_id}.png")
            print(f"\n{'='*60}")
            print(f"Rendering camera {cam_id}...")
            print(f"{'='*60}")
            renderer.render_training_camera(cam_id, output_path)
        except Exception as e:
            print(f"Error rendering camera {cam_id}: {e}")
    
    print(f"\n{'='*60}")
    print("All renders completed!")
    print(f"Rendered images saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
