#!/usr/bin/env python3
"""
Script to render training viewpoints from cameras.json
"""

from render import Renderer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Render training cameras from cameras.json")
    parser.add_argument("--model-path", required=True, help="Path to the trained model directory")
    parser.add_argument("--source-path", required=True, help="Path to the source data directory")
    parser.add_argument("--camera-ids", type=int, nargs='+', help="Specific camera IDs to render")
    parser.add_argument("--all", action="store_true", help="Render all training cameras")
    parser.add_argument("--output-dir", default="./outputs/training", help="Output directory for rendered images")
    
    args = parser.parse_args()
    
    print(f"Model path: {args.model_path}")
    print(f"Source path: {args.source_path}")
    print(f"Camera IDs: {args.camera_ids}")
    print(f"Output dir: {args.output_dir}")
    
    # Initialize renderer
    print("Initializing renderer...")
    renderer = Renderer(
        model_path=args.model_path,
        source_path=args.source_path
    )
    print("Created Renderer instance")
    print("Calling render_init...")
    renderer.render_init()
    print("render_init completed")
    print("Initialization completed.")
    print(f"Number of cameras loaded: {len(renderer.camera_uids)}")
    print(f"Camera dict keys: {list(renderer.camera_dict.keys())[:5]}...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which cameras to render
    camera_ids_to_render = []
    
    if args.all:
        camera_ids_to_render = renderer.camera_uids
        print(f"Rendering all {len(camera_ids_to_render)} training cameras...")
    elif args.camera_ids:
        camera_ids_to_render = args.camera_ids
        print(f"Rendering cameras: {camera_ids_to_render}")
    else:
        # Default: render first 5 cameras
        camera_ids_to_render = renderer.camera_uids[:5] if renderer.camera_uids else []
        print(f"No cameras specified, rendering first {len(camera_ids_to_render)} cameras...")
    
    # Render each camera
    for cam_id in camera_ids_to_render:
        try:
            output_path = os.path.join(args.output_dir, f"render_camera_{cam_id}.png")
            print(f"\nRendering camera {cam_id}...")
            renderer.render_training_camera(cam_id, output_path)
        except Exception as e:
            print(f"Error rendering camera {cam_id}: {e}")
    
    print("\nRendering completed!")
    print(f"Rendered images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
