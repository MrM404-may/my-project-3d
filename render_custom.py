#!/usr/bin/env python3
"""
Script to render custom viewpoints
"""

from render import Renderer
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Render custom viewpoints")
    parser.add_argument("--model-path", required=True, help="Path to the trained model directory")
    parser.add_argument("--source-path", required=True, help="Path to the source data directory")
    parser.add_argument("--position", type=float, nargs=3, default=[0.0, 0.0, 0.0], 
                        help="Camera position [x, y, z]")
    parser.add_argument("--rotation", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0], 
                        help="Camera rotation as quaternion [w, x, y, z]")
    parser.add_argument("--output", default="./outputs/custom/render_custom.png", 
                        help="Output file path")
    parser.add_argument("--demo", action="store_true", 
                        help="Run demo with multiple custom viewpoints")
    
    args = parser.parse_args()
    
    # Initialize renderer
    print("Initializing renderer...")
    renderer = Renderer(
        model_path=args.model_path,
        source_path=args.source_path
    )
    renderer.render_init()
    print("Initialization completed.")
    
    if args.demo:
        # Run demo with multiple viewpoints
        output_dir = "./outputs/custom_demo"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== Running Custom Viewpoints Demo ===")
        
        # Viewpoint 1: Identity orientation at origin
        print("\n1. Rendering identity orientation at origin...")
        renderer.render_new(
            position=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
            output_path=os.path.join(output_dir, "render_identity.png")
        )
        
        # Viewpoint 2: 90 degrees around Y-axis
        print("\n2. Rendering 90 degrees around Y-axis...")
        renderer.render_new(
            position=[2.0, 0.0, 0.0],
            rotation=[0.70710678, 0.0, 0.70710678, 0.0],  # 90° Y
            output_path=os.path.join(output_dir, "render_90y.png")
        )
        
        # Viewpoint 3: 90 degrees around X-axis
        print("\n3. Rendering 90 degrees around X-axis...")
        renderer.render_new(
            position=[0.0, 2.0, 0.0],
            rotation=[0.70710678, 0.70710678, 0.0, 0.0],  # 90° X
            output_path=os.path.join(output_dir, "render_90x.png")
        )
        
        # Viewpoint 4: 45 degrees around Z-axis
        print("\n4. Rendering 45 degrees around Z-axis...")
        renderer.render_new(
            position=[0.0, 0.0, 2.0],
            rotation=[0.92387953, 0.0, 0.0, 0.38268343],  # 45° Z
            output_path=os.path.join(output_dir, "render_45z.png")
        )
        
        print(f"\n=== Demo completed! ===")
        print(f"Rendered images saved to: {output_dir}")
        
    else:
        # Render single custom viewpoint
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        print(f"\nRendering custom viewpoint...")
        print(f"Position: {args.position}")
        print(f"Rotation (quaternion): {args.rotation}")
        
        renderer.render_new(
            position=args.position,
            rotation=args.rotation,
            output_path=args.output
        )

if __name__ == "__main__":
    main()
