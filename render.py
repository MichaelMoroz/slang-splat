from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src import create_default_device
from src.renderer import Camera, GaussianRenderer
from src.scene import load_gaussian_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic Slangpy Gaussian splat renderer.")
    parser.add_argument("--ply", type=Path, required=True, help="Input 3D Gaussian PLY file.")
    parser.add_argument("--output", type=Path, default=Path("render.png"), help="Output PNG path.")
    parser.add_argument("--width", type=int, default=1280, help="Output image width.")
    parser.add_argument("--height", type=int, default=720, help="Output image height.")
    parser.add_argument("--max-splats", type=int, default=0, help="Limit rendered splat count. 0 means all.")
    parser.add_argument("--radius-scale", type=float, default=2.6, help="Projected splat radius multiplier.")
    parser.add_argument("--max-splat-radius-px", type=float, default=64.0, help="Clamp for projected splat radius.")
    parser.add_argument("--prepass-memory-mb", type=int, default=512, help="Cap prepass key/value/scanline memory.")
    parser.add_argument("--cam-pos", type=float, nargs=3, default=(0.0, 0.0, 3.0), help="Camera position.")
    parser.add_argument("--cam-target", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Look-at target.")
    parser.add_argument("--fov", type=float, default=60.0, help="Vertical camera FOV in degrees.")
    parser.add_argument("--near", type=float, default=0.1, help="Near clip.")
    parser.add_argument("--far", type=float, default=120.0, help="Far clip.")
    parser.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Background RGB color.")
    parser.add_argument("--no-flip-y", action="store_true", help="Disable vertical flip before writing PNG.")
    parser.add_argument("--debug-layers", action="store_true", help="Enable graphics debug layers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = load_gaussian_ply(args.ply)
    if args.max_splats > 0:
        scene = scene.subset(args.max_splats)

    device = create_default_device(enable_debug_layers=args.debug_layers)
    renderer = GaussianRenderer(
        device=device,
        width=args.width,
        height=args.height,
        radius_scale=args.radius_scale,
        max_splat_radius_px=args.max_splat_radius_px,
        max_prepass_memory_mb=args.prepass_memory_mb,
    )
    camera = Camera.look_at(
        position=np.array(args.cam_pos, dtype=np.float32),
        target=np.array(args.cam_target, dtype=np.float32),
        fov_y_degrees=float(args.fov),
        near=float(args.near),
        far=float(args.far),
    )
    output = renderer.render(scene, camera, background=np.asarray(args.bg, dtype=np.float32))
    print(output.stats)

    rgb = np.clip(output.image[:, :, :3], 0.0, 1.0)
    if not args.no_flip_y:
        rgb = np.flipud(rgb)
    image_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(args.output)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
