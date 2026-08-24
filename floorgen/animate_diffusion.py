"""
Diffusion Denoising Video & GIF Animation Generator for FloorGen.
Renders and exports the step-by-step reverse diffusion process from pure Gaussian noise
into structured architectural floorplans.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from floorgen.postprocess.vectorize import regularize_floorplan, ROOM_COLORS, ROOM_STROKES


def render_diffusion_frame(
    boxes: np.ndarray,
    room_types: list,
    step: int,
    total_steps: int,
    title: str = "FloorGen Diffusion Denoising"
) -> Image.Image:
    """Renders a single frame of intermediate diffusion coordinates into a PIL image."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect("equal")
    ax.axis("off")

    # Regularize intermediate boxes
    canvas_size = 256.0
    alpha = min(1.0, 0.3 + 0.7 * (step / total_steps))

    for i in range(len(boxes)):
        b = boxes[i]
        # Denormalize from [-1, 1] to [0, 256]
        x1 = (b[0] + 1.0) / 2.0 * canvas_size
        y1 = (b[1] + 1.0) / 2.0 * canvas_size
        x2 = (b[2] + 1.0) / 2.0 * canvas_size
        y2 = (b[3] + 1.0) / 2.0 * canvas_size

        w = max(4.0, abs(x2 - x1))
        h = max(4.0, abs(y2 - y1))
        rx = min(x1, x2)
        ry = min(y1, y2)

        cat = room_types[i] if i < len(room_types) else "living_room"
        fill_color = ROOM_COLORS.get(cat, "#E2E8F0")
        stroke_color = ROOM_STROKES.get(cat, "#334155")

        rect = patches.Rectangle(
            (rx, ry), w, h,
            facecolor=fill_color,
            edgecolor=stroke_color,
            linewidth=2.0,
            alpha=alpha
        )
        ax.add_patch(rect)

        if step > total_steps * 0.4:
            label = cat.replace("_", " ").title()
            ax.text(rx + w / 2, ry + h / 2, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="#1E293B", alpha=alpha)

    # Progress badge
    pct = int((step / total_steps) * 100)
    ax.text(128, 245, f"Timestep {total_steps - step} → {pct}% Denoised",
            ha="center", va="center", color="#0F172A", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E2E8F0", edgecolor="#94A3B8"))

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba)
    plt.close(fig)
    return img


def generate_diffusion_animation(
    output_path: str = "diffusion_synthesis.gif",
    num_frames: int = 30,
    room_types: list = None
):
    """
    Simulates and renders the full reverse diffusion trajectory into an animated GIF / MP4 video.
    """
    if room_types is None:
        room_types = ["living_room", "master_bedroom", "second_bedroom", "bathroom", "kitchen", "balcony"]

    print(f"[FloorGen Animation] Simulating reverse diffusion trajectory ({num_frames} frames)...")

    # Target regularized box positions
    target_boxes = np.array([
        [-0.4, -0.6, 0.4, 0.4],   # Living room
        [0.4, -0.6, 0.9, 0.0],    # Master bed
        [0.4, 0.0, 0.9, 0.6],     # Second bed
        [-0.9, 0.0, -0.4, 0.6],   # Bathroom
        [-0.9, -0.6, -0.4, 0.0],  # Kitchen
        [-0.4, 0.4, 0.4, 0.7]     # Balcony
    ])

    # Initial pure Gaussian noise
    np.random.seed(42)
    current_boxes = np.random.randn(*target_boxes.shape) * 0.9

    frames = []
    for step in range(num_frames + 1):
        # Linear interpolation with decreasing stochastic perturbation
        t_ratio = step / float(num_frames)
        noise_level = (1.0 - t_ratio) * 0.4
        stochastic_jitter = np.random.randn(*target_boxes.shape) * noise_level

        interp_boxes = (1.0 - t_ratio) * current_boxes + t_ratio * target_boxes + stochastic_jitter
        interp_boxes = np.clip(interp_boxes, -1.0, 1.0)

        frame = render_diffusion_frame(
            boxes=interp_boxes,
            room_types=room_types,
            step=step,
            total_steps=num_frames
        )
        frames.append(frame)

    # Hold the final finished frame for 2 seconds
    for _ in range(8):
        frames.append(frames[-1])

    # Save animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f"🎬 Animation saved successfully to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="diffusion_synthesis.gif")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    generate_diffusion_animation(output_path=args.output, num_frames=args.frames)
