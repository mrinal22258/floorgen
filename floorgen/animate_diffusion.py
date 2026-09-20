"""
Diffusion Denoising Video & GIF Animation Generator for FloorGen.
Renders and exports the step-by-step reverse diffusion process from pure Gaussian noise
into structured architectural floorplans using real model inference trajectories.
"""

import os
import argparse
from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from floorgen.postprocess.vectorize import regularize_floorplan, ROOM_COLORS, ROOM_STROKES
from floorgen.data.dataset import ROOM_TYPE_TO_ID
from floorgen.models.diffusion_core.house_diffusion import DDPMScheduler
from floorgen.models.diffusion_core.rag_diffusion import RAGFloorplanDiffusion


def render_diffusion_frame(
    boxes: np.ndarray,
    room_types: list,
    step: int,
    total_steps: int,
    title_badge: str = "FloorGen Diffusion Denoising",
    is_preview: bool = False
) -> Image.Image:
    """Renders a single frame of intermediate diffusion coordinates into a PIL image."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect("equal")
    ax.axis("off")

    canvas_size = 256.0
    alpha = min(1.0, 0.3 + 0.7 * (step / max(1, total_steps)))

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

        if step > total_steps * 0.35:
            label = cat.replace("_", " ").title()
            ax.text(rx + w / 2, ry + h / 2, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="#1E293B", alpha=alpha)

    # Progress badge
    pct = int((step / max(1, total_steps)) * 100)
    badge_label = f"[Illustrative Preview] {pct}%" if is_preview else f"Timestep {max(0, total_steps - step)} → {pct}% Denoised"
    ax.text(128, 245, badge_label,
            ha="center", va="center", color="#0F172A", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E2E8F0", edgecolor="#94A3B8"))

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba)
    plt.close(fig)
    return img


def render_schematic_preview(
    output_path: str = "diffusion_preview.gif",
    num_frames: int = 30,
    room_types: Optional[List[str]] = None
) -> str:
    """
    Renders an illustrative preview animation using geometric interpolation.
    Explicitly labeled as an illustrative preview for UI responsiveness.
    """
    if room_types is None:
        room_types = ["living_room", "master_bedroom", "second_bedroom", "bathroom", "kitchen", "balcony"]

    print(f"[FloorGen Animation] Generating illustrative preview ({num_frames} frames)...")
    target_boxes = np.array([
        [-0.4, -0.6, 0.4, 0.4],
        [0.4, -0.6, 0.9, 0.0],
        [0.4, 0.0, 0.9, 0.6],
        [-0.9, 0.0, -0.4, 0.6],
        [-0.9, -0.6, -0.4, 0.0],
        [-0.4, 0.4, 0.4, 0.7]
    ])

    np.random.seed(42)
    current_boxes = np.random.randn(*target_boxes.shape) * 0.9

    frames = []
    for step in range(num_frames + 1):
        t_ratio = step / float(num_frames)
        noise_level = (1.0 - t_ratio) * 0.35
        stochastic_jitter = np.random.randn(*target_boxes.shape) * noise_level
        interp_boxes = (1.0 - t_ratio) * current_boxes + t_ratio * target_boxes + stochastic_jitter
        interp_boxes = np.clip(interp_boxes, -1.0, 1.0)

        frame = render_diffusion_frame(
            boxes=interp_boxes,
            room_types=room_types,
            step=step,
            total_steps=num_frames,
            is_preview=True
        )
        frames.append(frame)

    for _ in range(6):
        frames.append(frames[-1])

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    return output_path


def generate_diffusion_animation(
    output_path: str = "diffusion_synthesis.gif",
    num_frames: int = 30,
    room_types: Optional[List[str]] = None,
    device: Optional[str] = None
) -> str:
    """
    Renders real reverse-diffusion model inference trajectories into an animated GIF.
    Extracts true step-by-step x_t coordinate states directly from RAGFloorplanDiffusion.
    """
    if room_types is None:
        room_types = ["living_room", "master_bedroom", "second_bedroom", "bathroom", "kitchen", "balcony"]

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FloorGen Animation] Executing real reverse diffusion model trajectory ({num_frames} steps on {dev})...")

    scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine").to(dev)
    model = RAGFloorplanDiffusion(hidden_dim=256, num_layers=4).to(dev)

    # Load checkpoint if available
    ckpt_path = "checkpoints/rag_diffusion.pt"
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            sd = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(sd)
            print(f"[FloorGen Animation] Loaded weights from {ckpt_path}")
        except Exception as e:
            print(f"[FloorGen Animation] Notice loading checkpoint: {e}")
    model.eval()

    n_rooms = len(room_types)
    max_rooms = max(n_rooms, 16)
    room_types_t = torch.zeros((1, max_rooms), dtype=torch.long, device=dev)
    room_mask_t = torch.zeros((1, max_rooms), dtype=torch.bool, device=dev)
    adj_matrix_t = torch.zeros((1, max_rooms, max_rooms), dtype=torch.float32, device=dev)

    for i, r in enumerate(room_types):
        room_types_t[0, i] = ROOM_TYPE_TO_ID.get(r, 0)
        room_mask_t[0, i] = True
        if i > 0:
            adj_matrix_t[0, 0, i] = 1.0
            adj_matrix_t[0, i, 0] = 1.0

    # Sample with real step recording
    with torch.no_grad():
        final_boxes, intermediate_steps = model.sample(
            scheduler=scheduler,
            room_types=room_types_t,
            adj_matrix=adj_matrix_t,
            room_mask=room_mask_t,
            num_inference_steps=num_frames,
            method="ddim",
            return_intermediates=True
        )

    print(f"[FloorGen Animation] Recorded {len(intermediate_steps)} real diffusion denoising states.")

    frames = []
    total_steps = len(intermediate_steps) - 1
    for step_idx, step_boxes in enumerate(intermediate_steps):
        boxes_np = step_boxes[0, :n_rooms].cpu().numpy()
        frame = render_diffusion_frame(
            boxes=boxes_np,
            room_types=room_types,
            step=step_idx,
            total_steps=total_steps,
            is_preview=False
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
    print(f"[Animation] Real model diffusion animation saved successfully to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorGen Diffusion Animation Generator")
    parser.add_argument("--output", type=str, default="diffusion_synthesis.gif")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--preview", action="store_true", help="Generate fast illustrative preview instead of real model inference")
    args = parser.parse_args()

    if args.preview:
        render_schematic_preview(output_path=args.output, num_frames=args.frames)
    else:
        generate_diffusion_animation(output_path=args.output, num_frames=args.frames)
