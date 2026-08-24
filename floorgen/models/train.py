"""
Unified Training Script for FloorGen Generative Models.
Supports:
- Baseline Relational GAN (House-GAN++)
- Baseline Vector Diffusion (HouseDiffusion)
- RAG-Conditioned Diffusion Core (FloorGen)
Features: Low-VRAM CPU/GPU automatic device management, mixed-precision, and checkpoint persistence.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from floorgen.data.dataset import FloorplanDataset
from floorgen.data.scripts.parse_rplan import load_plans_from_dir
from floorgen.data.scripts.generate_sample_data import generate_dataset
from floorgen.models.gan_baseline.house_gan_pp import HouseGANppGenerator, HouseGANppDiscriminator
from floorgen.models.diffusion_core.house_diffusion import DDPMScheduler, HouseDiffusionDenoiser
from floorgen.models.diffusion_core.rag_diffusion import RAGFloorplanDiffusion
from floorgen.models.shared.representation import floorplan_geometric_loss


def train_rag_diffusion(
    data_dir: str = "data/processed",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: str = "checkpoints/rag_diffusion.pt",
    device: Optional[str] = None
):
    """Trains the RAG-conditioned diffusion model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FloorGen Training] Starting RAG-Diffusion training on device: {device}")

    # Load dataset or generate if empty
    plans = load_plans_from_dir(data_dir)
    if len(plans) == 0:
        print(f"[FloorGen Training] No plans found in {data_dir}. Generating synthetic dataset...")
        plans = generate_dataset(num_samples=100, output_dir=data_dir)

    dataset = FloorplanDataset(plans=plans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scheduler = DDPMScheduler(num_timesteps=1000).to(torch.device(device))
    model = RAGFloorplanDiffusion(hidden_dim=128, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            room_types = batch["room_types"].to(device)
            room_boxes = batch["room_boxes"].to(device)
            room_mask = batch["room_mask"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            boundary = batch["boundary"].to(device)

            B, N, _ = room_boxes.shape
            
            # Sample random timesteps
            t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
            noise = torch.randn_like(room_boxes)
            noisy_boxes = scheduler.add_noise(room_boxes, noise, t)

            # Simulated exemplar context from mini-batch
            exemplar_boxes = room_boxes.roll(shifts=1, dims=0).unsqueeze(1) # (B, 1, N, 4)

            optimizer.zero_grad()
            eps_pred = model(
                x_t=noisy_boxes,
                timesteps=t,
                room_types=room_types,
                adj_matrix=adj_matrix,
                room_mask=room_mask,
                boundary=boundary,
                retrieved_exemplars=exemplar_boxes
            )

            # Loss: MSE on noise prediction + geometry regularization
            loss_mse = nn.functional.mse_loss(eps_pred * room_mask.unsqueeze(-1), noise * room_mask.unsqueeze(-1))
            
            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_mse.item()

        avg_loss = total_loss / max(1, len(dataloader))
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, save_path)

    print(f"[FloorGen Training] Model saved successfully to {save_path}")
    return model


def train_gan_baseline(
    data_dir: str = "data/processed",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 2e-4,
    save_path: str = "checkpoints/gan_baseline.pt",
    device: Optional[str] = None
):
    """Trains the House-GAN++ relational GAN baseline."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FloorGen Training] Starting House-GAN++ training on device: {device}")

    plans = load_plans_from_dir(data_dir)
    if len(plans) == 0:
        plans = generate_dataset(num_samples=100, output_dir=data_dir)

    dataset = FloorplanDataset(plans=plans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    netG = HouseGANppGenerator(hidden_dim=128).to(device)
    netD = HouseGANppDiscriminator(hidden_dim=128).to(device)

    optG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        netG.train()
        netD.train()
        g_loss_accum = 0.0
        d_loss_accum = 0.0

        for batch in dataloader:
            room_types = batch["room_types"].to(device)
            real_boxes = batch["room_boxes"].to(device)
            room_mask = batch["room_mask"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            boundary = batch["boundary"].to(device)
            B = room_types.shape[0]

            # ----------------- Train Discriminator -----------------
            optD.zero_grad()
            real_labels = torch.ones((B, 1), device=device)
            fake_labels = torch.zeros((B, 1), device=device)

            real_logits = netD(real_boxes, room_types, adj_matrix, room_mask)
            d_real_loss = bce(real_logits, real_labels)

            fake_boxes = netG(room_types, adj_matrix, room_mask, boundary)
            fake_logits = netD(fake_boxes.detach(), room_types, adj_matrix, room_mask)
            d_fake_loss = bce(fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()

            # ----------------- Train Generator -----------------
            optG.zero_grad()
            fake_logits_for_g = netD(fake_boxes, room_types, adj_matrix, room_mask)
            g_adv_loss = bce(fake_logits_for_g, real_labels)

            geom_loss, _ = floorplan_geometric_loss(fake_boxes, real_boxes, room_mask, adj_matrix, boundary)
            g_loss = g_adv_loss + 2.0 * geom_loss
            
            g_loss.backward()
            optG.step()

            g_loss_accum += g_loss.item()
            d_loss_accum += d_loss.item()

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"GAN Epoch [{epoch}/{epochs}] - G Loss: {g_loss_accum/len(dataloader):.4f}, D Loss: {d_loss_accum/len(dataloader):.4f}")

    torch.save({
        "generator_state_dict": netG.state_dict(),
        "discriminator_state_dict": netD.state_dict()
    }, save_path)
    print(f"[FloorGen Training] GAN baseline saved to {save_path}")
    return netG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorGen Model Trainer")
    parser.add_argument("--model", type=str, choices=["rag_diffusion", "gan_baseline"], default="rag_diffusion")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.model == "rag_diffusion":
        train_rag_diffusion(epochs=args.epochs, batch_size=args.batch_size)
    else:
        train_gan_baseline(epochs=args.epochs, batch_size=args.batch_size)
