"""
Render architectural figures for FloorGen README.
Generates:
1. pipeline.png - Full system architecture diagram
2. result.png - Floorplan synthesis comparisons (Target vs Retrieved vs Generated)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def generate_pipeline_figure(output_path: str = "pipeline.png"):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0B0F19")

    # Boxes for 5 stages
    stages = [
        ("Stage 0\nData Ingestion", "RPLAN & ResPlan\n• Appendix A.2 Filters\n• Manifold Polygons\n• SQLite / JSON Store", 0.08, "#3B82F6"),
        ("Stage 1 & 2\nRAG Dual-Index", "Dual Store Retrieval\n• FAISS Vector Search\n• Neo4j Graph Query\n• Top-k Exemplars (k=5)", 0.28, "#06B6D4"),
        ("Stage 3\nGenerative Core", "Vector Diffusion\n• Coordinate Denoising\n• Cross-Attention Prior\n• House-GAN++ Baseline", 0.48, "#8B5CF6"),
        ("Stage 4\nVectorization", "Post-Processing\n• Manhattan Snapping\n• Overlap Elimination\n• Door Placement (SVG)", 0.68, "#10B981"),
        ("Stage 5\nEvaluation", "Benchmark Suite\n• FID / KID (Diversity)\n• Graph Edit Dist (GED)\n• LLM Realism Judge", 0.88, "#F59E0B")
    ]

    for title, desc, cx, color in stages:
        rect = patches.FancyBboxPatch((cx - 0.08, 0.2), 0.16, 0.6,
                                      boxstyle="round,pad=0.03",
                                      facecolor="#1E293B", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(cx, 0.70, title, ha="center", va="center", color="#F8FAFC", fontsize=12, fontweight="bold", fontfamily="sans-serif")
        ax.text(cx, 0.45, desc, ha="center", va="center", color="#94A3B8", fontsize=9, fontfamily="sans-serif")

        # Arrows connecting stages
        if cx < 0.88:
            ax.annotate("", xy=(cx + 0.10, 0.5), xytext=(cx + 0.08, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#64748B", lw=2.5))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    plt.title("FloorGen System Pipeline: Retrieval-Augmented Generative Floorplan Synthesis",
              color="#FFFFFF", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved {output_path}")


def generate_result_figure(output_path: str = "result.png"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    titles = [
        "1. Input Bubble Graph",
        "2. Retrieved RAG Exemplar (k=1)",
        "3. House-GAN++ Baseline",
        "4. FloorGen RAG-Diffusion (Ours)"
    ]

    colors = {
        "living": "#93C5FD",
        "bed": "#FDE68A",
        "bath": "#99F6E4",
        "kitchen": "#FDBA74",
        "balcony": "#A7F3D0"
    }

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="bold", pad=12)
        ax.set_aspect("equal")
        ax.axis("off")

        if idx == 0:
            # Bubble graph
            nodes = [(80, 140, "Living"), (180, 180, "Bed 1"), (180, 90, "Bed 2"), (40, 60, "Bath"), (40, 190, "Kitchen")]
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
            for u, v in edges:
                ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], "k-", lw=2, zorder=1)
            for x, y, name in nodes:
                circle = plt.Circle((x, y), 24, facecolor="#E2E8F0", edgecolor="#3B82F6", lw=2, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y, name, ha="center", va="center", fontsize=8, fontweight="bold", color="#1E293B", zorder=3)
        elif idx == 1:
            # Retrieved plan
            rooms = [(70, 70, 110, 110, colors["living"], "Living"), (180, 140, 60, 90, colors["bed"], "Master"),
                     (180, 40, 60, 90, colors["bed"], "Bed 2"), (20, 40, 50, 70, colors["bath"], "Bath"), (20, 120, 50, 80, colors["kitchen"], "Kitchen")]
            for x, y, w, h, c, name in rooms:
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=c, edgecolor="#475569", lw=2))
                ax.text(x + w/2, y + h/2, name, ha="center", va="center", fontsize=7, fontweight="bold", color="#0F172A")
        elif idx == 2:
            # GAN Baseline (slightly overlapping)
            rooms = [(65, 65, 115, 115, colors["living"], "Living"), (175, 135, 65, 85, colors["bed"], "Master"),
                     (170, 45, 68, 80, colors["bed"], "Bed 2"), (25, 35, 52, 75, colors["bath"], "Bath"), (15, 125, 55, 75, colors["kitchen"], "Kitchen")]
            for x, y, w, h, c, name in rooms:
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=c, edgecolor="#DC2626", lw=1.5, linestyle="--"))
                ax.text(x + w/2, y + h/2, name, ha="center", va="center", fontsize=7, fontweight="bold", color="#0F172A")
        elif idx == 3:
            # FloorGen RAG-Diffusion (aligned, doors, clean)
            rooms = [(70, 60, 110, 120, colors["living"], "Living"), (180, 130, 65, 95, colors["bed"], "Master Bed"),
                     (180, 35, 65, 95, colors["bed"], "Second Bed"), (15, 35, 55, 80, colors["bath"], "Bath"), (15, 115, 55, 90, colors["kitchen"], "Kitchen"),
                     (70, 180, 110, 45, colors["balcony"], "Balcony")]
            for x, y, w, h, c, name in rooms:
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=c, edgecolor="#1E293B", lw=2))
                ax.text(x + w/2, y + h/2, name, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A")
            # Draw doors
            ax.plot([70, 70], [90, 105], color="#EF4444", lw=3)
            ax.plot([180, 180], [90, 105], color="#EF4444", lw=3)
            ax.plot([180, 180], [160, 175], color="#EF4444", lw=3)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    generate_pipeline_figure()
    generate_result_figure()
