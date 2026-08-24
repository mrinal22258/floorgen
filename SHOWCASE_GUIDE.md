# FloorGen: Publication, Video Showcase & Viral Demo Playbook

This document details how to take **FloorGen** from code into a **publishable research paper, viral video demo, and interactive portfolio project**.

---

## 1. What is the Core Use Case & Value Proposition?

### 🏢 Practical & Commercial Impact
1. **Instant Architectural Drafting**: Architects & designers spend hours drafting initial CAD bubble diagrams. FloorGen generates vector floorplans in **under 500 milliseconds**.
2. **Real Estate & PropTech Generation**: Generate hundreds of customized variations of apartments matching a building site boundary and client preferences.
3. **Gaming & Virtual Worlds**: Procedural generation of indoor building layouts for Unreal Engine 5 / Unity, robotics simulators (Habitat, AI2-THOR).

### 🔬 Academic Novelty (Why it's publishable)
- **Problem**: Standard generative models either hallucinate invalid rooms (LLMs) or produce blurry raster pixels (Stable Diffusion).
- **FloorGen's Contribution**: Retrieval-Augmented Vector Diffusion (RAG-Diffusion) that retrieves the top-$k$ nearest architectural exemplars from a dual **vector (FAISS) + topological graph (Neo4j)** index and injects them as continuous cross-attention priors into coordinate diffusion denoising.

---

## 2. Generating Eye-Catching Videos & Visuals

### A. Automatic Diffusion Denoising Video / GIF
Run the animation generator to create a high-framerate animation showing Gaussian noise resolving into an architectural floorplan:
```bash
python -m floorgen.animate_diffusion --output diffusion_synthesis.gif --frames 30
```
This produces an animated demonstration of the reverse diffusion trajectory, showing the gradual crystallization of walls, doors, and room labels.

### B. Interactive 3D Isometric View & Real-Time Scrubbing
Launch the Web UI:
```bash
python -m floorgen.demo.web_app --port 8000
```
- Open `http://localhost:8000`
- Click **"3D Isometric Model"** to see live 3D wall extrusions and lighting rendered in WebGL/Three.js with an orbiting camera.
- Drag the **"Diffusion Reverse Trajectory" slider** to scrub through diffusion timesteps $T=100 \to 0$ in real time.

---

## 3. How to Make It Publishable (Step-by-Step)

### Step 1: Benchmark Submission & LaTeX Paper
- Use the comparative benchmark table in `README.md` containing **FID, KID, Graph Edit Distance (GED), and Realism scores**.
- Write a 4 to 8-page workshop/conference paper (targeted for **CVPR / ICCV / NeurIPS Machine Learning for Design Workshop / SIGGRAPH** or **arXiv**).
- Paper title template: *"FloorGen: Retrieval-Augmented Generative Floorplan Synthesis via Structural Graph Diffusion"*.

### Step 2: Hugging Face Spaces 1-Click Interactive Demo
- Upload the repository to **Hugging Face Spaces** (Gradio / HTML) under a free CPU instance.
- Tag with `#GenerativeAI`, `#DiffusionModels`, `#Architecture`, `#RAG`.

### Step 3: Social & Video Showcase (LinkedIn / X / YouTube)
1. **Hook (0:00 - 0:03)**: Screen recording of dragging the slider to watch the 3D floorplan extrude and materialize from noise.
2. **Concept (0:03 - 0:15)**: Show the RAG retrieval module picking 5 similar real floorplans from FAISS.
3. **Comparison (0:15 - 0:30)**: Side-by-side showing standard GAN baseline (messy overlaps) vs FloorGen RAG-Diffusion (crisp, regularized vector layout with doors).
4. **Call to Action**: Link to the GitHub repo and free Google Colab notebook.
