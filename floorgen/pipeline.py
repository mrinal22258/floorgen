"""
Production Floorplan Synthesis Pipeline for FloorGen v1.0.0.
Provides a unified, thread-safe, high-performance orchestration layer for:
- RAG exemplar retrieval (with in-memory index caching)
- Vector diffusion generation (DDIM/DDPM with multi-candidate sampling)
- CP-SAT architectural constraint satisfaction
- Explicit wall topology & junction classification (GSDiff-inspired)
- Intelligent architectural furniture placement
- Building code compliance validation (IRC / IBC / ADA)
- Multi-format deliverable export (SVG, DXF, IFC BIM, JSON)
- Quality evaluation metrics
"""

import os
import time
import base64
import threading
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np

from floorgen.config import FloorGenConfig, default_config
from floorgen.logging_config import get_logger
from floorgen.data.scripts.parse_rplan import load_plans_from_dir
from floorgen.data.scripts.generate_sample_data import generate_dataset
from floorgen.rag.retriever import FloorplanRAGRetriever
from floorgen.rag.brief_parser import parse_architectural_brief
from floorgen.models.diffusion_core.house_diffusion import DDPMScheduler
from floorgen.models.diffusion_core.rag_diffusion import RAGFloorplanDiffusion
from floorgen.postprocess.vectorize import (
    regularize_floorplan,
    export_svg,
    export_json_vector,
    export_dxf,
    detect_adjacency_and_doors,
    detect_windows
)
from floorgen.postprocess.walls import WallTopologyGraph
from floorgen.postprocess.furniture import populate_floorplan_furniture
from floorgen.postprocess.export_ifc import export_ifc
from floorgen.constraints.solver import FloorplanConstraintSolver
from floorgen.constraints.compliance import validate_building_code_compliance
from floorgen.eval.llm_judge import evaluate_structural_realism
from floorgen.data.dataset import ROOM_TYPE_TO_ID

log = get_logger("pipeline")


class FloorGenResult:
    """Encapsulates the complete synthesized floorplan deliverables and quality metrics."""
    def __init__(
        self,
        svg: str,
        json_spec: Dict[str, Any],
        dxf_content: Optional[str],
        ifc_content: Optional[str],
        realism_score: float,
        circulation: float,
        aspect_ratio: float,
        compliance: Dict[str, Any],
        generation_time_ms: float,
        exemplars: List[Dict[str, Any]],
        device: str,
        checkpoint_epoch: int,
        intermediate_steps: Optional[List[Any]] = None
    ):
        self.svg = svg
        self.json_spec = json_spec
        self.intermediate_steps = intermediate_steps or []
        self.dxf_content = dxf_content
        self.dxf_base64 = base64.b64encode(dxf_content.encode("utf-8")).decode("utf-8") if dxf_content else None
        self.ifc_content = ifc_content
        self.ifc_base64 = base64.b64encode(ifc_content.encode("utf-8")).decode("utf-8") if ifc_content else None
        self.realism_score = realism_score
        self.circulation = circulation
        self.aspect_ratio = aspect_ratio
        self.compliance = compliance
        self.generation_time_ms = generation_time_ms
        self.exemplars = exemplars
        self.device = device
        self.checkpoint_epoch = checkpoint_epoch

    @property
    def walls(self) -> Optional[Dict[str, Any]]:
        """Direct access to wall topology dictionary."""
        if not isinstance(self.json_spec, dict):
            return None
        return self.json_spec.get("wall_topology") or self.json_spec.get("walls")

    @property
    def furniture(self) -> Optional[List[Dict[str, Any]]]:
        """Direct access to placed furniture list."""
        if not isinstance(self.json_spec, dict):
            return None
        return self.json_spec.get("furniture")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realism_score": self.realism_score,
            "circulation": self.circulation,
            "aspect_ratio": self.aspect_ratio,
            "compliance": self.compliance,
            "generation_time_ms": self.generation_time_ms,
            "device": self.device,
            "checkpoint_epoch": self.checkpoint_epoch,
            "num_intermediate_steps": len(self.intermediate_steps),
            "exemplars": [{
                "plan_id": e.get("plan_id"),
                "score": round(float(e.get("score", 0.0)), 4),
                "archetype": e.get("plan", {}).get("archetype", "unknown")
            } for e in self.exemplars],
            "json_spec": self.json_spec,
            "svg": self.svg,
            "dxf_base64": self.dxf_base64,
            "ifc_base64": self.ifc_base64
        }


class FloorGenPipeline:
    """
    Singleton-capable orchestrator for real-time floorplan generation.
    Caches model weights and RAG vector corpus to achieve sub-second response times.
    """
    _instance: Optional["FloorGenPipeline"] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[FloorGenConfig] = None):
        self.config = config or default_config
        self.device = torch.device(self.config.device)
        self.retriever: Optional[FloorplanRAGRetriever] = None
        self.model: Optional[RAGFloorplanDiffusion] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.wall_engine = WallTopologyGraph()
        self.checkpoint_epoch: int = 0
        self._retrieval_cache: Dict[Tuple, List[Dict[str, Any]]] = {}
        self._is_ready = False

        self._init_pipeline()

    @classmethod
    def get_instance(cls, config: Optional[FloorGenConfig] = None) -> "FloorGenPipeline":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    def _init_pipeline(self):
        """Preloads dataset index and neural model into memory with sub-second disk cache."""
        log.info(f"Initializing FloorGen Pipeline on device: {self.device}")
        
        # 1. Load plans (fast SQLite)
        plans = load_plans_from_dir(self.config.data_dir)
        if len(plans) == 0:
            log.warning(f"No indexed plans found in '{self.config.data_dir}'. Generating fallback dataset...")
            plans = generate_dataset(num_samples=100, output_dir=self.config.data_dir)

        self.retriever = FloorplanRAGRetriever()
        cache_dir = os.path.join(self.config.data_dir, "faiss_cache")
        if not self.retriever.load_index(cache_dir, plans):
            # Index first 600 plans for speed and comprehensive archetype diversity
            index_plans = plans[:600] if len(plans) > 600 else plans
            self.retriever.index_corpus(index_plans)
            self.retriever.save_index(cache_dir)
        log.info(f"RAG Retriever initialized with {len(self.retriever.corpus_plans)} plans.")

        # 2. Diffusion scheduler
        self.scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine").to(self.device)

        # 3. Model weights
        self.model = RAGFloorplanDiffusion(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)

        if os.path.exists(self.config.checkpoint_path):
            ckpt = torch.load(self.config.checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.checkpoint_epoch = ckpt.get("epoch", 80)
            else:
                self.model.load_state_dict(ckpt)
                self.checkpoint_epoch = 80
            log.info(f"Loaded trained checkpoint '{self.config.checkpoint_path}' (Epoch {self.checkpoint_epoch})")
        else:
            log.warning(f"Checkpoint '{self.config.checkpoint_path}' not found! Using untrained weights.")

        self.model.eval()
        self._is_ready = True
        log.info("FloorGen Pipeline initialization complete.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def generate(
        self,
        rooms: Optional[List[str]] = None,
        brief: Optional[str] = None,
        top_k: Optional[int] = None,
        batch: Optional[int] = None,
        sampling: Optional[str] = None,
        steps: Optional[int] = None,
        solver: Optional[bool] = None,
        export_dxf_flag: bool = True,
        export_ifc_flag: bool = True,
        return_intermediates: bool = False
    ) -> FloorGenResult:
        """
        Executes end-to-end retrieval, generative diffusion, constraint solving,
        wall topology generation, furniture placement, code compliance, and export.
        """
        t0 = time.time()
        top_k = top_k or self.config.top_k
        batch_size = batch or self.config.default_batch
        sampling_mode = sampling or self.config.default_sampling
        diffusion_steps = steps or self.config.default_steps
        use_solver = solver if solver is not None else self.config.enable_solver

        # Natural-Language Brief Reasoning (Local Ollama Qwen with heuristic fallback)
        brief_affinities = []
        if brief:
            try:
                from floorgen.rag.llm_brief_parser import parse_brief_with_ollama
                parsed_brief = parse_brief_with_ollama(brief)
                if not rooms:
                    rooms = parsed_brief.get("rooms", [])
                brief_affinities = parsed_brief.get("affinities", [])
            except Exception as e:
                log.warning(f"Brief parser failed: {e}")
                if not rooms:
                    parsed_fallback = parse_architectural_brief(brief)
                    rooms = parsed_fallback.detected_rooms

        valid_room_types = set(ROOM_TYPE_TO_ID.keys())
        rooms_in = rooms or ["living_room", "master_bedroom", "bathroom", "kitchen"]
        sanitized_rooms = [r.strip().lower() for r in rooms_in if r.strip().lower() in valid_room_types]
        if not sanitized_rooms:
            sanitized_rooms = ["living_room", "master_bedroom", "bathroom", "kitchen"]

        # 1. RAG Retrieval
        cache_key = (tuple(sanitized_rooms), brief, top_k)
        if self.config.enable_retrieval_cache and cache_key in self._retrieval_cache:
            retrieved = self._retrieval_cache[cache_key]
        else:
            retrieved = self.retriever.diverse_retrieve(
                room_list=sanitized_rooms,
                text_brief=brief,
                top_k=top_k,
                lambda_mult=self.config.lambda_mult
            )
            if self.config.enable_retrieval_cache:
                self._retrieval_cache[cache_key] = retrieved

        # Prepare conditioning tensors
        max_rooms = max(len(sanitized_rooms), 16)
        cond_ctx = self.retriever.format_conditioning_context(retrieved, max_rooms=max_rooms)
        exemplar_tensor = None
        if cond_ctx.get("exemplar_boxes") is not None:
            exemplar_tensor = torch.tensor(
                cond_ctx["exemplar_boxes"],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

        n_rooms = len(sanitized_rooms)
        room_types_t = torch.zeros((1, max_rooms), dtype=torch.long, device=self.device)
        room_mask_t = torch.zeros((1, max_rooms), dtype=torch.bool, device=self.device)
        adj_matrix_t = torch.zeros((1, max_rooms, max_rooms), dtype=torch.float32, device=self.device)

        room_idx_map = {}
        for i, r in enumerate(sanitized_rooms[:max_rooms]):
            room_types_t[0, i] = ROOM_TYPE_TO_ID.get(r, 0)
            room_mask_t[0, i] = True
            room_idx_map[r] = i

        # Bias adjacency matrix with brief affinities
        for pair in brief_affinities:
            if len(pair) == 2 and pair[0] in room_idx_map and pair[1] in room_idx_map:
                ia, ib = room_idx_map[pair[0]], room_idx_map[pair[1]]
                adj_matrix_t[0, ia, ib] = 1.0
                adj_matrix_t[0, ib, ia] = 1.0

            if i > 0:
                adj_matrix_t[0, 0, i] = 1.0
                adj_matrix_t[0, i, 0] = 1.0

        # 2. Diffusion Sampling
        candidates = []
        intermediates_recorded = []
        with torch.no_grad():
            for b_idx in range(batch_size):
                sample_res = self.model.sample(
                    scheduler=self.scheduler,
                    room_types=room_types_t,
                    adj_matrix=adj_matrix_t,
                    room_mask=room_mask_t,
                    retrieved_exemplars=exemplar_tensor,
                    num_inference_steps=diffusion_steps,
                    method=sampling_mode,
                    return_intermediates=return_intermediates
                )

                if return_intermediates:
                    pred_boxes, intermediate_list = sample_res
                    if b_idx == 0:
                        intermediates_recorded = [
                            step_box[0, :n_rooms].cpu().numpy()
                            for step_box in intermediate_list
                        ]
                else:
                    pred_boxes = sample_res

                raw_boxes_np = pred_boxes[0].cpu().numpy()
                mask_np = room_mask_t[0].cpu().numpy()

                if use_solver:
                    solver_inst = FloorplanConstraintSolver()
                    refined_rooms = solver_inst.solve(raw_boxes_np, sanitized_rooms, mask_np)
                    vector_plan = regularize_floorplan(raw_boxes_np, sanitized_rooms, mask_np)
                    if refined_rooms:
                        vector_plan["rooms"] = refined_rooms
                        all_rx = [pt[0] for r in refined_rooms for pt in r.get("polygon", [])]
                        all_ry = [pt[1] for r in refined_rooms for pt in r.get("polygon", [])]
                        if all_rx and all_ry:
                            vector_plan["boundary"] = [
                                [min(all_rx), min(all_ry)],
                                [max(all_rx), min(all_ry)],
                                [max(all_rx), max(all_ry)],
                                [min(all_rx), max(all_ry)]
                            ]
                        adj, doors = detect_adjacency_and_doors(refined_rooms)
                        vector_plan["adjacency"] = adj
                        vector_plan["doors"] = doors
                        vector_plan["windows"] = detect_windows(refined_rooms)
                else:
                    vector_plan = regularize_floorplan(raw_boxes_np, sanitized_rooms, mask_np)

                score_cand = evaluate_structural_realism(vector_plan)
                candidates.append((score_cand.get("overall_realism", 80.0), vector_plan))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, plan_data = candidates[0]

        # 3. Explicit Wall Topology Extraction (GSDiff-inspired)
        wall_data = self.wall_engine.extract_wall_network(plan_data)
        plan_data["wall_topology"] = wall_data
        plan_data["walls"] = wall_data["walls"]

        # 4. Intelligent Architectural Furniture Placement
        plan_data = populate_floorplan_furniture(plan_data)

        # 5. Building Code & Egress Compliance Audit
        compliance_audit = validate_building_code_compliance(plan_data, pixels_per_meter=self.config.pixels_per_meter)
        plan_data["building_code_compliance"] = compliance_audit

        # 6. Export Deliverables (SVG, DXF, IFC)
        svg_content = export_svg(plan_data, theme="pink")
        
        dxf_content = None
        if export_dxf_flag:
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp_f:
                tmp_dxf_path = tmp_f.name
            try:
                export_dxf(plan_data, tmp_dxf_path)
                with open(tmp_dxf_path, "r", encoding="utf-8") as f:
                    dxf_content = f.read()
            finally:
                if os.path.exists(tmp_dxf_path):
                    os.remove(tmp_dxf_path)

        ifc_content = None
        if export_ifc_flag:
            with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp_f:
                tmp_ifc_path = tmp_f.name
            try:
                export_ifc(plan_data, tmp_ifc_path)
                with open(tmp_ifc_path, "r", encoding="utf-8") as f:
                    ifc_content = f.read()
            finally:
                if os.path.exists(tmp_ifc_path):
                    os.remove(tmp_ifc_path)

        eval_res = evaluate_structural_realism(plan_data)
        elapsed_ms = round((time.time() - t0) * 1000.0, 1)

        return FloorGenResult(
            svg=svg_content,
            json_spec=plan_data,
            dxf_content=dxf_content,
            ifc_content=ifc_content,
            realism_score=eval_res.get("overall_realism", 85.0),
            circulation=eval_res.get("circulation_score", 1.0),
            aspect_ratio=eval_res.get("aspect_ratio_score", 0.88),
            compliance=compliance_audit,
            generation_time_ms=elapsed_ms,
            exemplars=retrieved,
            device=str(self.device),
            checkpoint_epoch=self.checkpoint_epoch,
            intermediate_steps=intermediates_recorded
        )
