"""
MoE Theoretical Analysis Engine
================================
A comprehensive theoretical analysis toolkit for Mixture-of-Experts (MoE) models.
Fetches model configs from HuggingFace and computes hardware-aware projections
across compute, memory, communication, and efficiency dimensions.

Author: [Your Name]
License: MIT

Usage:
    analyzer = MoEAnalyzer("mistralai/Mixtral-8x7B-v0.1")
    report = analyzer.full_analysis()
    analyzer.print_report(report)
"""

import json
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError


# =============================================================================
# Section 1: GPU Hardware Database
# =============================================================================

@dataclass
class GPUSpec:
    """Hardware specification for a single GPU."""
    name: str
    # Compute
    fp16_tflops: float          # Peak FP16/BF16 TFLOPS (tensor core)
    fp32_tflops: float          # Peak FP32 TFLOPS
    int8_tops: float            # Peak INT8 TOPS
    # Memory
    hbm_capacity_gb: float      # HBM/VRAM capacity in GB
    mem_bandwidth_gbps: float   # Memory bandwidth in GB/s
    l2_cache_mb: float          # L2 cache size in MB
    # Interconnect
    nvlink_bw_gbps: float       # NVLink bidirectional BW in GB/s (0 if no NVLink)
    pcie_bw_gbps: float         # PCIe bandwidth in GB/s
    # Multi-GPU
    max_gpus_per_node: int      # Typical max GPUs per node
    has_nvswitch: bool          # Whether NVSwitch is available
    # Cost
    cloud_cost_per_hour: float  # Approximate $/hr on major clouds (0 if N/A)
    # Architecture
    architecture: str           # e.g., "Ampere", "Hopper", "Ada Lovelace"
    tdp_watts: float            # Thermal design power


GPU_DATABASE: dict[str, GPUSpec] = {
    # ---- Data Center ----
    "A100_40GB": GPUSpec(
        name="NVIDIA A100 40GB", fp16_tflops=312, fp32_tflops=19.5,
        int8_tops=624, hbm_capacity_gb=40, mem_bandwidth_gbps=1555,
        l2_cache_mb=40, nvlink_bw_gbps=600, pcie_bw_gbps=31.5,
        max_gpus_per_node=8, has_nvswitch=True, cloud_cost_per_hour=3.5,
        architecture="Ampere", tdp_watts=250
    ),
    "A100_80GB": GPUSpec(
        name="NVIDIA A100 80GB SXM", fp16_tflops=312, fp32_tflops=19.5,
        int8_tops=624, hbm_capacity_gb=80, mem_bandwidth_gbps=2039,
        l2_cache_mb=40, nvlink_bw_gbps=600, pcie_bw_gbps=31.5,
        max_gpus_per_node=8, has_nvswitch=True, cloud_cost_per_hour=5.0,
        architecture="Ampere", tdp_watts=400
    ),
    "H100_SXM": GPUSpec(
        name="NVIDIA H100 SXM", fp16_tflops=989, fp32_tflops=67,
        int8_tops=1979, hbm_capacity_gb=80, mem_bandwidth_gbps=3350,
        l2_cache_mb=50, nvlink_bw_gbps=900, pcie_bw_gbps=63,
        max_gpus_per_node=8, has_nvswitch=True, cloud_cost_per_hour=8.5,
        architecture="Hopper", tdp_watts=700
    ),
    "H100_NVL": GPUSpec(
        name="NVIDIA H100 NVL", fp16_tflops=835, fp32_tflops=67,
        int8_tops=1671, hbm_capacity_gb=94, mem_bandwidth_gbps=3938,
        l2_cache_mb=50, nvlink_bw_gbps=900, pcie_bw_gbps=63,
        max_gpus_per_node=8, has_nvswitch=True, cloud_cost_per_hour=10.0,
        architecture="Hopper", tdp_watts=400
    ),
    "H200": GPUSpec(
        name="NVIDIA H200 SXM", fp16_tflops=989, fp32_tflops=67,
        int8_tops=1979, hbm_capacity_gb=141, mem_bandwidth_gbps=4800,
        l2_cache_mb=50, nvlink_bw_gbps=900, pcie_bw_gbps=63,
        max_gpus_per_node=8, has_nvswitch=True, cloud_cost_per_hour=12.0,
        architecture="Hopper", tdp_watts=700
    ),
    "B200": GPUSpec(
        name="NVIDIA B200", fp16_tflops=2250, fp32_tflops=180,
        int8_tops=4500, hbm_capacity_gb=192, mem_bandwidth_gbps=8000,
        l2_cache_mb=64, nvlink_bw_gbps=1800, pcie_bw_gbps=63,
        max_gpus_per_node=8, has_nvswitch=True, cloud_cost_per_hour=18.0,
        architecture="Blackwell", tdp_watts=1000
    ),
    "MI300X": GPUSpec(
        name="AMD Instinct MI300X", fp16_tflops=1307, fp32_tflops=163.4,
        int8_tops=2614, hbm_capacity_gb=192, mem_bandwidth_gbps=5300,
        l2_cache_mb=256, nvlink_bw_gbps=0, pcie_bw_gbps=63,
        max_gpus_per_node=8, has_nvswitch=False, cloud_cost_per_hour=8.0,
        architecture="CDNA3", tdp_watts=750
    ),
    # ---- Consumer / Prosumer ----
    "RTX_3090": GPUSpec(
        name="NVIDIA RTX 3090", fp16_tflops=71, fp32_tflops=35.6,
        int8_tops=142, hbm_capacity_gb=24, mem_bandwidth_gbps=936,
        l2_cache_mb=6, nvlink_bw_gbps=0, pcie_bw_gbps=25,
        max_gpus_per_node=1, has_nvswitch=False, cloud_cost_per_hour=0.5,
        architecture="Ampere", tdp_watts=350
    ),
    "RTX_4090": GPUSpec(
        name="NVIDIA RTX 4090", fp16_tflops=165, fp32_tflops=82.6,
        int8_tops=661, hbm_capacity_gb=24, mem_bandwidth_gbps=1008,
        l2_cache_mb=72, nvlink_bw_gbps=0, pcie_bw_gbps=31.5,
        max_gpus_per_node=1, has_nvswitch=False, cloud_cost_per_hour=0.7,
        architecture="Ada Lovelace", tdp_watts=450
    ),
    "RTX_5090": GPUSpec(
        name="NVIDIA RTX 5090", fp16_tflops=419, fp32_tflops=105,
        int8_tops=838, hbm_capacity_gb=32, mem_bandwidth_gbps=1792,
        l2_cache_mb=96, nvlink_bw_gbps=0, pcie_bw_gbps=63,
        max_gpus_per_node=1, has_nvswitch=False, cloud_cost_per_hour=1.0,
        architecture="Blackwell", tdp_watts=575
    ),
    # ---- Apple Silicon (Unified Memory - special case) ----
    "M2_Ultra": GPUSpec(
        name="Apple M2 Ultra", fp16_tflops=27.2, fp32_tflops=13.6,
        int8_tops=54.4, hbm_capacity_gb=192, mem_bandwidth_gbps=800,
        l2_cache_mb=48, nvlink_bw_gbps=0, pcie_bw_gbps=0,
        max_gpus_per_node=1, has_nvswitch=False, cloud_cost_per_hour=0,
        architecture="Apple Silicon", tdp_watts=150
    ),
    "M4_Max": GPUSpec(
        name="Apple M4 Max", fp16_tflops=19.4, fp32_tflops=9.7,
        int8_tops=38.8, hbm_capacity_gb=128, mem_bandwidth_gbps=546,
        l2_cache_mb=48, nvlink_bw_gbps=0, pcie_bw_gbps=0,
        max_gpus_per_node=1, has_nvswitch=False, cloud_cost_per_hour=0,
        architecture="Apple Silicon", tdp_watts=75
    ),
}


# =============================================================================
# Section 2: MoE Model Config Parsing
# =============================================================================

@dataclass
class MoEConfig:
    """Unified MoE model configuration extracted from HuggingFace."""
    model_name: str
    # Core architecture
    hidden_size: int                # d_model
    intermediate_size: int          # FFN intermediate (per expert)
    num_hidden_layers: int          # Total transformer layers
    num_attention_heads: int        # Attention heads
    num_key_value_heads: int        # KV heads (for GQA)
    head_dim: int                   # Per-head dimension
    vocab_size: int
    max_position_embeddings: int
    # MoE-specific
    num_experts: int                # Total experts per MoE layer
    num_experts_per_tok: int        # top-k experts selected
    num_moe_layers: int             # How many layers are MoE (vs dense)
    num_shared_experts: int         # Shared experts (DeepSeek-style)
    expert_intermediate_size: int   # Expert FFN hidden dim (may differ from intermediate_size)
    # Precision
    dtype_bytes: int                # 2 for BF16/FP16, 4 for FP32
    # Router
    router_aux_loss_coef: float     # Auxiliary load-balancing loss weight
    # Derived
    num_dense_layers: int = 0

    def __post_init__(self):
        self.num_dense_layers = self.num_hidden_layers - self.num_moe_layers


def fetch_hf_config(model_name: str) -> dict:
    """Fetch model config.json from HuggingFace Hub."""
    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    req = Request(url, headers={"User-Agent": "MoE-Analyzer/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"Model '{model_name}' not found on HuggingFace. "
                "Check the model name (e.g., 'mistralai/Mixtral-8x7B-v0.1')."
            ) from e
        raise ConnectionError(f"Failed to fetch config: HTTP {e.code}") from e
    except Exception as e:
        raise ConnectionError(f"Failed to fetch config for '{model_name}': {e}") from e


def parse_moe_config(model_name: str, raw: dict) -> MoEConfig:
    """
    Parse raw HuggingFace config into unified MoEConfig.
    Handles: Mixtral, DeepSeek-MoE/V2/V3, Qwen2-MoE, DBRX, Grok, Jamba,
    Arctic, OLMoE, and other common MoE architectures.
    """
    arch = raw.get("architectures", [None])[0] or ""
    model_type = raw.get("model_type", "").lower()

    hidden = raw.get("hidden_size", raw.get("d_model", 4096))
    n_layers = raw.get("num_hidden_layers", raw.get("n_layers", 32))
    n_heads = raw.get("num_attention_heads", raw.get("n_heads", 32))
    n_kv = raw.get("num_key_value_heads", raw.get("num_kv_heads", n_heads))
    head_dim = raw.get("head_dim", hidden // n_heads)
    vocab = raw.get("vocab_size", 32000)
    max_pos = raw.get("max_position_embeddings", raw.get("max_seq_len", 4096))

    # --- MoE params: handle various config conventions ---

    # Check for nested router/moe config (DBRX style)
    ffn_config = raw.get("ffn_config", {})
    moe_config = raw.get("moe", {})

    # Number of experts
    n_experts = (
        raw.get("num_local_experts")
        or raw.get("num_experts")
        or raw.get("n_routed_experts")
        or ffn_config.get("moe_num_experts")
        or moe_config.get("num_experts")
        or 8
    )

    # Top-k
    top_k = (
        raw.get("num_experts_per_tok")
        or raw.get("num_selected_experts")
        or raw.get("num_experts_per_token")
        or raw.get("topk")
        or raw.get("top_k")
        or ffn_config.get("moe_top_k")
        or moe_config.get("num_experts_per_tok")
        or 2
    )

    # Intermediate size
    intermediate = raw.get("intermediate_size", hidden * 4)
    expert_intermediate = (
        raw.get("expert_intermediate_size")
        or raw.get("moe_intermediate_size")
        or ffn_config.get("ffn_hidden_size")
        or intermediate
    )

    # Shared experts (DeepSeek V2/V3 style)
    n_shared = raw.get("n_shared_experts", raw.get("num_shared_experts", 0))

    # Number of MoE layers — some models alternate dense/MoE layers
    moe_layer_freq = raw.get("moe_layer_frequency", raw.get("expert_interval", 1))
    # DeekSeek V2: first layer is dense, rest are MoE
    first_moe = raw.get("first_k_dense_replace", 0)
    if moe_layer_freq > 0:
        n_moe_layers = max(0, (n_layers - first_moe) // moe_layer_freq)
    else:
        n_moe_layers = n_layers  # all layers are MoE

    # Router aux loss
    aux_loss = (
        raw.get("router_aux_loss_coef")
        or raw.get("router_aux_loss_factor")
        or raw.get("aux_loss_alpha", 0.01)
    )

    return MoEConfig(
        model_name=model_name,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        num_experts=n_experts,
        num_experts_per_tok=top_k,
        num_moe_layers=n_moe_layers,
        num_shared_experts=n_shared,
        expert_intermediate_size=expert_intermediate,
        dtype_bytes=2,  # BF16 default
        router_aux_loss_coef=aux_loss,
    )


# =============================================================================
# Section 3: Parameter & Memory Analysis
# =============================================================================

@dataclass
class ParameterBreakdown:
    """Detailed parameter count breakdown."""
    embedding_params: int
    attention_params_per_layer: int
    expert_params_per_expert: int
    router_params_per_layer: int
    shared_expert_params_per_layer: int
    layernorm_params_per_layer: int
    total_expert_params: int
    total_attention_params: int
    total_shared_params: int        # attention + embedding + layernorm + shared experts
    total_params: int               # everything
    active_params_per_token: int    # what fires for a single token


def compute_parameter_breakdown(cfg: MoEConfig) -> ParameterBreakdown:
    """Count parameters across all model components."""
    d = cfg.hidden_size
    d_ff = cfg.expert_intermediate_size
    d_dense_ff = cfg.intermediate_size
    V = cfg.vocab_size

    # Embedding: input + output (often tied, count once if tied)
    embed = V * d  # assume tied embeddings — count once

    # Attention per layer: Q, K, V projections + output projection
    # With GQA: Q has n_heads * head_dim, K/V have n_kv_heads * head_dim
    q_proj = d * (cfg.num_attention_heads * cfg.head_dim)
    kv_proj = 2 * d * (cfg.num_key_value_heads * cfg.head_dim)
    o_proj = (cfg.num_attention_heads * cfg.head_dim) * d
    attn_per_layer = q_proj + kv_proj + o_proj

    # Expert FFN: typically gate_proj + up_proj + down_proj (SwiGLU style)
    # gate: d -> d_ff, up: d -> d_ff, down: d_ff -> d
    expert_params = 3 * d * d_ff  # SwiGLU: gate + up + down

    # Shared experts
    shared_expert = cfg.num_shared_experts * 3 * d * d_ff if cfg.num_shared_experts > 0 else 0

    # Router: linear projection d_model -> num_experts
    router_per_layer = d * cfg.num_experts

    # LayerNorm: 2 per layer (pre-attn, pre-ffn), each has 'd' params
    ln_per_layer = 2 * d

    # Dense FFN layers (non-MoE layers)
    dense_ffn = 3 * d * d_dense_ff  # same SwiGLU structure

    # Totals
    total_expert = cfg.num_moe_layers * cfg.num_experts * expert_params
    total_attn = cfg.num_hidden_layers * attn_per_layer
    total_router = cfg.num_moe_layers * router_per_layer
    total_ln = cfg.num_hidden_layers * ln_per_layer
    total_shared_expert = cfg.num_moe_layers * shared_expert
    total_dense_ffn = cfg.num_dense_layers * dense_ffn

    total_shared = total_attn + embed + total_ln + total_router + total_shared_expert + total_dense_ffn
    total = total_shared + total_expert

    # Active params per token: shared + top_k experts per MoE layer + dense FFN layers
    active_expert_per_token = cfg.num_moe_layers * cfg.num_experts_per_tok * expert_params
    active_dense_ffn = cfg.num_dense_layers * dense_ffn
    active = total_attn + embed + total_ln + total_router + active_expert_per_token + total_shared_expert + active_dense_ffn

    return ParameterBreakdown(
        embedding_params=embed,
        attention_params_per_layer=attn_per_layer,
        expert_params_per_expert=expert_params,
        router_params_per_layer=router_per_layer,
        shared_expert_params_per_layer=shared_expert,
        layernorm_params_per_layer=ln_per_layer,
        total_expert_params=total_expert,
        total_attention_params=total_attn,
        total_shared_params=total_shared,
        total_params=total,
        active_params_per_token=active,
    )


@dataclass
class MemoryBreakdown:
    """Memory usage breakdown in bytes."""
    model_weights: float
    kv_cache: float
    activations: float
    optimizer_states: float   # for training
    gradients: float          # for training
    total_inference: float
    total_training: float


def compute_memory_breakdown(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    batch_size: int = 1,
    seq_len: int = 2048,
    quant_bits: int = 16,
    training: bool = False,
    optimizer: str = "adamw",  # "adamw" | "sgd" | "adam_8bit"
) -> MemoryBreakdown:
    """
    Compute memory requirements for inference and training.

    Quantization aware: adjusts weight memory based on quant_bits.
    Accounts for KV cache with GQA compression.
    """
    bpw = quant_bits / 8  # bytes per weight

    # Model weights
    weight_mem = params.total_params * bpw

    # KV Cache: per layer, per token, K and V each have n_kv_heads * head_dim values
    kv_per_token_per_layer = 2 * cfg.num_key_value_heads * cfg.head_dim * cfg.dtype_bytes
    kv_cache = cfg.num_hidden_layers * kv_per_token_per_layer * batch_size * seq_len

    # Activation memory (approximate peak per-layer activation)
    # Main contributors: attention scores, FFN intermediates, residual streams
    # For MoE: only top-k experts' activations are live
    attn_act = batch_size * seq_len * cfg.num_attention_heads * seq_len * 2  # attention scores (BF16)
    ffn_act = batch_size * seq_len * cfg.expert_intermediate_size * cfg.num_experts_per_tok * cfg.dtype_bytes
    residual = batch_size * seq_len * cfg.hidden_size * cfg.dtype_bytes * 2  # residual stream
    act_per_layer = attn_act + ffn_act + residual
    # Total activation: need to hold ~2 layers worth (pipeline)
    activations = act_per_layer * 2  # conservative: 2 layers in flight

    # Training-specific memory
    if training:
        # Gradients: same size as weights (FP32 or BF16 depending on mixed precision)
        grad_mem = params.total_params * 4  # usually FP32 gradients

        # Optimizer states
        if optimizer == "adamw":
            # AdamW: 2 states (m, v) in FP32 per parameter + master weights
            opt_mem = params.total_params * (4 + 4 + 4)  # m + v + master_weights
        elif optimizer == "adam_8bit":
            opt_mem = params.total_params * (1 + 1 + 4)  # 8-bit states + master
        else:  # SGD with momentum
            opt_mem = params.total_params * (4 + 4)  # momentum + master

        # Training activations: need full recompute-aware estimate
        # With gradient checkpointing: ~sqrt(n_layers) * per-layer activation
        activations_train = act_per_layer * math.ceil(math.sqrt(cfg.num_hidden_layers))
    else:
        grad_mem = 0
        opt_mem = 0
        activations_train = 0

    total_inf = weight_mem + kv_cache + activations
    total_train = weight_mem + kv_cache + max(activations, activations_train) + grad_mem + opt_mem

    return MemoryBreakdown(
        model_weights=weight_mem,
        kv_cache=kv_cache,
        activations=max(activations, activations_train) if training else activations,
        optimizer_states=opt_mem,
        gradients=grad_mem,
        total_inference=total_inf,
        total_training=total_train,
    )


# =============================================================================
# Section 4: FLOPs & Compute Analysis
# =============================================================================

@dataclass
class FLOPsBreakdown:
    """FLOPs per forward pass (per token and total)."""
    attention_flops_per_layer: int
    expert_flops_per_layer: int       # for active experts only
    router_flops_per_layer: int
    shared_expert_flops_per_layer: int
    dense_ffn_flops_per_layer: int
    total_flops_per_token: int
    total_flops_batch: int            # for the entire batch
    # Comparison
    equivalent_dense_flops: int       # same total params, all dense
    flops_savings_ratio: float        # MoE FLOPs / Dense FLOPs
    # Training
    total_train_flops_per_token: int  # ~3x forward (fwd + bwd + weight update)


def compute_flops(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    batch_size: int = 1,
    seq_len: int = 2048,
) -> FLOPsBreakdown:
    """
    Compute theoretical FLOPs for MoE forward pass.
    Uses the standard approximation: FLOPs ≈ 2 * MACs for matrix multiplications.
    """
    d = cfg.hidden_size
    d_ff = cfg.expert_intermediate_size
    S = seq_len
    B = batch_size
    N = B * S  # total tokens

    # Attention FLOPs per layer per token:
    # QKV projections: 2 * d * (d_q + d_k + d_v) where d_q = n_heads*head_dim, etc.
    qkv_flops = 2 * d * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
    # Attention scores: 2 * n_heads * head_dim * S (averaged over positions)
    attn_score_flops = 2 * cfg.num_attention_heads * cfg.head_dim * S
    # Attention output: 2 * n_heads * head_dim * d
    attn_out_flops = 2 * cfg.num_attention_heads * cfg.head_dim * d
    attn_per_layer = qkv_flops + attn_score_flops + attn_out_flops

    # Expert FFN FLOPs per token (SwiGLU: gate + up + down)
    # gate: 2 * d * d_ff, up: 2 * d * d_ff, down: 2 * d_ff * d
    single_expert_flops = 3 * 2 * d * d_ff
    # Active experts per token
    active_expert_flops = cfg.num_experts_per_tok * single_expert_flops

    # Shared expert FLOPs
    shared_flops = cfg.num_shared_experts * single_expert_flops if cfg.num_shared_experts > 0 else 0

    # Router FLOPs: linear d -> num_experts + softmax (negligible)
    router_flops = 2 * d * cfg.num_experts

    # Dense FFN layers
    dense_ffn_flops = 3 * 2 * d * cfg.intermediate_size if cfg.num_dense_layers > 0 else 0

    # Per-token total
    per_token = (
        cfg.num_hidden_layers * attn_per_layer
        + cfg.num_moe_layers * (active_expert_flops + shared_flops + router_flops)
        + cfg.num_dense_layers * dense_ffn_flops
    )

    # Dense equivalent (all experts fire = all layers dense with total FFN params)
    dense_equiv_ffn = cfg.num_experts * single_expert_flops  # all experts
    dense_equiv_per_token = (
        cfg.num_hidden_layers * attn_per_layer
        + cfg.num_moe_layers * (dense_equiv_ffn + shared_flops)
        + cfg.num_dense_layers * dense_ffn_flops
    )

    savings = per_token / dense_equiv_per_token if dense_equiv_per_token > 0 else 1.0

    return FLOPsBreakdown(
        attention_flops_per_layer=attn_per_layer,
        expert_flops_per_layer=active_expert_flops,
        router_flops_per_layer=router_flops,
        shared_expert_flops_per_layer=shared_flops,
        dense_ffn_flops_per_layer=dense_ffn_flops,
        total_flops_per_token=per_token,
        total_flops_batch=per_token * N,
        equivalent_dense_flops=dense_equiv_per_token,
        flops_savings_ratio=savings,
        total_train_flops_per_token=per_token * 3,  # fwd + bwd ≈ 3x fwd
    )


# =============================================================================
# Section 5: Roofline Model Analysis
# =============================================================================

@dataclass
class RooflineAnalysis:
    """Roofline model analysis for a specific GPU."""
    gpu_name: str
    # Per-expert analysis
    arithmetic_intensity_expert: float   # FLOPs / Byte for expert computation
    arithmetic_intensity_attention: float
    # Crossover points
    tokens_per_expert_to_saturate: float # tokens needed per expert to be compute-bound
    batch_size_to_saturate: int          # batch size to reach compute-bound regime
    # Achievable performance
    achievable_tflops: float             # actual TFLOPS considering memory bandwidth
    compute_utilization: float           # fraction of peak TFLOPS achievable
    # Bottleneck
    bottleneck: str                      # "compute" | "memory_bandwidth" | "communication"
    # Roofline coordinates
    operational_intensity: float         # actual AI for this model+GPU+batch combo
    peak_compute: float                  # GPU peak in TFLOPS
    ridge_point: float                   # AI where compute = bandwidth ceiling


def compute_roofline(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    flops: FLOPsBreakdown,
    gpu: GPUSpec,
    batch_size: int = 1,
    seq_len: int = 2048,
) -> RooflineAnalysis:
    """
    Roofline model: determines whether workload is compute-bound or memory-bound.

    Key MoE insight: each expert processes fewer tokens than a dense FFN,
    so the arithmetic intensity drops and MoE is more often memory-bandwidth bound.
    """
    d = cfg.hidden_size
    d_ff = cfg.expert_intermediate_size
    N = batch_size * seq_len

    # --- Expert Arithmetic Intensity ---
    # Tokens routed to each expert ≈ N * top_k / num_experts
    tokens_per_expert = N * cfg.num_experts_per_tok / cfg.num_experts

    # Expert weight bytes (loaded from HBM)
    expert_weight_bytes = params.expert_params_per_expert * cfg.dtype_bytes
    # Expert FLOPs for these tokens
    expert_flops_total = tokens_per_expert * 3 * 2 * d * d_ff  # SwiGLU

    # AI = FLOPs / Bytes (for expert computation)
    ai_expert = expert_flops_total / expert_weight_bytes if expert_weight_bytes > 0 else 0

    # --- Attention Arithmetic Intensity ---
    attn_weight_bytes = params.attention_params_per_layer * cfg.dtype_bytes
    attn_flops_total = N * flops.attention_flops_per_layer
    ai_attention = attn_flops_total / attn_weight_bytes if attn_weight_bytes > 0 else 0

    # --- Overall operational intensity ---
    total_bytes_per_layer = expert_weight_bytes * cfg.num_experts + attn_weight_bytes  # worst case: load all weights
    total_flops_per_layer = N * (flops.attention_flops_per_layer + flops.expert_flops_per_layer)
    operational_intensity = total_flops_per_layer / total_bytes_per_layer if total_bytes_per_layer > 0 else 0

    # --- GPU roofline ---
    peak_tflops = gpu.fp16_tflops
    mem_bw_tflops = gpu.mem_bandwidth_gbps / 1000  # Convert GB/s to TB/s for unit consistency

    # Ridge point: where compute ceiling meets bandwidth ceiling
    # peak_tflops (TFLOP/s) = ridge_point (FLOP/Byte) * mem_bw (TB/s)
    # ridge_point = peak_tflops / mem_bw_tflops (in FLOP/Byte, but need unit care)
    ridge_point = (peak_tflops * 1e12) / (gpu.mem_bandwidth_gbps * 1e9)  # FLOP/Byte

    # Achievable performance
    achievable = min(peak_tflops, operational_intensity * gpu.mem_bandwidth_gbps * 1e9 / 1e12)
    utilization = achievable / peak_tflops if peak_tflops > 0 else 0

    # Tokens per expert needed to saturate compute
    # When AI_expert = ridge_point, we're at the crossover
    # AI_expert ≈ tokens_per_expert (simplified for SwiGLU where FLOPs/Byte ≈ T_e)
    # More precisely: T_e * 6*d*d_ff / (3*d*d_ff*dtype_bytes) = T_e * 2 / dtype_bytes
    tokens_to_saturate = ridge_point * cfg.dtype_bytes / 2
    # Corresponding batch size
    bs_to_saturate = max(1, int(math.ceil(
        tokens_to_saturate * cfg.num_experts / (cfg.num_experts_per_tok * seq_len)
    )))

    bottleneck = "compute" if operational_intensity >= ridge_point else "memory_bandwidth"

    return RooflineAnalysis(
        gpu_name=gpu.name,
        arithmetic_intensity_expert=ai_expert,
        arithmetic_intensity_attention=ai_attention,
        tokens_per_expert_to_saturate=tokens_to_saturate,
        batch_size_to_saturate=bs_to_saturate,
        achievable_tflops=achievable,
        compute_utilization=utilization,
        bottleneck=bottleneck,
        operational_intensity=operational_intensity,
        peak_compute=peak_tflops,
        ridge_point=ridge_point,
    )


# =============================================================================
# Section 6: Communication & Distributed Analysis
# =============================================================================

@dataclass
class CommunicationAnalysis:
    """Analysis of distributed MoE communication costs."""
    # All-to-all costs
    all2all_volume_per_layer_bytes: float
    all2all_time_per_layer_ms: float
    all2all_time_total_ms: float
    # Compute-to-communication ratio
    compute_comm_ratio: float     # > 1 means compute-dominant (good)
    # Scaling
    min_gpus_for_memory: int
    recommended_ep_degree: int
    scaling_efficiency: dict       # {n_gpus: efficiency}
    # Expert parallelism
    experts_per_gpu: dict          # {n_gpus: experts_per_gpu}


def compute_communication(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    flops: FLOPsBreakdown,
    gpu: GPUSpec,
    batch_size: int = 1,
    seq_len: int = 2048,
    quant_bits: int = 16,
) -> CommunicationAnalysis:
    """
    Compute communication costs for Expert Parallelism (EP).

    MoE all-to-all pattern: each GPU sends tokens to expert-owning GPUs
    and receives processed results back. Volume = 2 * tokens * hidden * dtype.
    """
    N = batch_size * seq_len
    d = cfg.hidden_size
    bpw = quant_bits / 8

    # Model memory requirement
    total_mem = params.total_params * bpw
    single_gpu_mem = gpu.hbm_capacity_gb * 1e9
    min_gpus = max(1, math.ceil(total_mem / (single_gpu_mem * 0.85)))  # 85% usable

    # All-to-all volume per MoE layer
    # Dispatch: send tokens to experts. Combine: receive results back.
    # Each direction: N tokens * d_model * dtype_bytes
    a2a_volume = 2 * N * d * cfg.dtype_bytes

    # Effective bandwidth for all-to-all
    if gpu.nvlink_bw_gbps > 0:
        # NVLink: use bisection bandwidth ≈ nvlink_bw / 2 for all-to-all
        effective_bw = gpu.nvlink_bw_gbps * 1e9 / 2  # bytes/s
    else:
        effective_bw = gpu.pcie_bw_gbps * 1e9 / 2

    a2a_time_per_layer = a2a_volume / effective_bw if effective_bw > 0 else float('inf')
    a2a_total = a2a_time_per_layer * cfg.num_moe_layers

    # Compute time per MoE layer
    expert_compute_flops = N * flops.expert_flops_per_layer
    compute_time = expert_compute_flops / (gpu.fp16_tflops * 1e12)

    comp_comm_ratio = compute_time / a2a_time_per_layer if a2a_time_per_layer > 0 else float('inf')

    # Scaling efficiency across GPU counts
    scaling = {}
    experts_per = {}
    for n_gpu in [1, 2, 4, 8, 16, 32, 64]:
        if n_gpu == 1:
            scaling[1] = 1.0
            experts_per[1] = cfg.num_experts
            continue

        ep = min(n_gpu, cfg.num_experts)
        experts_per[n_gpu] = cfg.num_experts // ep

        # Communication overhead scales with EP degree
        # More GPUs = more all-to-all participants = higher latency
        # Simplified model: T_comm ~ V / BW * log2(n_gpu) for realistic networks
        comm_overhead = a2a_time_per_layer * math.log2(n_gpu) * cfg.num_moe_layers
        compute_per_gpu = compute_time * cfg.num_moe_layers / ep

        total_time_1gpu = compute_time * cfg.num_moe_layers
        total_time_ngpu = compute_per_gpu + comm_overhead

        scaling[n_gpu] = min(1.0, total_time_1gpu / (total_time_ngpu * n_gpu)) if total_time_ngpu > 0 else 0

    # Recommended EP degree: highest scaling efficiency above 70%
    recommended_ep = 1
    for n, eff in sorted(scaling.items()):
        if eff >= 0.7:
            recommended_ep = n

    return CommunicationAnalysis(
        all2all_volume_per_layer_bytes=a2a_volume,
        all2all_time_per_layer_ms=a2a_time_per_layer * 1000,
        all2all_time_total_ms=a2a_total * 1000,
        compute_comm_ratio=comp_comm_ratio,
        min_gpus_for_memory=min_gpus,
        recommended_ep_degree=recommended_ep,
        scaling_efficiency=scaling,
        experts_per_gpu=experts_per,
    )


# =============================================================================
# Section 7: Latency & Throughput Estimation
# =============================================================================

@dataclass
class LatencyAnalysis:
    """Latency and throughput projections."""
    # Prefill
    time_to_first_token_ms: float       # TTFT
    prefill_tflops_achieved: float
    # Decode
    decode_ms_per_token: float
    decode_tokens_per_sec: float
    # Throughput
    max_throughput_tokens_per_sec: float  # at optimal batch size
    optimal_batch_size: int
    # Comparison
    dense_equivalent_ttft_ms: float      # same active params, dense model
    moe_speedup_factor: float
    # Cost
    cost_per_million_tokens: float       # at cloud $/hr


def compute_latency(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    flops: FLOPsBreakdown,
    gpu: GPUSpec,
    batch_size: int = 1,
    seq_len: int = 2048,
    quant_bits: int = 16,
    num_gpus: int = 1,
) -> LatencyAnalysis:
    """
    Estimate TTFT, decode latency, and throughput.

    MoE-specific: during decode (BS=1), we load only top-k expert weights,
    giving MoE better decode throughput per total parameter than dense.
    """
    bpw = quant_bits / 8
    N = batch_size * seq_len

    # --- Prefill (compute all tokens at once) ---
    total_prefill_flops = flops.total_flops_per_token * N
    # Use roofline-adjusted throughput (simplified)
    effective_tflops = min(
        gpu.fp16_tflops,
        (N * gpu.mem_bandwidth_gbps * 1e9) / (params.total_params * bpw * 1e12)
        * gpu.fp16_tflops
    )
    effective_tflops = min(effective_tflops, gpu.fp16_tflops) * 0.7  # practical efficiency
    effective_tflops = max(effective_tflops, gpu.fp16_tflops * 0.1)  # floor

    if num_gpus > 1:
        effective_tflops *= num_gpus * 0.8  # rough multi-GPU scaling

    ttft = total_prefill_flops / (effective_tflops * 1e12) if effective_tflops > 0 else float('inf')

    # --- Decode (one token at a time, memory-bandwidth bound) ---
    # Bytes to read per token during decode:
    # Shared weights + top_k expert weights per MoE layer + KV cache read
    shared_bytes = params.total_shared_params * bpw
    active_expert_bytes = cfg.num_moe_layers * cfg.num_experts_per_tok * params.expert_params_per_expert * bpw
    kv_read = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * cfg.dtype_bytes * seq_len

    total_bytes_per_decode = (shared_bytes + active_expert_bytes + kv_read) / num_gpus

    mem_bw = gpu.mem_bandwidth_gbps * 1e9  # bytes/sec
    decode_time = total_bytes_per_decode / mem_bw if mem_bw > 0 else float('inf')
    decode_tps = 1.0 / decode_time if decode_time > 0 else 0

    # --- Optimal batch size for max throughput ---
    # Find batch size where we transition to compute-bound
    optimal_bs = max(1, int(math.ceil(
        (params.active_params_per_token * bpw * gpu.fp16_tflops * 1e12)
        / (flops.total_flops_per_token * mem_bw)
    )))

    max_throughput = min(
        optimal_bs * decode_tps,
        gpu.fp16_tflops * 1e12 * num_gpus / flops.total_flops_per_token
    )

    # --- Dense equivalent comparison ---
    # Dense model with same active params
    dense_bytes_per_decode = params.active_params_per_token * bpw + kv_read
    dense_decode_time = dense_bytes_per_decode / (mem_bw * num_gpus)
    dense_ttft = total_prefill_flops / (effective_tflops * 1e12)  # similar compute

    speedup = dense_decode_time / decode_time if decode_time > 0 else 1.0

    # --- Cost ---
    cost_per_sec = gpu.cloud_cost_per_hour * num_gpus / 3600
    cost_per_m_tokens = (cost_per_sec / decode_tps * 1e6) if decode_tps > 0 else float('inf')

    return LatencyAnalysis(
        time_to_first_token_ms=ttft * 1000,
        prefill_tflops_achieved=effective_tflops,
        decode_ms_per_token=decode_time * 1000,
        decode_tokens_per_sec=decode_tps,
        max_throughput_tokens_per_sec=max_throughput,
        optimal_batch_size=optimal_bs,
        dense_equivalent_ttft_ms=dense_ttft * 1000,
        moe_speedup_factor=speedup,
        cost_per_million_tokens=cost_per_m_tokens,
    )


# =============================================================================
# Section 8: Expert Offloading Analysis (Consumer GPUs)
# =============================================================================

@dataclass
class OffloadingAnalysis:
    """Expert offloading strategy analysis for memory-constrained devices."""
    model_fits_in_memory: bool
    # Offloading strategy
    experts_in_vram: int
    experts_offloaded: int
    vram_for_shared_params_gb: float
    vram_for_cached_experts_gb: float
    vram_remaining_gb: float
    # Performance impact
    cache_hit_rate_needed: float   # to achieve target latency
    avg_expert_load_time_ms: float
    best_case_tokens_per_sec: float  # all cache hits
    worst_case_tokens_per_sec: float # all cache misses
    expected_tokens_per_sec: float   # with estimated hit rate


def compute_offloading(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    gpu: GPUSpec,
    quant_bits: int = 16,
    target_tokens_per_sec: float = 10.0,
) -> OffloadingAnalysis:
    """
    Analyze expert offloading strategy for consumer GPUs.

    Strategy: keep shared params + frequently used experts in VRAM,
    load remaining experts from CPU RAM via PCIe on demand.
    """
    bpw = quant_bits / 8

    shared_mem = params.total_shared_params * bpw
    expert_mem_each = params.expert_params_per_expert * bpw * cfg.num_moe_layers  # all layers' worth per expert "slot"
    # Actually, each expert per layer is independent
    single_expert_single_layer = params.expert_params_per_expert * bpw
    total_experts_total = cfg.num_experts * cfg.num_moe_layers

    total_model_mem = params.total_params * bpw
    fits = total_model_mem <= gpu.hbm_capacity_gb * 1e9 * 0.85

    # Available VRAM for expert caching
    vram_available = gpu.hbm_capacity_gb * 1e9 * 0.85  # 85% usable
    vram_for_shared = shared_mem
    vram_for_experts = max(0, vram_available - vram_for_shared)

    # How many expert-layer slots can we cache?
    experts_cached = int(vram_for_experts / single_expert_single_layer) if single_expert_single_layer > 0 else 0
    experts_cached = min(experts_cached, total_experts_total)
    experts_offloaded = total_experts_total - experts_cached

    vram_remaining = max(0, vram_available - vram_for_shared - experts_cached * single_expert_single_layer)

    # Expert load time via PCIe
    pcie_bw = gpu.pcie_bw_gbps * 1e9  # bytes/sec
    # For Apple Silicon, use memory bandwidth (unified memory, no offloading needed in same way)
    if "Apple" in gpu.architecture:
        load_time = 0  # unified memory
    else:
        load_time = single_expert_single_layer / pcie_bw if pcie_bw > 0 else float('inf')

    # Cache hit rate estimation (assuming Zipf distribution of expert usage)
    # With top-k routing, some experts are used much more than others
    cache_fraction = experts_cached / total_experts_total if total_experts_total > 0 else 1.0
    # Zipf model: hit_rate ≈ cache_fraction^0.7 (empirical for MoE routing)
    estimated_hit_rate = min(1.0, cache_fraction ** 0.7)

    # Throughput estimates
    # Base decode time (compute, ignoring weight loading)
    compute_per_token = params.active_params_per_token * bpw / (gpu.mem_bandwidth_gbps * 1e9)

    # Per-layer: need top_k experts, probability of miss per expert
    miss_rate = 1 - estimated_hit_rate
    expected_loads_per_layer = cfg.num_experts_per_tok * miss_rate
    expected_load_time = expected_loads_per_layer * cfg.num_moe_layers * load_time

    best_tps = 1.0 / compute_per_token if compute_per_token > 0 else 0
    worst_load = cfg.num_experts_per_tok * cfg.num_moe_layers * load_time
    worst_tps = 1.0 / (compute_per_token + worst_load) if (compute_per_token + worst_load) > 0 else 0
    expected_tps = 1.0 / (compute_per_token + expected_load_time) if (compute_per_token + expected_load_time) > 0 else 0

    # Required hit rate for target TPS
    target_total_time = 1.0 / target_tokens_per_sec
    available_load_budget = target_total_time - compute_per_token
    if available_load_budget <= 0:
        needed_hit_rate = 1.0  # can't hit target even with all cache hits
    elif load_time > 0:
        total_possible_loads = cfg.num_experts_per_tok * cfg.num_moe_layers
        max_misses = available_load_budget / load_time
        needed_hit_rate = max(0, 1 - max_misses / total_possible_loads)
    else:
        needed_hit_rate = 0

    return OffloadingAnalysis(
        model_fits_in_memory=fits,
        experts_in_vram=experts_cached,
        experts_offloaded=experts_offloaded,
        vram_for_shared_params_gb=vram_for_shared / 1e9,
        vram_for_cached_experts_gb=(experts_cached * single_expert_single_layer) / 1e9,
        vram_remaining_gb=vram_remaining / 1e9,
        cache_hit_rate_needed=needed_hit_rate,
        avg_expert_load_time_ms=load_time * 1000,
        best_case_tokens_per_sec=best_tps,
        worst_case_tokens_per_sec=worst_tps,
        expected_tokens_per_sec=expected_tps,
    )


# =============================================================================
# Section 9: Advanced / Standout Concepts
# =============================================================================

@dataclass
class AdvancedAnalysis:
    """Advanced theoretical analysis — the 'wow factor' metrics."""
    # ---- Information-Theoretic Expert Capacity ----
    routing_entropy_max: float            # Maximum possible routing entropy (uniform)
    routing_entropy_min_useful: float     # Minimum entropy for useful specialization
    expert_capacity_bits: float           # Information capacity per expert (bits)
    total_model_capacity_bits: float      # Total information capacity

    # ---- Granular Scaling Laws (Unified Scaling Laws for Routed Language Models) ----
    effective_parameter_count: float      # E_eff from scaling law perspective
    granularity_coefficient: float        # G = num_experts * top_k / total_experts
    scaling_law_multiplier: float         # How MoE scales vs dense of same compute

    # ---- Expert Redundancy & Dropout Tolerance ----
    theoretical_expert_dropout_tolerance: float  # fraction of experts removable before quality degrades
    min_experts_for_coverage: int                # minimum experts needed for >95% token coverage

    # ---- Token Dropping Analysis ----
    capacity_factor: float                # standard capacity factor
    expected_token_drop_rate: float       # theoretical token dropping probability
    tokens_dropped_per_batch: float

    # ---- MoE vs Dense Pareto Analysis ----
    moe_pareto_efficiency: float          # quality per FLOP relative to dense frontier
    equivalent_dense_model_size: float    # dense model with same quality (params)
    parameter_efficiency_ratio: float     # equivalent_dense / total_moe_params

    # ---- Routing Collapse Risk ----
    collapse_risk_score: float            # 0-1 probability of routing collapse
    load_imbalance_theoretical: float     # expected CV of expert loads (random routing)

    # ---- Communication-Computation Overlap Potential ----
    overlap_ratio: float                  # fraction of communication hideable behind compute
    effective_comm_overhead: float        # actual overhead after overlap


def compute_advanced_analysis(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    flops: FLOPsBreakdown,
    gpu: GPUSpec,
    batch_size: int = 1,
    seq_len: int = 2048,
) -> AdvancedAnalysis:
    """
    Compute advanced theoretical metrics that go beyond basic analysis.
    These are the metrics that make your tool stand out.
    """
    E = cfg.num_experts
    K = cfg.num_experts_per_tok
    N = batch_size * seq_len  # total tokens
    d = cfg.hidden_size

    # ================================================================
    # 1. Information-Theoretic Expert Capacity
    # ================================================================
    # Maximum routing entropy: uniform distribution over experts
    # H_max = log2(C(E, K)) ≈ K * log2(E/K) for large E
    # Simplified: H_max = log2(E) per token for top-1, adjusted for top-K
    routing_entropy_max = K * math.log2(E) if E > 1 else 0

    # Minimum useful entropy: experts should be somewhat specialized
    # If entropy is too low, few experts are used (collapse)
    # Rule of thumb: at least log2(K+1) bits of routing information
    routing_entropy_min = math.log2(K + 1) if K > 0 else 0

    # Expert capacity in bits: each expert can encode different transformations
    # Capacity ≈ params_per_expert * log2(precision_levels)
    precision_bits = cfg.dtype_bytes * 8
    # Using rate-distortion theory approximation
    expert_capacity_bits = params.expert_params_per_expert * math.log2(precision_bits)
    total_capacity = E * cfg.num_moe_layers * expert_capacity_bits

    # ================================================================
    # 2. Granular Scaling Laws
    # (Based on "Unified Scaling Laws for Routed Language Models", Clark et al.)
    # ================================================================
    # Effective parameters: N_eff = N_active * (N_total / N_active)^alpha
    # where alpha ≈ 0.2-0.3 for typical MoE models
    alpha = 0.25  # empirical scaling exponent
    N_active = params.active_params_per_token
    N_total = params.total_params

    if N_active > 0 and N_total > N_active:
        effective_params = N_active * (N_total / N_active) ** alpha
    else:
        effective_params = N_total

    # Granularity coefficient: measures how "fine-grained" the MoE is
    # G = 1 for dense, increases with more experts
    # Higher G = more specialized experts = better scaling
    granularity = E * K / max(E, 1)

    # Scaling multiplier: how much better MoE is than dense at same FLOPs
    # Empirically: ~1.5-3x for well-tuned MoE
    scaling_multiplier = (N_total / N_active) ** alpha if N_active > 0 else 1.0

    # ================================================================
    # 3. Expert Redundancy & Dropout Tolerance
    # ================================================================
    # With random routing, the probability of any expert being "essential"
    # follows a coupon collector-like problem
    # Tolerance: fraction of experts removable = 1 - K/E (rough bound)
    dropout_tolerance = max(0, 1 - (K / E) - 0.1) if E > K else 0

    # Minimum experts for 95% token coverage (probabilistic)
    # Using coupon collector: E(coverage) = E * (1 - (1-K/E)^n) where n = tokens
    # For 95% coverage: n ≈ E/K * ln(20)
    if K > 0:
        min_experts_95 = min(E, max(K, int(math.ceil(E / K * math.log(20)))))
    else:
        min_experts_95 = E

    # ================================================================
    # 4. Token Dropping Analysis
    # ================================================================
    # Capacity factor C: each expert can process C * (N * K / E) tokens
    # Standard C = 1.0-1.5
    capacity_factor = 1.25  # standard value

    # With uniform routing, token drop rate ≈ 0 if C >= 1
    # With skewed routing (realistic), drops follow:
    # P(drop) ≈ 1 - erf(C * sqrt(E / (2*K*N))) for large N
    tokens_per_expert_expected = N * K / E if E > 0 else N
    capacity_limit = capacity_factor * tokens_per_expert_expected

    # Using Poisson approximation for token drops
    # Lambda = tokens_per_expert_expected (Poisson parameter)
    # P(exceeds capacity) ≈ 1 - CDF(capacity_limit)
    if tokens_per_expert_expected > 0:
        # Normal approximation to Poisson for large lambda
        z_score = (capacity_limit - tokens_per_expert_expected) / math.sqrt(tokens_per_expert_expected)
        drop_rate = max(0, 0.5 * math.erfc(z_score / math.sqrt(2)))
    else:
        drop_rate = 0

    tokens_dropped = N * drop_rate

    # ================================================================
    # 5. MoE vs Dense Pareto Efficiency
    # ================================================================
    # Quality per FLOP: MoE achieves quality of a larger dense model
    # with the FLOPs of a smaller one
    # Pareto efficiency = effective_params / active_params
    pareto_eff = effective_params / N_active if N_active > 0 else 1.0

    # Equivalent dense model: dense model that achieves same quality
    equivalent_dense = effective_params
    param_efficiency = equivalent_dense / N_total if N_total > 0 else 1.0

    # ================================================================
    # 6. Routing Collapse Risk
    # ================================================================
    # Collapse risk increases with:
    # - High expert count (harder to balance)
    # - Low aux loss coefficient
    # - Small batch sizes (fewer tokens to distribute)
    # Heuristic score:
    expert_ratio = E / max(K, 1)
    batch_factor = min(1.0, 64 / max(N, 1))  # risk higher with small batches
    aux_factor = min(1.0, 0.01 / max(cfg.router_aux_loss_coef, 1e-6))

    collapse_risk = min(1.0, 0.1 * math.log(expert_ratio) * batch_factor * aux_factor)
    collapse_risk = max(0, collapse_risk)

    # Theoretical load imbalance (CV) with random routing
    # CV = sqrt(E / (N * K)) for uniform random routing
    load_imbalance_cv = math.sqrt(E / (N * K)) if N * K > 0 else float('inf')

    # ================================================================
    # 7. Communication-Computation Overlap Potential
    # ================================================================
    # Can we overlap all-to-all communication with expert computation?
    # Overlap ratio: how much of comm can be hidden behind compute
    if gpu.nvlink_bw_gbps > 0:
        effective_bw = gpu.nvlink_bw_gbps * 1e9 / 2
    else:
        effective_bw = gpu.pcie_bw_gbps * 1e9 / 2

    a2a_volume = 2 * N * d * cfg.dtype_bytes
    comm_time = a2a_volume / effective_bw if effective_bw > 0 else float('inf')

    expert_compute_time = (N * flops.expert_flops_per_layer) / (gpu.fp16_tflops * 1e12)

    # Overlap: can pipeline dispatch/compute/combine
    # Typically 50-80% of comm can overlap with compute
    if expert_compute_time > 0 and comm_time < float('inf'):
        overlap = min(1.0, expert_compute_time / comm_time)
    else:
        overlap = 0

    effective_overhead = comm_time * (1 - overlap) if comm_time < float('inf') else 0

    return AdvancedAnalysis(
        routing_entropy_max=routing_entropy_max,
        routing_entropy_min_useful=routing_entropy_min,
        expert_capacity_bits=expert_capacity_bits,
        total_model_capacity_bits=total_capacity,
        effective_parameter_count=effective_params,
        granularity_coefficient=granularity,
        scaling_law_multiplier=scaling_multiplier,
        theoretical_expert_dropout_tolerance=dropout_tolerance,
        min_experts_for_coverage=min_experts_95,
        capacity_factor=capacity_factor,
        expected_token_drop_rate=drop_rate,
        tokens_dropped_per_batch=tokens_dropped,
        moe_pareto_efficiency=pareto_eff,
        equivalent_dense_model_size=equivalent_dense,
        parameter_efficiency_ratio=param_efficiency,
        collapse_risk_score=collapse_risk,
        load_imbalance_theoretical=load_imbalance_cv,
        overlap_ratio=overlap,
        effective_comm_overhead=effective_overhead * 1000,  # ms
    )


# =============================================================================
# Section 10: Quantization Impact Analysis
# =============================================================================

@dataclass
class QuantizationAnalysis:
    """Impact of quantization on MoE models."""
    quant_bits: int
    memory_savings_ratio: float
    model_size_gb: float
    # Per-component sensitivity (theoretical)
    attention_sensitivity: str     # "low" | "medium" | "high"
    expert_sensitivity: str
    router_sensitivity: str
    embedding_sensitivity: str
    # Recommendations
    recommended_strategy: str
    mixed_precision_breakdown: dict  # component -> recommended bits


def compute_quantization_analysis(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    quant_bits_options: list[int] = None,
) -> list[QuantizationAnalysis]:
    """
    Analyze quantization impact across bit widths.

    MoE-specific insight: experts can often tolerate more aggressive quantization
    than attention layers, because each expert sees fewer tokens (lower utilization
    = less sensitivity to precision loss). Router weights are ALWAYS sensitive.
    """
    if quant_bits_options is None:
        quant_bits_options = [16, 8, 4, 3, 2]

    results = []
    baseline_mem = params.total_params * 2  # BF16 baseline

    for bits in quant_bits_options:
        bpw = bits / 8
        model_mem = params.total_params * bpw

        # Sensitivity heuristics (backed by literature):
        # - Router: ALWAYS keep FP16+ (routing decisions are critical)
        # - Attention: moderate sensitivity (QKV precision matters)
        # - Experts: lower sensitivity (redundancy helps)
        # - Embeddings: moderate sensitivity

        if bits >= 8:
            attn_sens = "low"
            expert_sens = "low"
            router_sens = "low"
            embed_sens = "low"
            strategy = f"Uniform {bits}-bit quantization. Safe for all components."
        elif bits >= 4:
            attn_sens = "medium"
            expert_sens = "low"
            router_sens = "high"
            embed_sens = "medium"
            strategy = (
                f"Mixed precision recommended: {bits}-bit for experts, "
                f"8-bit for attention, FP16 for router weights."
            )
        elif bits >= 3:
            attn_sens = "high"
            expert_sens = "medium"
            router_sens = "high"
            embed_sens = "high"
            strategy = (
                f"Aggressive quantization. Use {bits}-bit only for experts, "
                f"keep attention at 4-bit+, router at FP16. Expect quality degradation."
            )
        else:
            attn_sens = "high"
            expert_sens = "high"
            router_sens = "high"
            embed_sens = "high"
            strategy = (
                f"Extreme quantization ({bits}-bit). Significant quality loss expected. "
                f"Only viable with quantization-aware training (QAT)."
            )

        mixed_precision = {
            "router": 16,  # always FP16
            "attention": max(bits, 8) if bits < 8 else bits,
            "experts": bits,
            "embeddings": max(bits, 4) if bits < 4 else bits,
            "layernorm": 16,  # always FP16
        }

        # Actual memory with mixed precision
        mixed_mem = (
            params.total_expert_params * (mixed_precision["experts"] / 8)
            + params.total_attention_params * (mixed_precision["attention"] / 8)
            + params.embedding_params * (mixed_precision["embeddings"] / 8)
            + (cfg.num_moe_layers * cfg.hidden_size * cfg.num_experts) * (mixed_precision["router"] / 8)
            + (cfg.num_hidden_layers * 2 * cfg.hidden_size) * (mixed_precision["layernorm"] / 8)
        )

        results.append(QuantizationAnalysis(
            quant_bits=bits,
            memory_savings_ratio=model_mem / baseline_mem,
            model_size_gb=mixed_mem / 1e9,
            attention_sensitivity=attn_sens,
            expert_sensitivity=expert_sens,
            router_sensitivity=router_sens,
            embedding_sensitivity=embed_sens,
            recommended_strategy=strategy,
            mixed_precision_breakdown=mixed_precision,
        ))

    return results


# =============================================================================
# Section 11: Training Cost Estimation
# =============================================================================

@dataclass
class TrainingCostEstimate:
    """Estimated training cost and time."""
    total_train_flops: float           # total FLOPs for training run
    gpu_hours: float                   # GPU-hours needed
    wall_clock_days: float             # calendar days (with given GPU count)
    estimated_cost_usd: float          # $$
    tokens_per_second_per_gpu: float
    mfu: float                         # Model FLOPs Utilization
    # MoE-specific
    aux_loss_overhead_percent: float   # overhead from load-balancing computation
    communication_overhead_percent: float


def estimate_training_cost(
    cfg: MoEConfig,
    params: ParameterBreakdown,
    flops: FLOPsBreakdown,
    gpu: GPUSpec,
    num_gpus: int = 8,
    training_tokens: float = 1e12,  # 1T tokens
    mfu_estimate: float = 0.35,     # typical MFU for MoE
) -> TrainingCostEstimate:
    """
    Estimate training cost using Chinchilla-aware scaling.

    MoE MFU is typically lower than dense models (30-40% vs 45-55%)
    due to load imbalance and communication overhead.
    """
    # Total training FLOPs ≈ 6 * N_active * T (6 accounts for fwd + bwd + optimizer)
    # For MoE, use active params not total params
    total_flops = 6 * params.active_params_per_token * training_tokens

    # Effective throughput
    peak_cluster_tflops = gpu.fp16_tflops * num_gpus
    effective_tflops = peak_cluster_tflops * mfu_estimate

    # Time
    total_seconds = total_flops / (effective_tflops * 1e12)
    gpu_hours = total_seconds * num_gpus / 3600
    wall_days = total_seconds / 86400

    # Cost
    cost = gpu_hours * gpu.cloud_cost_per_hour

    # Tokens per second per GPU
    tps_per_gpu = training_tokens / total_seconds / num_gpus if total_seconds > 0 else 0

    # Aux loss overhead: typically 1-3% of total compute
    aux_overhead = 2.0  # percent

    # Communication overhead: depends on EP degree
    # Rough estimate: 5-15% for 8-GPU EP
    comm_overhead = min(20, 5 * math.log2(max(num_gpus, 1)))

    return TrainingCostEstimate(
        total_train_flops=total_flops,
        gpu_hours=gpu_hours,
        wall_clock_days=wall_days,
        estimated_cost_usd=cost,
        tokens_per_second_per_gpu=tps_per_gpu,
        mfu=mfu_estimate,
        aux_loss_overhead_percent=aux_overhead,
        communication_overhead_percent=comm_overhead,
    )


# =============================================================================
# Section 12: Main Analyzer Class
# =============================================================================

def _fmt_num(n: float, suffix: str = "") -> str:
    """Format large numbers readably."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T{suffix}"
    if n >= 1e9:
        return f"{n/1e9:.2f}B{suffix}"
    if n >= 1e6:
        return f"{n/1e6:.2f}M{suffix}"
    if n >= 1e3:
        return f"{n/1e3:.2f}K{suffix}"
    return f"{n:.2f}{suffix}"


def _fmt_bytes(b: float) -> str:
    """Format bytes as GB/TB."""
    if b >= 1e12:
        return f"{b/1e12:.2f} TB"
    return f"{b/1e9:.2f} GB"


class MoEAnalyzer:
    """
    Main entry point for MoE theoretical analysis.

    Usage:
        analyzer = MoEAnalyzer("mistralai/Mixtral-8x7B-v0.1")
        report = analyzer.full_analysis(gpu_names=["H100_SXM", "RTX_4090"])
        analyzer.print_report(report)
    """

    def __init__(self, model_name: str, custom_config: dict = None):
        """
        Initialize analyzer with a HuggingFace model name or custom config dict.

        Args:
            model_name: HuggingFace model identifier (e.g., "mistralai/Mixtral-8x7B-v0.1")
            custom_config: Optional dict to override/supplement fetched config
        """
        self.model_name = model_name
        print(f"Fetching config for '{model_name}'...")
        raw = fetch_hf_config(model_name)
        if custom_config:
            raw.update(custom_config)
        self.raw_config = raw
        self.cfg = parse_moe_config(model_name, raw)
        self.params = compute_parameter_breakdown(self.cfg)
        print(f"Config loaded: {_fmt_num(self.params.total_params)} total params, "
              f"{_fmt_num(self.params.active_params_per_token)} active per token, "
              f"{self.cfg.num_experts} experts, top-{self.cfg.num_experts_per_tok}")

    def analyze_for_gpu(
        self,
        gpu_name: str,
        batch_size: int = 1,
        seq_len: int = 2048,
        quant_bits: int = 16,
        num_gpus: int = 1,
    ) -> dict:
        """Run full analysis for a specific GPU configuration."""
        gpu = GPU_DATABASE[gpu_name]
        f = compute_flops(self.cfg, self.params, batch_size, seq_len)
        return {
            "gpu": gpu,
            "flops": f,
            "memory": compute_memory_breakdown(self.cfg, self.params, batch_size, seq_len, quant_bits),
            "memory_training": compute_memory_breakdown(self.cfg, self.params, batch_size, seq_len, quant_bits, training=True),
            "roofline": compute_roofline(self.cfg, self.params, f, gpu, batch_size, seq_len),
            "communication": compute_communication(self.cfg, self.params, f, gpu, batch_size, seq_len, quant_bits),
            "latency": compute_latency(self.cfg, self.params, f, gpu, batch_size, seq_len, quant_bits, num_gpus),
            "offloading": compute_offloading(self.cfg, self.params, gpu, quant_bits),
            "advanced": compute_advanced_analysis(self.cfg, self.params, f, gpu, batch_size, seq_len),
        }

    def full_analysis(
        self,
        gpu_names: list[str] = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        quant_bits: int = 16,
    ) -> dict:
        """Run analysis across multiple GPUs."""
        if gpu_names is None:
            gpu_names = ["A100_80GB", "H100_SXM", "RTX_4090"]

        report = {
            "model": self.cfg,
            "parameters": self.params,
            "quantization": compute_quantization_analysis(self.cfg, self.params),
            "gpu_analyses": {},
        }

        for name in gpu_names:
            if name not in GPU_DATABASE:
                print(f"Warning: GPU '{name}' not in database, skipping.")
                continue
            report["gpu_analyses"][name] = self.analyze_for_gpu(
                name, batch_size, seq_len, quant_bits
            )

        return report

    def estimate_training(
        self,
        gpu_name: str = "H100_SXM",
        num_gpus: int = 64,
        training_tokens: float = 1e12,
    ) -> TrainingCostEstimate:
        """Estimate training cost."""
        gpu = GPU_DATABASE[gpu_name]
        f = compute_flops(self.cfg, self.params)
        return estimate_training_cost(self.cfg, self.params, f, gpu, num_gpus, training_tokens)

    @staticmethod
    def list_gpus() -> list[str]:
        """List all available GPU names."""
        return list(GPU_DATABASE.keys())

    # ==== Pretty Printing ====

    def print_report(self, report: dict):
        """Print a comprehensive human-readable report."""
        cfg = report["model"]
        p = report["parameters"]

        print("\n" + "=" * 80)
        print(f"  MoE THEORETICAL ANALYSIS: {cfg.model_name}")
        print("=" * 80)

        # Architecture
        print(f"\n{'─' * 40}")
        print("  ARCHITECTURE")
        print(f"{'─' * 40}")
        print(f"  Hidden size:          {cfg.hidden_size}")
        print(f"  Layers:               {cfg.num_hidden_layers} ({cfg.num_moe_layers} MoE + {cfg.num_dense_layers} dense)")
        print(f"  Attention heads:      {cfg.num_attention_heads} (KV: {cfg.num_key_value_heads})")
        print(f"  Experts:              {cfg.num_experts} total, top-{cfg.num_experts_per_tok} selected")
        if cfg.num_shared_experts > 0:
            print(f"  Shared experts:       {cfg.num_shared_experts}")
        print(f"  Expert FFN dim:       {cfg.expert_intermediate_size}")
        print(f"  Vocab size:           {cfg.vocab_size}")
        print(f"  Max seq length:       {cfg.max_position_embeddings}")

        # Parameters
        print(f"\n{'─' * 40}")
        print("  PARAMETERS")
        print(f"{'─' * 40}")
        print(f"  Total params:         {_fmt_num(p.total_params)}")
        print(f"  Active per token:     {_fmt_num(p.active_params_per_token)}")
        print(f"  Sparsity ratio:       {p.active_params_per_token/p.total_params:.1%}")
        print(f"  Expert params (all):  {_fmt_num(p.total_expert_params)}")
        print(f"  Shared params:        {_fmt_num(p.total_shared_params)}")
        print(f"  Per expert:           {_fmt_num(p.expert_params_per_expert)}")

        # Quantization
        print(f"\n{'─' * 40}")
        print("  QUANTIZATION OPTIONS")
        print(f"{'─' * 40}")
        for q in report["quantization"]:
            print(f"  {q.quant_bits:2d}-bit: {q.model_size_gb:7.2f} GB  ({q.memory_savings_ratio:.0%} of FP16)  "
                  f"| Expert sens: {q.expert_sensitivity:<6s} | {q.recommended_strategy[:60]}")

        # Per-GPU analysis
        for gpu_name, analysis in report["gpu_analyses"].items():
            gpu = analysis["gpu"]
            mem = analysis["memory"]
            f = analysis["flops"]
            roof = analysis["roofline"]
            comm = analysis["communication"]
            lat = analysis["latency"]
            off = analysis["offloading"]
            adv = analysis["advanced"]

            print(f"\n{'═' * 80}")
            print(f"  GPU: {gpu.name}")
            print(f"  {gpu.fp16_tflops} BF16 TFLOPS | {gpu.hbm_capacity_gb} GB HBM | "
                  f"{gpu.mem_bandwidth_gbps} GB/s BW | ${gpu.cloud_cost_per_hour}/hr")
            print(f"{'═' * 80}")

            # Memory
            print(f"\n  Memory Analysis:")
            fits = "✅ YES" if mem.total_inference <= gpu.hbm_capacity_gb * 1e9 else "❌ NO"
            print(f"    Fits in VRAM:       {fits}")
            print(f"    Model weights:      {_fmt_bytes(mem.model_weights)}")
            print(f"    KV cache:           {_fmt_bytes(mem.kv_cache)}")
            print(f"    Activations:        {_fmt_bytes(mem.activations)}")
            print(f"    Total inference:    {_fmt_bytes(mem.total_inference)}")
            train_mem = analysis["memory_training"]
            print(f"    Total training:     {_fmt_bytes(train_mem.total_training)}")

            # Compute
            print(f"\n  Compute Analysis:")
            print(f"    FLOPs/token:        {_fmt_num(f.total_flops_per_token, ' FLOPs')}")
            print(f"    Dense equivalent:   {_fmt_num(f.equivalent_dense_flops, ' FLOPs')}")
            print(f"    FLOPs savings:      {f.flops_savings_ratio:.1%} of dense")

            # Roofline
            print(f"\n  Roofline Analysis:")
            print(f"    Bottleneck:         {'🔴 ' + roof.bottleneck.upper()}")
            print(f"    Arithmetic intens:  {roof.operational_intensity:.1f} FLOP/Byte")
            print(f"    Ridge point:        {roof.ridge_point:.1f} FLOP/Byte")
            print(f"    Achievable TFLOPS:  {roof.achievable_tflops:.1f} / {roof.peak_compute:.1f} ({roof.compute_utilization:.1%})")
            print(f"    BS to saturate:     {roof.batch_size_to_saturate}")

            # Communication
            print(f"\n  Communication (Expert Parallelism):")
            print(f"    Min GPUs (memory):  {comm.min_gpus_for_memory}")
            print(f"    Recommended EP:     {comm.recommended_ep_degree} GPUs")
            print(f"    All2All/layer:      {comm.all2all_time_per_layer_ms:.3f} ms")
            print(f"    Compute/Comm ratio: {comm.compute_comm_ratio:.2f}x")
            print(f"    Scaling efficiency:")
            for n, eff in sorted(comm.scaling_efficiency.items()):
                bar = "█" * int(eff * 20)
                print(f"      {n:3d} GPUs: {eff:5.1%} {bar}")

            # Latency
            print(f"\n  Latency & Throughput:")
            print(f"    TTFT:               {lat.time_to_first_token_ms:.1f} ms")
            print(f"    Decode:             {lat.decode_ms_per_token:.2f} ms/token ({lat.decode_tokens_per_sec:.1f} tok/s)")
            print(f"    Max throughput:     {lat.max_throughput_tokens_per_sec:.0f} tok/s (BS={lat.optimal_batch_size})")
            print(f"    MoE speedup:        {lat.moe_speedup_factor:.2f}x vs dense")
            if gpu.cloud_cost_per_hour > 0:
                print(f"    Cost:               ${lat.cost_per_million_tokens:.2f} / M tokens")

            # Offloading
            if not off.model_fits_in_memory:
                print(f"\n  Expert Offloading (model doesn't fit):")
                print(f"    Experts in VRAM:    {off.experts_in_vram} / {off.experts_in_vram + off.experts_offloaded}")
                print(f"    VRAM for shared:    {off.vram_for_shared_params_gb:.2f} GB")
                print(f"    VRAM for experts:   {off.vram_for_cached_experts_gb:.2f} GB")
                print(f"    Expert load time:   {off.avg_expert_load_time_ms:.2f} ms (via PCIe)")
                print(f"    Expected tok/s:     {off.expected_tokens_per_sec:.1f}")
                print(f"    Hit rate for 10t/s: {off.cache_hit_rate_needed:.1%}")

            # Advanced
            print(f"\n  Advanced Analysis:")
            print(f"    Effective params (scaling law): {_fmt_num(adv.effective_parameter_count)}")
            print(f"    Scaling multiplier:    {adv.scaling_law_multiplier:.2f}x vs dense")
            print(f"    Granularity coeff:     {adv.granularity_coefficient:.2f}")
            print(f"    Pareto efficiency:     {adv.moe_pareto_efficiency:.2f}")
            print(f"    Routing entropy max:   {adv.routing_entropy_max:.2f} bits")
            print(f"    Expert dropout tol:    {adv.theoretical_expert_dropout_tolerance:.1%}")
            print(f"    Token drop rate:       {adv.expected_token_drop_rate:.4%}")
            print(f"    Collapse risk:         {adv.collapse_risk_score:.2f}")
            print(f"    Comm overlap:          {adv.overlap_ratio:.1%}")
            print(f"    Expert capacity:       {_fmt_num(adv.total_model_capacity_bits)} bits")

        print(f"\n{'=' * 80}")
        print("  Analysis complete.")
        print(f"{'=' * 80}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mixtral-8x7B-v0.1"
    gpus = sys.argv[2].split(",") if len(sys.argv) > 2 else ["A100_80GB", "H100_SXM", "RTX_4090", "M4_Max"]

    print(f"\nAvailable GPUs: {MoEAnalyzer.list_gpus()}\n")

    analyzer = MoEAnalyzer(model)
    report = analyzer.full_analysis(gpu_names=gpus, batch_size=1, seq_len=2048)
    analyzer.print_report(report)

    # Training cost estimate
    print("\n  TRAINING COST ESTIMATE (1T tokens on 64x H100):")
    tc = analyzer.esti
    .mate_training("H100_SXM", num_gpus=64, training_tokens=1e12)
    print(f"    Total FLOPs:    {_fmt_num(tc.total_train_flops)}")
    print(f"    GPU-hours:      {tc.gpu_hours:,.0f}")
    print(f"    Wall time:      {tc.wall_clock_days:.1f} days")
    print(f"    Est. cost:      ${tc.estimated_cost_usd:,.0f}")
    print(f"    Tokens/s/GPU:   {tc.tokens_per_second_per_gpu:,.0f}")
    print(f"    MFU:            {tc.mfu:.0%}")