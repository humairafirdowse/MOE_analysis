import {
  DerivedOverviewMetrics,
  InferenceConfig,
  ModelArchitectureConfig,
  MoeConfig,
  TrainingConfig,
  PrecisionTraining,
  PrecisionInference
} from "../state/useConfigStore";
import { GpuSpec } from "./gpus";

// ---- Shared low-level helpers ----

export function bytesPerParamTraining(precision: PrecisionTraining): number {
  if (precision === "fp32") return 4;
  if (precision === "fp8") return 1;
  return 2; // bf16
}

export function bytesPerParamInference(precision: PrecisionInference): number {
  switch (precision) {
    case "fp16":
    case "bf16":
      return 2;
    case "fp8":
    case "int8":
      return 1;
    case "int4":
      return 0.5;
    default:
      return 2;
  }
}

export function attentionFlopsPerToken(
  model: ModelArchitectureConfig,
  seqLen: number
): number {
  const dModel = model.dModel;
  const L = model.layers;
  const H = model.nHeads;
  const S = seqLen;

  const hasMLAParams =
    model.useMLA &&
    (model.mlaQLoraRank ?? 0) > 0 &&
    (model.mlaKvLoraRank ?? 0) > 0 &&
    (model.mlaQkRopeHeadDim ?? 0) > 0 &&
    (model.mlaQkNopeHeadDim ?? 0) > 0 &&
    (model.mlaVHeadDim ?? 0) > 0;

  if (hasMLAParams) {
    const qLora = model.mlaQLoraRank!;
    const kvLora = model.mlaKvLoraRank!;
    const qkRope = model.mlaQkRopeHeadDim!;
    const qkNope = model.mlaQkNopeHeadDim!;
    const vHead = model.mlaVHeadDim!;
    const qHeadDim = qkNope + qkRope;

    // MLA: q_a, q_b, kv_a, kv_b, o_proj (2× for matmul FLOPs).
    const FqA = 2 * dModel * qLora;
    const FqB = 2 * qLora * (H * qHeadDim);
    const FkvA = 2 * dModel * (kvLora + qkRope);
    const FkvB = 2 * kvLora * (H * (qkNope + vHead));
    const Fo = 2 * (H * vHead) * dModel;
    const Fproj = FqA + FqB + FkvA + FkvB + Fo;

    // Attention scores (Q @ K^T) and context (attn @ V).
    const Fscore = 2 * S * H * qHeadDim;
    const Fcontext = 2 * S * H * vHead;
    const FattnCompute = Fscore + Fcontext;

    const perLayer = Fproj + FattnCompute;
    return L * perLayer;
  }

  const Hkv = model.nKvHeads;
  const dh = model.dHead;

  // Projection FLOPs (Q, K, V, O) with GQA.
  const Fq = 2 * dModel * (H * dh);
  const Fk = 2 * dModel * (Hkv * dh);
  const Fv = 2 * dModel * (Hkv * dh);
  const Fo = 2 * (H * dh) * dModel;
  const Fproj = Fq + Fk + Fv + Fo;

  // Attention scores and context (per layer, per token).
  const Fscore = 2 * S * H * dh;
  const Fcontext = 2 * S * H * dh;
  const FattnCompute = Fscore + Fcontext;

  const perLayer = Fproj + FattnCompute;
  return L * perLayer;
}

export function moeFlopsComponentsPerToken(
  model: ModelArchitectureConfig,
  moe: MoeConfig
): {
  moeFfn: number;
  sharedFfn: number;
  gating: number;
} {
  const d = model.dModel;
  const L_moe = model.layersMoe ?? model.layers;

  // Per active expert (SwiGLU): 6 × d_model × d_ff
  const perExpert = 6 * d * moe.dFf;
  const perShared = 6 * d * moe.dSharedFf;

  const moeFfn = L_moe * moe.topK * perExpert;
  const sharedFfn = L_moe * moe.numSharedExperts * perShared;

  // Gating: 2 × d_model × E per MoE layer
  const gating = L_moe * 2 * d * moe.numExperts;

  return { moeFfn, sharedFfn, gating };
}

export function forwardFlopsPerToken(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  seqLen: number
): {
  attention: number;
  moeFfn: number;
  sharedFfn: number;
  gating: number;
  denseFfn: number;
  residualMlpFfn: number;
  total: number;
} {
  const attention = attentionFlopsPerToken(model, seqLen);
  const { moeFfn, sharedFfn, gating } = moeFlopsComponentsPerToken(model, moe);
  const numDense = model.firstKDenseReplace ?? 0;
  const denseDff = model.denseIntermediateSize ?? moe.dFf;
  const denseFfn =
    numDense > 0 ? numDense * 6 * model.dModel * denseDff : 0;

  const L_moe = model.layersMoe ?? Math.max(0, model.layers - numDense);
  const residualMLPDff =
    model.residualMLPIntermediateSize ?? model.denseIntermediateSize ?? moe.dFf;
  const residualMlpFfn =
    model.useResidualMLP === true
      ? L_moe * 6 * model.dModel * residualMLPDff
      : 0;

  const total = attention + moeFfn + sharedFfn + gating + denseFfn + residualMlpFfn;
  return { attention, moeFfn, sharedFfn, gating, denseFfn, residualMlpFfn, total };
}

export function denseEquivalentForwardFlopsPerToken(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  seqLen: number
): number {
  const d = model.dModel;
  const numDense = model.firstKDenseReplace ?? 0;
  const L_moe = model.layersMoe ?? Math.max(0, model.layers - numDense);
  const denseDff = model.denseIntermediateSize ?? moe.dFf;

  const attention = attentionFlopsPerToken(model, seqLen);

  // Dense equivalent: all experts fire (K_dense = E) in MoE layers.
  const perExpertDense = 6 * d * moe.dFf;
  const moeDenseFfn = L_moe * moe.numExperts * perExpertDense;
  const shared = L_moe * moe.numSharedExperts * (6 * d * moe.dSharedFf);
  const denseLayersFfn =
    numDense > 0 ? numDense * 6 * d * denseDff : 0;

  const residualMLPDff =
    model.residualMLPIntermediateSize ?? model.denseIntermediateSize ?? moe.dFf;
  const residualMlpFfn =
    model.useResidualMLP === true
      ? L_moe * 6 * d * residualMLPDff
      : 0;

  return attention + moeDenseFfn + shared + denseLayersFfn + residualMlpFfn;
}

// ---- Training compute metrics ----

export interface TrainingComputeMetrics {
  attentionFlopsPerToken: number;
  moeFfnFlopsPerToken: number;
  sharedFfnFlopsPerToken: number;
  gatingFlopsPerToken: number;
  denseFfnFlopsPerToken: number;
  residualMlpFfnFlopsPerToken: number;
  forwardFlopsPerToken: number;
  backwardFlopsPerToken: number;
  totalFlopsPerToken: number;
  totalTrainingFlops: number;
  denseEquivalentFlopsPerToken: number;
  totalGpus: number;
  gpuHoursApprox: number;
  trainingCostUSD: number;
}

export function computeTrainingComputeMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  training: TrainingConfig,
  trainingGpu: GpuSpec,
  mfu: number
): TrainingComputeMetrics {
  const S = model.maxSeqLen;
  const { attention, moeFfn, sharedFfn, gating, denseFfn, residualMlpFfn, total } =
    forwardFlopsPerToken(model, moe, S);
  const denseEq = denseEquivalentForwardFlopsPerToken(model, moe, S);

  const forward = total;
  const backward = 2 * forward;
  const perTokenTotal = forward + backward; // ~3× forward (fwd + bwd)

  const totalTrainingFlops = perTokenTotal * training.totalTrainingTokens;

  const totalGpus = Math.max(
    1,
    training.tp * training.ep * training.pp * training.dp
  );

  const precision = training.precision;
  const peakPerGpuTflops =
    precision === "fp32"
      ? trainingGpu.fp32Tflops
      : precision === "fp8"
        ? (trainingGpu.fp8TrainingTflops ??
           trainingGpu.fp8Tflops ??
           trainingGpu.fp16Tflops * 2)
        : trainingGpu.fp16Tflops;

  const effectivePerGpuTflops = peakPerGpuTflops * mfu;

  const timeSecondsCluster =
    totalTrainingFlops / (effectivePerGpuTflops * totalGpus * 1e12);
  const gpuHours = (timeSecondsCluster * totalGpus) / 3600;
  const trainingCostUSD = gpuHours * trainingGpu.costPerHourUSD;

  return {
    attentionFlopsPerToken: attention,
    moeFfnFlopsPerToken: moeFfn,
    sharedFfnFlopsPerToken: sharedFfn,
    gatingFlopsPerToken: gating,
    denseFfnFlopsPerToken: denseFfn,
    residualMlpFfnFlopsPerToken: residualMlpFfn,
    forwardFlopsPerToken: forward,
    backwardFlopsPerToken: backward,
    totalFlopsPerToken: perTokenTotal,
    totalTrainingFlops,
    denseEquivalentFlopsPerToken: denseEq,
    totalGpus,
    gpuHoursApprox: gpuHours,
    trainingCostUSD
  };
}

// ---- Training memory metrics ----

export interface TrainingMemoryMetrics {
  paramsBytes: number;
  optimizerBytes: number;
  gradientBytes: number;
  activationsBytes: number;
  peakBytes: number;
  perGpuBytes: number;
}

export function computeTrainingMemoryMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  training: TrainingConfig
): TrainingMemoryMetrics {
  const bytesParam = bytesPerParamTraining(training.precision);
  const totalParams = overview.totalParams;

  const paramsBytes = totalParams * bytesParam;

  // ── Optimizer state bytes ───────────────────────────────────────────────────
  // Adam byte budget per parameter:
  //   pure FP32 training        → m_fp32(4) + v_fp32(4)          =  8 B/param
  //   mixed-precision, fp32 m/v → master_fp32(4)+m_fp32(4)+v_fp32(4) = 12 B/param
  //   mixed-precision, bf16 m/v → master_fp32(4)+m_bf16(2)+v_bf16(2) =  8 B/param
  // DeepSeek-V3 Technical Report §3.3 explicitly stores m and v in BF16.
  let optimizerBytesPerParam: number;
  if (training.optimizer === "adam") {
    if (training.precision === "fp32") {
      optimizerBytesPerParam = 8; // m_fp32 + v_fp32, no master copy needed
    } else if (training.adamMomentPrecision === "bf16") {
      optimizerBytesPerParam = 8; // master_fp32 + m_bf16 + v_bf16
    } else {
      optimizerBytesPerParam = 12; // master_fp32 + m_fp32 + v_fp32 (classic)
    }
  } else {
    optimizerBytesPerParam = 4; // adafactor: single FP32 accumulator
  }
  const optimizerBytes = totalParams * optimizerBytesPerParam;

  const gradientBytes =
    totalParams * (training.gradPrecision === "bf16" ? 2 : 4);

  // ── Activation memory ───────────────────────────────────────────────────────
  // Activations are sized for one pipeline micro-batch, not the full global
  // batch.  Using globalBatchTokens would overcount by the number of gradient-
  // accumulation steps (e.g. ×1024 for DeepSeek-V3).
  //
  // EP sharding of stored activations:
  //   The residual stream (dModel-dimensional per token) always lives on the
  //   HOME GPU — the GPU that received the token from the previous pipeline
  //   stage.  It is never touched by EP dispatch, so it does NOT shard by EP.
  //
  //   Expert FFN intermediates (gate/up projections) live on the EXPERT GPU
  //   after the All-to-All dispatch.  Whether they are stored depends on the
  //   checkpointing mode:
  //     • "none"      → stored on expert GPU → EP-sharded  (÷ TP×EP×PP)
  //     • "sqrt"      → stored on expert GPU → EP-sharded  (÷ TP×EP×PP)
  //     • "selective" → recomputed via a second All-to-All → NOT stored at all
  //
  //   Attention activations (if not using FlashAttention) live on the home
  //   GPU and are therefore NOT EP-sharded.  With selective checkpointing
  //   attention is the component being recomputed, so it is also not stored.

  const seqLen = model.maxSeqLen;
  const d = model.dModel;
  const nLayers = model.layers;
  const microBatch = Math.max(1, training.microBatchSeqCount ?? 1);

  // Per-layer activation sizes (micro-batch scope)
  const attnActPerLayer =
    training.useFlashAttention
      ? 0
      : 2 * microBatch * model.nHeads * seqLen * seqLen * bytesParam;

  const residualActPerLayer = microBatch * seqLen * d * bytesParam * 2;

  // Expert FFN intermediates per layer — only stored without selective checkpointing.
  // Shape approximation: topK dispatched tokens × dFf × bytes, averaged over EP ranks.
  const expertFfnActPerLayer =
    microBatch * seqLen * moe.dFf * Math.max(moe.topK, 1) * bytesParam;

  // expertActivationsBytes  → lives on EXPERT GPU  → will be ÷ denomEP
  // homeActivationsBytes    → lives on HOME  GPU   → will be ÷ denomNoEP
  let expertActivationsBytes: number;
  let homeActivationsBytes: number;

  switch (training.activationCheckpointing) {
    case "none":
      // Everything stored.
      expertActivationsBytes = expertFfnActPerLayer * nLayers;
      homeActivationsBytes   = (attnActPerLayer + residualActPerLayer) * nLayers;
      break;
    case "sqrt": {
      const sqrtLayers = 2 * Math.ceil(Math.sqrt(nLayers || 1));
      expertActivationsBytes = expertFfnActPerLayer * sqrtLayers;
      homeActivationsBytes   = (attnActPerLayer + residualActPerLayer) * sqrtLayers;
      break;
    }
    case "selective":
    default:
      // Expert FFN recomputed via a second All-to-All → zero storage cost.
      // Attention recomputed (selective skips it).
      // Only the residual stream is retained on the home GPU.
      expertActivationsBytes = 0;
      homeActivationsBytes   = residualActPerLayer * nLayers;
      break;
  }

  const activationsBytes = expertActivationsBytes + homeActivationsBytes;

  const peakBytes = paramsBytes + optimizerBytes + gradientBytes + activationsBytes;

  // ── Per-GPU sharding ────────────────────────────────────────────────────────
  //
  // Routed expert weights (~655 B for DeepSeek-V3) are owned by specific EP
  // ranks, so they divide by TP × EP × PP.
  //
  // All other weights — attention (MLA), shared experts, dense FFN layers,
  // embeddings, output head, norms, gating — are REPLICATED across every EP
  // rank because every token must pass through them regardless of routing.
  // These divide by TP × PP only.
  //
  // ZeRO-1 shards optimizer states across DP; ZeRO-2 also shards gradients;
  // ZeRO-3 also shards parameters.  Non-expert weights are replicated across
  // EP, so their ZeRO shard denominator is TP × PP × DP (not × EP).

  const expertParamCount    = overview.expertParams;
  const nonExpertParamCount = Math.max(0, totalParams - expertParamCount);

  const gradBytesPerParam = training.gradPrecision === "bf16" ? 2 : 4;

  const denomEP  = Math.max(training.tp * training.ep * training.pp, 1); // expert MP group
  const denomNoEP = Math.max(training.tp * training.pp, 1);               // non-expert MP group
  const denomDP  = Math.max(training.dp, 1);
  const zeroStage = training.zeroStage ?? 0;

  // zeroDenom: apply ZeRO-stage DP sharding to a model-parallel base denominator.
  const zeroDenom = (base: number, minStage: number) =>
    zeroStage >= minStage ? base * denomDP : base;

  // Expert params/states — EP-sharded
  const expertParamsPerGpu    = expertParamCount * bytesParam             / zeroDenom(denomEP, 3);
  const expertOptimizerPerGpu = expertParamCount * optimizerBytesPerParam / zeroDenom(denomEP, 1);
  const expertGradsPerGpu     = expertParamCount * gradBytesPerParam      / zeroDenom(denomEP, 2);

  // Non-expert params/states — replicated across EP, NOT divided by EP
  const nonExpertParamsPerGpu    = nonExpertParamCount * bytesParam             / zeroDenom(denomNoEP, 3);
  const nonExpertOptimizerPerGpu = nonExpertParamCount * optimizerBytesPerParam / zeroDenom(denomNoEP, 1);
  const nonExpertGradsPerGpu     = nonExpertParamCount * gradBytesPerParam      / zeroDenom(denomNoEP, 2);

  const paramsStatesPerGpu =
    expertParamsPerGpu + expertOptimizerPerGpu + expertGradsPerGpu +
    nonExpertParamsPerGpu + nonExpertOptimizerPerGpu + nonExpertGradsPerGpu;

  // Expert FFN acts: stored on expert GPU → EP-sharded (denomEP = TP×EP×PP).
  // Home (residual/attn) acts: stored on home GPU → NOT EP-sharded (denomNoEP = TP×PP).
  const activationsPerGpu =
    expertActivationsBytes / denomEP +
    homeActivationsBytes   / denomNoEP;

  const perGpuBytes = paramsStatesPerGpu + activationsPerGpu;

  return {
    paramsBytes,
    optimizerBytes,
    gradientBytes,
    activationsBytes,
    peakBytes,
    perGpuBytes
  };
}

// ---- Inference metrics ----

export interface InferenceMetrics {
  // GPU / parallelism
  totalGpus: number;
  tp: number;
  ep: number;
  pp: number;
  dp: number;

  // FLOPs
  prefillFlopsPerToken: number;
  decodeFlopsPerToken: number;

  // Latency
  prefillLatencyMs: number;
  decodeLatencyMsPerToken: number;
  ttftMs: number;
  interTokenLatencyMs: number;
  totalDecodeTimeMs: number;
  totalGenerationTimeMs: number;

  // Throughput
  prefillTokensPerSec: number;
  decodeTokensPerSec: number;

  // Memory (total cluster)
  weightsBytes: number;
  kvBytesPerToken: number;
  kvTotalBytes: number;
  totalBytes: number;

  // Memory (per GPU)
  weightsPerGpuBytes: number;
  kvPerGpuBytes: number;
  totalPerGpuBytes: number;
  maxBatchSizeByMemory: number;

  // Decode bytes breakdown (for display/debugging)
  nonExpertActiveParams: number;
  activeExpertParams: number;
  activeSharedExpertParams: number;
  decodeBytesPerGpuPerStep: number;

  // Sequence-length sweep
  seqSamples: {
    seqLen: number;
    kvGB: number;
    decodeMs: number;
    ttftMs: number;
    totalTimeMs: number;
  }[];
}

function computeExpertParamSplit(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics
): {
  activeRoutedExpertParams: number;
  activeSharedExpertParams: number;
  totalActiveExpertParams: number;
  nonExpertActiveParams: number;
} {
  const numDenseLayers = model.firstKDenseReplace ?? 0;
  const moeLayers = model.layersMoe ?? Math.max(0, model.layers - numDenseLayers);

  // Routed experts: topK activated per token, SwiGLU = 3 * d_model * d_ff each.
  // These ARE sharded by EP — each EP rank owns only E/EP experts and processes
  // ~topK/EP token-expert assignments, so bandwidth divides by TP × EP.
  const perRoutedExpert = 3 * model.dModel * moe.dFf;
  const activeRoutedExpertParams = moeLayers * moe.topK * perRoutedExpert;

  // Shared experts: always active, SwiGLU = 3 * d_model * d_shared_ff each.
  // Shared experts are REPLICATED across all EP ranks (every token uses them,
  // so every EP rank must hold and compute them locally).  They shard by TP only.
  const perSharedExpert = 3 * model.dModel * moe.dSharedFf;
  const activeSharedExpertParams = moeLayers * moe.numSharedExperts * perSharedExpert;

  // EP-sharded bucket: routed experts only (÷ TP × EP in bandwidth calculation).
  const totalActiveExpertParams = activeRoutedExpertParams;

  // Non-EP-sharded bucket: shared experts + attention + embeddings + norms + gating
  // (÷ TP only in bandwidth calculation).
  const nonExpertActiveParams = Math.max(
    0,
    overview.activeParams - activeRoutedExpertParams
  );

  return {
    activeRoutedExpertParams,
    activeSharedExpertParams,
    totalActiveExpertParams,
    nonExpertActiveParams
  };
}

function computeKvBytesPerToken(
  model: ModelArchitectureConfig,
  bytesWeight: number
): { kvBytesPerToken: number; kvDimPerLayer: number; useMLACache: boolean } {
  const useMLACache =
    model.useMLA === true &&
    (model.mlaKvLoraRank ?? 0) > 0 &&
    (model.mlaQkRopeHeadDim ?? 0) > 0;

  const kvDimPerLayer = useMLACache
    ? model.mlaKvLoraRank! + model.mlaQkRopeHeadDim!   // d_c + d_h^rope
    : model.nKvHeads * model.dHead;                      // standard per-head

  // MLA: single compressed vector (no 2× for K+V separately)
  // Standard: 2× for separate K and V
  const kvBytesPerToken = useMLACache
    ? model.layers * kvDimPerLayer * bytesWeight
    : 2 * model.layers * kvDimPerLayer * bytesWeight;

  return { kvBytesPerToken, kvDimPerLayer, useMLACache };
}

function computeDecodeBytesPerGpu(
  nonExpertActiveParams: number,
  totalActiveExpertParams: number,
  bytesWeight: number,
  tp: number,
  ep: number,
  kvTotalBytes: number
): { decodeBytesPerGpu: number; nonExpertBytesPerGpu: number; expertBytesPerGpu: number; kvPerGpuBytes: number } {
  const nonExpertBytesPerGpu = (nonExpertActiveParams * bytesWeight) / tp;
  const expertBytesPerGpu = (totalActiveExpertParams * bytesWeight) / (tp * ep);
  const kvPerGpuBytes = kvTotalBytes / tp;

  const decodeBytesPerGpu = nonExpertBytesPerGpu + expertBytesPerGpu + kvPerGpuBytes;

  return { decodeBytesPerGpu, nonExpertBytesPerGpu, expertBytesPerGpu, kvPerGpuBytes };
}

export function computeInferenceMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  inference: InferenceConfig,
  gpu: GpuSpec
): InferenceMetrics {
  const bytesWeight = bytesPerParamInference(inference.precision);

  const tp = Math.max(inference.tp ?? 1, 1);
  const ep = Math.max(inference.ep ?? 1, 1);
  const pp = Math.max(inference.pp ?? 1, 1);
  const dp = Math.max(inference.dp ?? 1, 1);
  const totalGpus = tp * ep * pp * dp;

  const B = inference.batchSize;
  const S_in = inference.inputSeqLen;
  const S_out = inference.outputSeqLen;
  const S_total = S_in + S_out;

  // ── Expert vs non-expert param split ──
  const paramSplit = computeExpertParamSplit(model, moe, overview);

  // ── Precision-aware peak TFLOPS ──
  const infPrecision = inference.precision;
  const peakTflops =
    (infPrecision === "fp8" || infPrecision === "int8")
      ? (gpu.fp8Tflops ?? gpu.fp16Tflops)
      : gpu.fp16Tflops;
  const prefillUtil = 0.4;
  const effectiveTflops = peakTflops * prefillUtil;

  // ── FLOPs per token ──
  const forwardPrefill = forwardFlopsPerToken(model, moe, S_in).total;
  const forwardDecode = forwardFlopsPerToken(model, moe, S_total).total;

  // ── Prefill latency (compute-bound) ──
  // TP shards compute; EP parallelises expert compute; PP stages are sequential
  const computeGpus = tp * ep;
  const prefillFlopsTotal = forwardPrefill * B * Math.max(S_in, 1);
  const prefillLatencyMs =
    (prefillFlopsTotal / (computeGpus * effectiveTflops * 1e12)) * 1000;

  const prefillTokensProduced = B * Math.max(S_in, 1);
  const prefillTokensPerSec =
    prefillLatencyMs > 0 ? prefillTokensProduced / (prefillLatencyMs / 1000) : 0;

  // ── Memory bandwidth (per GPU) ──
  const memBW = gpu.memBandwidthGBs * 1e9; // bytes/s

  // ── KV cache ──
  const { kvBytesPerToken } = computeKvBytesPerToken(model, bytesWeight);
  const kvTotalBytes = kvBytesPerToken * B * Math.max(S_total, 1);

  // ── Decode latency (memory-bound) ──
  const decodeSharding = computeDecodeBytesPerGpu(
    paramSplit.nonExpertActiveParams,
    paramSplit.totalActiveExpertParams,
    bytesWeight,
    tp,
    ep,
    kvTotalBytes
  );

  const decodeLatencySec = memBW > 0 ? decodeSharding.decodeBytesPerGpu / memBW : 0;
  const decodeLatencyMsPerToken = decodeLatencySec * 1000;

  // Each decode step produces B tokens (one per sequence in the batch)
  const decodeTokensPerSec =
    decodeLatencySec > 0 ? B / decodeLatencySec : 0;

  // ── End-to-end timing ──
  const ttftMs = prefillLatencyMs;
  const interTokenLatencyMs = decodeLatencyMsPerToken;
  const totalDecodeTimeMs = S_out * decodeLatencyMsPerToken;
  const totalGenerationTimeMs = ttftMs + totalDecodeTimeMs;

  // ── Memory totals (cluster) ──
  const weightsBytes = overview.totalParams * bytesWeight;
  const totalBytes = weightsBytes + kvTotalBytes;

  // ── Per-GPU weight capacity ─────────────────────────────────────────────────
  // Routed expert weights are EP-sharded (each EP rank owns E/EP experts).
  // All other weights — attention, shared experts, norms, embeddings, output
  // head, gating — are replicated across every EP rank and do NOT divide by EP.
  // Both classes divide by TP (tensor-parallel within each group) and PP
  // (each pipeline stage holds 1/PP of layers).
  const expertWeightsBytes    = overview.expertParams * bytesWeight;
  const nonExpertWeightsBytes = Math.max(0, overview.totalParams - overview.expertParams) * bytesWeight;

  const weightsPerGpuBytes =
    expertWeightsBytes    / Math.max(tp * ep * pp, 1) +
    nonExpertWeightsBytes / Math.max(tp * pp, 1);

  // ── Per-GPU KV capacity ─────────────────────────────────────────────────────
  // KV cache for capacity purposes: each pipeline stage holds 1/PP of the
  // attention layers, and TP shards the heads within that stage.
  // (Note: computeDecodeBytesPerGpu.kvPerGpuBytes = kvTotal/tp is intentionally
  //  NOT divided by PP — that value is used for bandwidth/latency where
  //  sequential PP stages don't reduce total bytes streamed per decode step.)
  const kvCapacityPerGpu = kvTotalBytes / Math.max(tp * pp, 1);

  const totalPerGpuBytes = weightsPerGpuBytes + kvCapacityPerGpu;

  // ── Max batch by GPU memory ──
  const availableBytesPerGpu = Math.max(gpu.hbmGB * 1e9 - weightsPerGpuBytes, 0);
  const kvPerBatchPerSeq = kvBytesPerToken * Math.max(S_total, 1) / Math.max(tp * pp, 1);
  const maxBatchSizeByMemory =
    kvPerBatchPerSeq > 0 ? Math.floor(availableBytesPerGpu / kvPerBatchPerSeq) : 0;

  // ── Sequence-length sweep ──
  const seqSamples: InferenceMetrics["seqSamples"] = [];
  const maxSeq = Math.max(model.maxSeqLen, S_total);
  const samplePoints = [256, 512, 1024, 2048, 4096, 8192, 16384, maxSeq].filter(
    (v, idx, arr) => idx === arr.indexOf(v) && v > 0 && v <= maxSeq
  );

  for (const seqLen of samplePoints) {
    const kvBytes = kvBytesPerToken * B * Math.max(seqLen, 1);
    const kvGB = kvBytes / 1e9;

    // Recompute decode bytes at this sequence length
    const sweepSharding = computeDecodeBytesPerGpu(
      paramSplit.nonExpertActiveParams,
      paramSplit.totalActiveExpertParams,
      bytesWeight,
      tp,
      ep,
      kvBytes
    );
    const decodeMs = memBW > 0 ? (sweepSharding.decodeBytesPerGpu / memBW) * 1000 : 0;

    // TTFT at this sequence length (assuming entire seqLen is input)
    const fwdFlopsThisSeq = forwardFlopsPerToken(model, moe, seqLen).total;
    const prefillFlopsThisSeq = fwdFlopsThisSeq * B * seqLen;
    const ttftThisSeq = (prefillFlopsThisSeq / (computeGpus * effectiveTflops * 1e12)) * 1000;

    const totalTimeThisSeq = ttftThisSeq + S_out * decodeMs;

    seqSamples.push({ seqLen, kvGB, decodeMs, ttftMs: ttftThisSeq, totalTimeMs: totalTimeThisSeq });
  }

  return {
    totalGpus,
    tp,
    ep,
    pp,
    dp,
    prefillFlopsPerToken: forwardPrefill,
    decodeFlopsPerToken: forwardDecode,
    prefillLatencyMs,
    decodeLatencyMsPerToken,
    ttftMs,
    interTokenLatencyMs,
    totalDecodeTimeMs,
    totalGenerationTimeMs,
    prefillTokensPerSec,
    decodeTokensPerSec,
    weightsBytes,
    kvBytesPerToken,
    kvTotalBytes,
    totalBytes,
    weightsPerGpuBytes,
    kvPerGpuBytes: decodeSharding.kvPerGpuBytes,
    totalPerGpuBytes,
    maxBatchSizeByMemory,
    nonExpertActiveParams: paramSplit.nonExpertActiveParams,
    activeExpertParams: paramSplit.activeRoutedExpertParams,
    activeSharedExpertParams: paramSplit.activeSharedExpertParams,
    decodeBytesPerGpuPerStep: decodeSharding.decodeBytesPerGpu,
    seqSamples
  };
}

// ---- Parallelism & communication metrics ----

export interface ParallelismMetrics {
  totalGpus: number;
  expertsPerGpu: number;
  layersPerGpu: number;
  allToAllBytesPerStep: number;
  allReduceTpBytesPerStep: number;
  p2pBytesPerStep: number;
  allReduceDpBytesPerStep: number;
  totalBytesPerStep: number;
}

export function computeParallelismMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  training: TrainingConfig
): ParallelismMetrics {
  const d = model.dModel;
  const S = model.maxSeqLen;
  const bytes = bytesPerParamTraining(training.precision);

  const totalGpus = Math.max(
    1,
    training.tp * training.ep * training.pp * training.dp
  );

  const expertsPerGpu =
    training.ep > 0 ? moe.numExperts / training.ep : moe.numExperts;
  const layersPerGpu =
    training.pp > 0 ? model.layers / training.pp : model.layers;

  const tokensPerStep = training.globalBatchTokens;
  const tokensPerReplica = tokensPerStep / Math.max(training.dp, 1);
  const microBatchSeq = Math.max(tokensPerReplica / Math.max(S, 1), 1);

  // All-to-all (EP) per MoE layer per step — only MoE layers incur EP traffic.
  const numDenseLayers = model.firstKDenseReplace ?? 0;
  const moeLayers = model.layersMoe ?? Math.max(0, model.layers - numDenseLayers);
  const allToAllPerLayer =
    2 *
    microBatchSeq *
    S *
    d *
    bytes *
    (training.ep > 0 ? (training.ep - 1) / training.ep : 0);
  const allToAllBytesPerStep = allToAllPerLayer * moeLayers;

  // AllReduce (TP) per layer.
  const allReduceTpPerLayer =
    2 *
    microBatchSeq *
    S *
    d *
    bytes *
    2 *
    (training.tp > 0 ? (training.tp - 1) / training.tp : 0);
  const allReduceTpBytesPerStep = allReduceTpPerLayer * model.layers;

  // P2P (PP) per micro-batch boundary.
  const p2pBytesPerStep =
    microBatchSeq *
    S *
    d *
    bytes *
    Math.max(training.pp - 1, 0);

  // AllReduce (DP) once per step for gradients.
  const allReduceDpBytesPerStep =
    2 *
    overview.totalParams *
    bytes *
    (training.dp > 0 ? (training.dp - 1) / training.dp : 0);

  const totalBytesPerStep =
    allToAllBytesPerStep +
    allReduceTpBytesPerStep +
    p2pBytesPerStep +
    allReduceDpBytesPerStep;

  return {
    totalGpus,
    expertsPerGpu,
    layersPerGpu,
    allToAllBytesPerStep,
    allReduceTpBytesPerStep,
    p2pBytesPerStep,
    allReduceDpBytesPerStep,
    totalBytesPerStep
  };
}

// ---- Routing & efficiency metrics ----

export interface RoutingEfficiencyMetrics {
  activeParamRatio: number;
  gatingOverheadPct: number;
  expertCapacityTokens: number;
  theoreticalDropRate: number;
  flopsMoePerToken: number;
  flopsDenseTotalPerToken: number;
  flopsDenseActivePerToken: number;
}

export function computeRoutingEfficiencyMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  training: TrainingConfig
): RoutingEfficiencyMetrics {
  const activeParamRatio =
    moe.numExperts > 0 ? moe.topK / moe.numExperts : 0;

  const S = model.maxSeqLen;
  const fwd = forwardFlopsPerToken(model, moe, S);

  const gatingOverheadPct = fwd.total > 0 ? fwd.gating / fwd.total : 0;

  const tokensPerBatch = training.globalBatchTokens;
  const capacityFactor = 1.25;
  const tokensPerExpert =
    moe.numExperts > 0
      ? (tokensPerBatch * Math.max(moe.topK, 1)) / moe.numExperts
      : tokensPerBatch;
  const expertCapacityTokens = capacityFactor * tokensPerExpert;

  // Poisson/normal-approximation-based drop rate.
  let theoreticalDropRate = 0;
  if (tokensPerExpert > 0 && moe.numExperts > 0) {
    const lambda = tokensPerExpert;
    const z = (capacityFactor - 1) * Math.sqrt(lambda);
    const tail = normalTail(z);
    theoreticalDropRate = Math.min(Math.max(tail, 0), 1);
  }

  const flopsMoePerToken = fwd.total;
  const flopsDenseTotalPerToken = denseEquivalentForwardFlopsPerToken(
    model,
    moe,
    S
  );

  const flopsDenseActivePerToken =
    flopsDenseTotalPerToken *
    (overview.activeParams / Math.max(overview.totalParams, 1));

  return {
    activeParamRatio,
    gatingOverheadPct,
    expertCapacityTokens,
    theoreticalDropRate,
    flopsMoePerToken,
    flopsDenseTotalPerToken,
    flopsDenseActivePerToken
  };
}

// ---- Hardware efficiency metrics ----

export interface HardwareEfficiencyMetrics {
  // Training
  mfu: number;
  hfu: number;
  recomputeMultiplier: number;
  pipelineBubblePct: number;
  numMicrobatches: number;
  effectiveThroughputTokensPerSecPerGpu: number;
  computeTimePerStepMs: number;
  commTimePerStepMs: number;
  computeCommOverlapPct: number;

  // Inference roofline
  prefillArithmeticIntensity: number;
  decodeArithmeticIntensity: number;
  ridgePointFlopsPerByte: number;
  peakTflops: number;
  memBWTBs: number;
  prefillAchievableTflops: number;
  decodeAchievableTflops: number;
  prefillGpuUtil: number;
  decodeGpuUtil: number;
  prefillBound: "compute" | "memory";
  decodeBound: "compute" | "memory";
  rooflineCurve: { ai: number; tflops: number }[];
  rooflinePoints: { label: string; ai: number; tflops: number }[];
}

export function computeHardwareEfficiencyMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  training: TrainingConfig,
  trainingGpu: GpuSpec,
  mfu: number,
  inference: InferenceConfig,
  inferenceGpu: GpuSpec,
  parallelism: ParallelismMetrics
): HardwareEfficiencyMetrics {
  const S = model.maxSeqLen;
  const fwd = forwardFlopsPerToken(model, moe, S);

  // ── Training efficiency ───────────────────────────────────

  const precision = training.precision;
  const peakTrainTflops =
    precision === "fp32"
      ? trainingGpu.fp32Tflops
      : precision === "fp8"
        ? (trainingGpu.fp8TrainingTflops ??
           trainingGpu.fp8Tflops ??
           trainingGpu.fp16Tflops * 2)
        : trainingGpu.fp16Tflops;

  // HFU: includes recomputation FLOPs from activation checkpointing.
  let recomputeMultiplier = 1.0;
  switch (training.activationCheckpointing) {
    case "sqrt":
      recomputeMultiplier = 4 / 3;
      break;
    case "selective":
      recomputeMultiplier = fwd.total > 0
        ? 1 + fwd.attention / (3 * fwd.total)
        : 1;
      break;
    case "none":
    default:
      recomputeMultiplier = 1.0;
  }
  const hfu = mfu * recomputeMultiplier;

  // Pipeline bubble
  const totalGpus = Math.max(1, training.tp * training.ep * training.pp * training.dp);
  const tokensPerDp = training.globalBatchTokens / Math.max(training.dp, 1);
  const numMicrobatches = Math.max(1, Math.floor(tokensPerDp / Math.max(S, 1)));
  const pipelineBubblePct = training.pp > 1
    ? ((training.pp - 1) / numMicrobatches) * 100
    : 0;

  // Effective throughput: tokens/sec/GPU
  const effectiveTflops = peakTrainTflops * mfu;
  const flopsPerToken3x = 3 * fwd.total;
  const tokensPerSecPerGpu = flopsPerToken3x > 0
    ? (effectiveTflops * 1e12) / flopsPerToken3x
    : 0;

  // Compute-comm overlap
  const tokensPerStepPerGpu = training.globalBatchTokens / totalGpus;
  const computeFlopsPerStep = flopsPerToken3x * tokensPerStepPerGpu;
  const computeTimeMs = peakTrainTflops > 0
    ? (computeFlopsPerStep / (peakTrainTflops * mfu * 1e12)) * 1000
    : 0;
  const interconnectBW = trainingGpu.interconnectBWGBs * 1e9;
  const commTimeMs = interconnectBW > 0
    ? (parallelism.totalBytesPerStep / interconnectBW) * 1000
    : 0;
  const computeCommOverlapPct = Math.max(computeTimeMs, commTimeMs) > 0
    ? (Math.min(computeTimeMs, commTimeMs) / Math.max(computeTimeMs, commTimeMs)) * 100
    : 100;

  // ── Inference roofline ────────────────────────────────────

  const bytesWeight = bytesPerParamInference(inference.precision);
  const infPrecision = inference.precision;
  const peakInfTflops = (infPrecision === "fp8" || infPrecision === "int8")
    ? (inferenceGpu.fp8Tflops ?? inferenceGpu.fp16Tflops)
    : inferenceGpu.fp16Tflops;
  const memBWBytesPerSec = inferenceGpu.memBandwidthGBs * 1e9;
  const memBWTBs = inferenceGpu.memBandwidthGBs / 1000;

  const ridgePointFlopsPerByte = memBWBytesPerSec > 0
    ? (peakInfTflops * 1e12) / memBWBytesPerSec
    : 1;

  const fwdInf = forwardFlopsPerToken(model, moe, inference.inputSeqLen);
  const weightsBytes = overview.totalParams * bytesWeight;

  // Prefill: batch × seq tokens processed, weights read once
  const prefillTotalFlops = fwdInf.total * inference.batchSize * Math.max(inference.inputSeqLen, 1);
  const prefillBytesAccessed = weightsBytes;
  const prefillArithmeticIntensity = prefillBytesAccessed > 0
    ? prefillTotalFlops / prefillBytesAccessed
    : 0;

  // Decode: single token, read active weights + KV cache (using proper sharding)
  const paramSplit = computeExpertParamSplit(model, moe, overview);
  const { kvBytesPerToken } = computeKvBytesPerToken(model, bytesWeight);
  const S_total = inference.inputSeqLen + inference.outputSeqLen;
  const kvCacheBytesTotal = kvBytesPerToken * inference.batchSize * S_total;

  // For roofline, use per-GPU bytes (sharded)
  const tp = Math.max(inference.tp ?? 1, 1);
  const ep = Math.max(inference.ep ?? 1, 1);
  const decodeSharding = computeDecodeBytesPerGpu(
    paramSplit.nonExpertActiveParams,
    paramSplit.totalActiveExpertParams,
    bytesWeight,
    tp,
    ep,
    kvCacheBytesTotal
  );
  const decodeBytesAccessed = decodeSharding.decodeBytesPerGpu;

  const decodeFwdFlops = fwdInf.total;
  const decodeArithmeticIntensity = decodeBytesAccessed > 0
    ? decodeFwdFlops / decodeBytesAccessed
    : 0;

  // Achievable TFLOPS on roofline
  const roofline = (ai: number) =>
    Math.min(memBWBytesPerSec * ai, peakInfTflops * 1e12) / 1e12;

  const prefillAchievableTflops = roofline(prefillArithmeticIntensity);
  const decodeAchievableTflops = roofline(decodeArithmeticIntensity);

  const prefillBound: "compute" | "memory" =
    prefillArithmeticIntensity >= ridgePointFlopsPerByte ? "compute" : "memory";
  const decodeBound: "compute" | "memory" =
    decodeArithmeticIntensity >= ridgePointFlopsPerByte ? "compute" : "memory";

  // GPU utilization
  const decodeLatencySec = memBWBytesPerSec > 0 ? decodeBytesAccessed / memBWBytesPerSec : 0;
  const decodeActualFlops = decodeLatencySec > 0 ? decodeFwdFlops / decodeLatencySec : 0;
  const decodeGpuUtil = peakInfTflops > 0
    ? (decodeActualFlops / (peakInfTflops * 1e12)) * 100
    : 0;
  const prefillUtil = 0.4;
  const prefillGpuUtil = prefillUtil * 100;

  // Roofline curve data (log-spaced AI values)
  const rooflineCurve: { ai: number; tflops: number }[] = [];
  for (let exp = -1; exp <= 5; exp += 0.15) {
    const ai = Math.pow(10, exp);
    rooflineCurve.push({ ai, tflops: roofline(ai) });
  }

  const rooflinePoints = [
    { label: "Prefill", ai: prefillArithmeticIntensity, tflops: prefillAchievableTflops },
    { label: "Decode", ai: decodeArithmeticIntensity, tflops: decodeAchievableTflops }
  ];

  return {
    mfu,
    hfu,
    recomputeMultiplier,
    pipelineBubblePct,
    numMicrobatches,
    effectiveThroughputTokensPerSecPerGpu: tokensPerSecPerGpu,
    computeTimePerStepMs: computeTimeMs,
    commTimePerStepMs: commTimeMs,
    computeCommOverlapPct,
    prefillArithmeticIntensity,
    decodeArithmeticIntensity,
    ridgePointFlopsPerByte,
    peakTflops: peakInfTflops,
    memBWTBs,
    prefillAchievableTflops,
    decodeAchievableTflops,
    prefillGpuUtil,
    decodeGpuUtil,
    prefillBound,
    decodeBound,
    rooflineCurve,
    rooflinePoints
  };
}

// ---- Cost analysis metrics ----

export interface CostAnalysisMetrics {
  // Training
  gpuHours: number;
  costPerGpuHour: number;
  totalTrainingCost: number;
  costPerTrillionTokens: number;
  wallClockHours: number;
  wallClockDays: number;
  totalGpus: number;
  totalTrainingTokens: number;

  // Inference (at configured batch/precision)
  prefillTokensPerSec: number;
  decodeTokensPerSec: number;
  costPer1MInputTokens: number;
  costPer1MOutputTokens: number;

  // Replica GPU counts
  nPrefillGpus: number;
  nDecodeGpus: number;

  // Batch size sweep
  batchSweep: {
    batchSize: number;
    decodeTokensPerSec: number;
    costPer1MOutput: number;
    costPer1MInput: number;
  }[];

  // Quantization sweep
  quantSweep: {
    precision: PrecisionInference;
    label: string;
    weightsGB: number;
    decodeTokensPerSec: number;
    costPer1MOutput: number;
    relativeToFp16: number;
  }[];
}

export function computeCostAnalysisMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  training: TrainingConfig,
  trainingGpu: GpuSpec,
  mfu: number,
  inference: InferenceConfig,
  inferenceGpu: GpuSpec,
  trainingCompute: TrainingComputeMetrics
): CostAnalysisMetrics {
  const costPerGpuHour = trainingGpu.costPerHourUSD;
  const gpuHours = trainingCompute.gpuHoursApprox;
  const totalTrainingCost = trainingCompute.trainingCostUSD;
  const totalGpus = trainingCompute.totalGpus;
  const totalTrainingTokens = training.totalTrainingTokens;

  const costPerTrillionTokens = totalTrainingTokens > 0
    ? totalTrainingCost / (totalTrainingTokens / 1e12)
    : 0;

  const wallClockHours = totalGpus > 0 ? gpuHours / totalGpus : 0;
  const wallClockDays = wallClockHours / 24;

  // ── Inference costs ─────────────────────────────────────

  const memBW = inferenceGpu.memBandwidthGBs * 1e9;
  const fwdFlops = forwardFlopsPerToken(model, moe, inference.inputSeqLen).total;

  const peakInfTflops = (inference.precision === "fp8" || inference.precision === "int8")
    ? (inferenceGpu.fp8Tflops ?? inferenceGpu.fp16Tflops)
    : inferenceGpu.fp16Tflops;
  const prefillUtil = 0.4;
  const effectiveInfTflops = peakInfTflops * prefillUtil;

  const tp = Math.max(inference.tp ?? 1, 1);
  const ep = Math.max(inference.ep ?? 1, 1);
  const pp = Math.max(inference.pp ?? 1, 1);
  const nPrefillGpus = tp * ep * pp;
  const nDecodeGpus = tp * ep * pp;

  const paramSplit = computeExpertParamSplit(model, moe, overview);

  // Helper: compute inference cost at a given batch size and precision
  function inferCostAtBatchPrec(batchSize: number, prec: PrecisionInference) {
    const bpp = bytesPerParamInference(prec);
    const totalWeightBytes = overview.totalParams * bpp;
    const { kvBytesPerToken: kvPerToken } = computeKvBytesPerToken(model, bpp);
    const S_total = inference.inputSeqLen + inference.outputSeqLen;

    // Prefill: compute-bound, process B × S_in tokens
    const computeGpus = tp * ep;
    const prefillFlopsTotal = fwdFlops * batchSize * Math.max(inference.inputSeqLen, 1);
    const prefillTimeSec = prefillFlopsTotal / (computeGpus * effectiveInfTflops * 1e12);
    const prefillTokensProduced = batchSize * Math.max(inference.inputSeqLen, 1);
    const prefillTPS = prefillTimeSec > 0 ? prefillTokensProduced / prefillTimeSec : 0;

    // Decode: memory-bound with proper EP sharding
    const kvTotal = kvPerToken * batchSize * S_total;
    const decodeSharding = computeDecodeBytesPerGpu(
      paramSplit.nonExpertActiveParams,
      paramSplit.totalActiveExpertParams,
      bpp,
      tp,
      ep,
      kvTotal
    );
    const decodeTimeSec = memBW > 0 ? decodeSharding.decodeBytesPerGpu / memBW : 0;
    const decodeTPS = decodeTimeSec > 0 ? batchSize / decodeTimeSec : 0;

    // Cost per 1M tokens = (10^6 / TPS) * (costPerGpuHour / 3600) * N_replica
    const gpuCostPerSec = inferenceGpu.costPerHourUSD / 3600;
    const costPer1MInput = prefillTPS > 0
      ? (1e6 / prefillTPS) * gpuCostPerSec * nPrefillGpus
      : 0;
    const costPer1MOutput = decodeTPS > 0
      ? (1e6 / decodeTPS) * gpuCostPerSec * nDecodeGpus
      : 0;

    return {
      prefillTPS,
      decodeTPS,
      costPer1MInput,
      costPer1MOutput,
      weightsGB: totalWeightBytes / 1e9
    };
  }

  const base = inferCostAtBatchPrec(inference.batchSize, inference.precision);

  // Batch size sweep
  const batchSizes = [1, 4, 8, 16, 32, 64, 128];
  const batchSweep = batchSizes.map((bs) => {
    const r = inferCostAtBatchPrec(bs, inference.precision);
    return {
      batchSize: bs,
      decodeTokensPerSec: r.decodeTPS,
      costPer1MOutput: r.costPer1MOutput,
      costPer1MInput: r.costPer1MInput
    };
  });

  // Quantization sweep
  const precisions: { prec: PrecisionInference; label: string }[] = [
    { prec: "fp16", label: "FP16" },
    { prec: "bf16", label: "BF16" },
    { prec: "fp8", label: "FP8" },
    { prec: "int8", label: "INT8" },
    { prec: "int4", label: "INT4" }
  ];
  const fp16Base = inferCostAtBatchPrec(inference.batchSize, "fp16");
  const quantSweep = precisions.map(({ prec, label }) => {
    const r = inferCostAtBatchPrec(inference.batchSize, prec);
    return {
      precision: prec,
      label,
      weightsGB: r.weightsGB,
      decodeTokensPerSec: r.decodeTPS,
      costPer1MOutput: r.costPer1MOutput,
      relativeToFp16: fp16Base.costPer1MOutput > 0
        ? fp16Base.costPer1MOutput / r.costPer1MOutput
        : 1
    };
  });

  return {
    gpuHours,
    costPerGpuHour,
    totalTrainingCost,
    costPerTrillionTokens,
    wallClockHours,
    wallClockDays,
    totalGpus,
    totalTrainingTokens,
    prefillTokensPerSec: base.prefillTPS,
    decodeTokensPerSec: base.decodeTPS,
    costPer1MInputTokens: base.costPer1MInput,
    costPer1MOutputTokens: base.costPer1MOutput,
    nPrefillGpus,
    nDecodeGpus,
    batchSweep,
    quantSweep
  };
}

// Approximate upper-tail probability for standard normal N(0,1).
function normalTail(z: number): number {
  const absZ = Math.abs(z);
  const t = 1 / (1 + 0.2316419 * absZ);
  const d = 0.3989423 * Math.exp(-0.5 * absZ * absZ);
  const prob =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z < 0 ? 1 - prob : prob;
}