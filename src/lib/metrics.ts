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
  return 2; // bf16 / fp8
}

export function bytesPerParamInference(precision: PrecisionInference): number {
  switch (precision) {
    case "fp16":
    case "bf16":
    case "fp8":
      return 2;
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
  const Hkv = model.nKvHeads;
  const dh = model.dHead;
  const S = seqLen;

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
  const L = model.layers;

  // Per active expert (SwiGLU): 6 × d_model × d_ff
  const perExpert = 6 * d * moe.dFf;
  const perShared = 6 * d * moe.dSharedFf;

  const moeFfn = L * moe.topK * perExpert;
  const sharedFfn = L * moe.numSharedExperts * perShared;

  // Gating: 2 × d_model × E per MoE layer
  const gating = L * 2 * d * moe.numExperts;

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
  total: number;
} {
  const attention = attentionFlopsPerToken(model, seqLen);
  const { moeFfn, sharedFfn, gating } = moeFlopsComponentsPerToken(model, moe);
  const total = attention + moeFfn + sharedFfn + gating;
  return { attention, moeFfn, sharedFfn, gating, total };
}

export function denseEquivalentForwardFlopsPerToken(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  seqLen: number
): number {
  const d = model.dModel;
  const L = model.layers;

  const attention = attentionFlopsPerToken(model, seqLen);

  // Dense equivalent: all experts fire (K_dense = E).
  const perExpertDense = 6 * d * moe.dFf;
  const denseFfn = L * moe.numExperts * perExpertDense;
  const shared = L * moe.numSharedExperts * (6 * d * moe.dSharedFf);

  return attention + denseFfn + shared;
}

// ---- Training compute metrics ----

export interface TrainingComputeMetrics {
  attentionFlopsPerToken: number;
  moeFfnFlopsPerToken: number;
  sharedFfnFlopsPerToken: number;
  gatingFlopsPerToken: number;
  forwardFlopsPerToken: number;
  backwardFlopsPerToken: number;
  totalFlopsPerToken: number;
  totalTrainingFlops: number;
  denseEquivalentFlopsPerToken: number;
  totalGpus: number;
  gpuHoursApprox: number;
}

export function computeTrainingComputeMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  training: TrainingConfig,
  gpu: GpuSpec
): TrainingComputeMetrics {
  const S = model.maxSeqLen;
  const { attention, moeFfn, sharedFfn, gating, total } = forwardFlopsPerToken(
    model,
    moe,
    S
  );
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
    precision === "fp32" ? gpu.fp32Tflops : gpu.fp16Tflops;

  const utilization = 0.35; // typical MFU for large MoE training
  const effectivePerGpuTflops = peakPerGpuTflops * utilization;

  const timeSecondsCluster =
    totalTrainingFlops / (effectivePerGpuTflops * totalGpus * 1e12);
  const gpuHours = (timeSecondsCluster * totalGpus) / 3600;

  return {
    attentionFlopsPerToken: attention,
    moeFfnFlopsPerToken: moeFfn,
    sharedFfnFlopsPerToken: sharedFfn,
    gatingFlopsPerToken: gating,
    forwardFlopsPerToken: forward,
    backwardFlopsPerToken: backward,
    totalFlopsPerToken: perTokenTotal,
    totalTrainingFlops,
    denseEquivalentFlopsPerToken: denseEq,
    totalGpus,
    gpuHoursApprox: gpuHours
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

  let optimizerBytesPerParam: number;
  if (training.optimizer === "adam") {
    optimizerBytesPerParam = training.precision === "fp32" ? 12 : 8;
  } else {
    optimizerBytesPerParam = 8;
  }
  const optimizerBytes = totalParams * optimizerBytesPerParam;

  const gradientBytes =
    totalParams * (training.gradPrecision === "bf16" ? 2 : 4);

  // Activation memory (approximate, based on Python engine structure).
  const seqLen = model.maxSeqLen;
  const d = model.dModel;
  const nLayers = model.layers;

  const tokensPerBatch = training.globalBatchTokens;
  const batchSizeSeq = Math.max(1, tokensPerBatch / Math.max(seqLen, 1));

  const attnAct =
    training.useFlashAttention
      ? 0
      : 2 *
        batchSizeSeq *
        model.nHeads *
        seqLen *
        seqLen *
        bytesParam;
  const ffnAct =
    batchSizeSeq *
    seqLen *
    moe.dFf *
    Math.max(moe.topK, 1) *
    bytesParam;
  const residual =
    batchSizeSeq * seqLen * d * bytesParam * 2; // residual stream
  const actPerLayer = attnAct + ffnAct + residual;

  let activationsBytes: number;
  switch (training.activationCheckpointing) {
    case "none":
      activationsBytes = actPerLayer * nLayers;
      break;
    case "sqrt":
      activationsBytes =
        actPerLayer * 2 * Math.ceil(Math.sqrt(nLayers || 1));
      break;
    case "selective":
    default: {
      const actPerLayerSelective = ffnAct + residual;
      activationsBytes = actPerLayerSelective * nLayers;
      break;
    }
  }

  const peakBytes = paramsBytes + optimizerBytes + gradientBytes + activationsBytes;

  const denomParams = Math.max(training.tp * training.ep * training.pp, 1);
  const denomActs = Math.max(training.tp * training.pp, 1);

  const paramsStatesPerGpu =
    (paramsBytes + optimizerBytes + gradientBytes) / denomParams;
  const activationsPerGpu = activationsBytes / denomActs;

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
  prefillFlopsPerToken: number;
  decodeFlopsPerToken: number;
  prefillLatencyMs: number;
  decodeLatencyMsPerToken: number;
  tokensPerSecond: number;
  weightsBytes: number;
  kvBytesPerToken: number;
  kvTotalBytes: number;
  totalBytes: number;
  maxBatchSizeByMemory: number;
  seqSamples: { seqLen: number; kvGB: number; decodeMs: number }[];
}

export function computeInferenceMetrics(
  model: ModelArchitectureConfig,
  moe: MoeConfig,
  overview: DerivedOverviewMetrics,
  inference: InferenceConfig,
  gpu: GpuSpec
): InferenceMetrics {
  const d = model.dModel;
  const L = model.layers;

  const bytesWeight = bytesPerParamInference(inference.precision);

  const S_in = inference.inputSeqLen;
  const S_total = inference.inputSeqLen + inference.outputSeqLen;

  const forwardPrefill = forwardFlopsPerToken(model, moe, S_in).total;
  const forwardDecode = forwardFlopsPerToken(model, moe, S_in).total;

  const prefillFlopsTotal =
    forwardPrefill * inference.batchSize * Math.max(S_in, 1);

  const peakTflops = gpu.fp16Tflops;
  const util = 0.4; // slightly higher for inference
  const effectiveTflops = peakTflops * util;

  const prefillLatencyMs =
    prefillFlopsTotal / (effectiveTflops * 1e12) * 1000;

  // Decode is memory-bound: time ≈ bytes moved / bandwidth.
  const memBW = gpu.memBandwidthGBs * 1e9; // bytes/s
  const decodeBytesPerToken = overview.activeParams * bytesWeight;
  const decodeLatencySec = decodeBytesPerToken / memBW;
  const decodeLatencyMsPerToken = decodeLatencySec * 1000;
  const tokensPerSecond = decodeLatencySec > 0 ? 1 / decodeLatencySec : 0;

  const weightsBytes = overview.totalParams * bytesWeight;

  // KV cache memory.
  const kvBytesPerToken =
    2 * L * model.nKvHeads * model.dHead * bytesWeight;
  const kvTotalBytes =
    kvBytesPerToken * inference.batchSize * Math.max(S_total, 1);

  const totalBytes = weightsBytes + kvTotalBytes;

  const availableBytes = Math.max(gpu.hbmGB * 1e9 - weightsBytes, 0);
  const denom = kvBytesPerToken * Math.max(S_total, 1);
  const maxBatchSizeByMemory =
    denom > 0 ? Math.floor(availableBytes / denom) : 0;

  // Samples for KV vs seq_len + decode latency.
  const seqSamples: { seqLen: number; kvGB: number; decodeMs: number }[] = [];
  const maxSeq = Math.max(model.maxSeqLen, S_total);
  const samplePoints = [256, 512, 1024, 2048, 4096, maxSeq].filter(
    (v, idx, arr) => idx === arr.indexOf(v) && v > 0 && v <= maxSeq
  );

  for (const seqLen of samplePoints) {
    const kvBytes =
      kvBytesPerToken * inference.batchSize * Math.max(seqLen, 1);
    const kvGB = kvBytes / 1e9;
    const decodeBytesThisSeq =
      overview.activeParams * bytesWeight +
      kvBytesPerToken * inference.batchSize * Math.max(seqLen, 1);
    const decodeMs =
      (decodeBytesThisSeq / memBW) * 1000;
    seqSamples.push({ seqLen, kvGB, decodeMs });
  }

  return {
    prefillFlopsPerToken: forwardPrefill,
    decodeFlopsPerToken: forwardDecode,
    prefillLatencyMs,
    decodeLatencyMsPerToken,
    tokensPerSecond,
    weightsBytes,
    kvBytesPerToken,
    kvTotalBytes,
    totalBytes,
    maxBatchSizeByMemory,
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

  // All-to-all (EP) per MoE layer per step.
  const allToAllPerLayer =
    2 *
    microBatchSeq *
    S *
    d *
    bytes *
    (training.ep > 0 ? (training.ep - 1) / training.ep : 0);
  const allToAllBytesPerStep = allToAllPerLayer * model.layers;

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
  const { attention, moeFfn, sharedFfn, gating, total } =
    forwardFlopsPerToken(model, moe, S);

  const gatingOverheadPct = total > 0 ? gating / total : 0;

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

  const flopsMoePerToken = attention + moeFfn + sharedFfn + gating;
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

