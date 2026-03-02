import { create } from "zustand";

export type PrecisionTraining = "fp32" | "bf16" | "fp8";
export type PrecisionInference = "fp16" | "bf16" | "fp8" | "int8" | "int4";

export type ExpertGranularity = "fine" | "coarse";

export type TrainingOptimizer = "adam" | "adafactor";

export type GradPrecision = "fp32" | "bf16";

export type ActivationCheckpointingMode = "none" | "sqrt" | "selective";

export interface ModelArchitectureConfig {
  totalParamsB: number;
  dModel: number;
  layers: number;
  /** Number of MoE layers (L_moe). When undefined, assumed equal to L (all layers MoE). */
  layersMoe?: number;
  /** First K layers are dense FFN (not MoE). DeepSeek-V3 uses 3. */
  firstKDenseReplace?: number;
  /** Dense FFN intermediate size for non-MoE layers. DeepSeek-V3 uses 18432. */
  denseIntermediateSize?: number;
  /** Parallel residual MLP per MoE layer (Arctic). When true, each MoE layer has dense MLP + MoE. */
  useResidualMLP?: boolean;
  /** Residual MLP intermediate size (when useResidualMLP). Arctic uses 4864. */
  residualMLPIntermediateSize?: number;
  /** Multi-head Latent Attention (MLA) - reduces KV cache. DeepSeek-V2/V3. */
  useMLA?: boolean;
  /** MLA: Q low-rank projection dim. */
  mlaQLoraRank?: number;
  /** MLA: KV low-rank projection dim. */
  mlaKvLoraRank?: number;
  /** MLA: head dim for Q/K RoPE (positional). */
  mlaQkRopeHeadDim?: number;
  /** MLA: head dim for Q/K non-positional. */
  mlaQkNopeHeadDim?: number;
  /** MLA: head dim for V. */
  mlaVHeadDim?: number;
  /** Tied embedding and output head — don't double-count. */
  tieWordEmbeddings?: boolean;
  nHeads: number;
  nKvHeads: number;
  dHead: number;
  vocabSize: number;
  maxSeqLen: number;
}

export interface MoeConfig {
  numExperts: number;
  topK: number;
  numSharedExperts: number;
  dFf: number;
  dSharedFf: number;
  expertGranularity: ExpertGranularity;
}

export interface TrainingConfig {
  globalBatchTokens: number;
  totalTrainingTokens: number;
  precision: PrecisionTraining;
  optimizer: TrainingOptimizer;
  gradPrecision: GradPrecision;
  activationCheckpointing: ActivationCheckpointingMode;
  useFlashAttention: boolean;
  /** GPU type used for training (Fgpu). DeepSeek-V3 trained on H800, not H100. */
  trainingGpuType: GpuType;
  /** Model FLOPs Utilization (MFU). Critical for GPU-hours; DeepSeek-V3 reported ~0.55+. */
  mfu: number;
  tp: number;
  ep: number;
  pp: number;
  dp: number;
}

export type GpuType =
  | "A100-80G"
  | "H100-80G"
  | "H800-80G"
  | "H200-141G"
  | "B200-192G"
  | "RTX-4090"
  | "M4-Max";

export interface InferenceConfig {
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  precision: PrecisionInference;
  gpuType: GpuType;
  tp: number;
  ep: number;
  pp: number;
  dp: number;
}

export type PresetId =
  | "deepseek-v2"
  | "deepseek-v3"
  | "mixtral-8x7b"
  | "mixtral-8x22b"
  | "custom";

export interface PaperReferenceMetrics {
  name: string;
  totalParamsB?: number;
  activeParamsB?: number;
  /** Per-token training FLOPs in trillions (TF). */
  trainingFlopsPerTokenTF?: number;
  /** Total training FLOPs in petaflops (PF). */
  totalTrainingFlopsPF?: number;
  /** Peak training memory in GB. */
  trainingMemoryGB?: number;
  /** KV cache bytes per token in KB. */
  kvCachePerTokenKB?: number;
  /** Reported GPU hours for full training. */
  gpuHoursReported?: number;
  /** Inference model weights in GB (e.g. bf16). */
  inferenceWeightsGB?: number;
  /** Training memory in TB (for very large models). */
  trainingMemoryTB?: number;
  /** Training cost in USD as reported by the paper/blog. */
  trainingCostUSD?: number;
  /** Number of GPUs used for training. */
  numGpus?: number;
  /** Forward FLOPs per token in billions (10^9). */
  forwardFlopsPerTokenB?: number;
  /** Expert parameters per expert across all MoE layers, in billions. */
  expertParamsPerExpertB?: number;
  /** Embedding parameters in millions. */
  embeddingParamsM?: number;
}

export interface DerivedOverviewMetrics {
  totalParams: number;
  embeddingParams: number;
  attentionParams: number;
  expertParams: number;
  sharedExpertParams: number;
  denseParams: number;
  residualMLPParams: number;
  gatingParams: number;
  normParams: number;
  outputHeadParams: number;
  activeParams: number;
}

export interface ConfigState {
  // Active config (used for metrics) — updated only on Run or preset change
  model: ModelArchitectureConfig;
  moe: MoeConfig;
  training: TrainingConfig;
  inference: InferenceConfig;
  preset: PresetId;

  // Draft config (edited by knobs) — updates on every knob change
  draftModel: ModelArchitectureConfig;
  draftMoe: MoeConfig;
  draftTraining: TrainingConfig;
  draftInference: InferenceConfig;

  // Paper reference for validation tab
  paperReference?: PaperReferenceMetrics;

  // Computed overview metrics (from active config)
  overview: DerivedOverviewMetrics;

  // Actions
  setModel: (partial: Partial<ModelArchitectureConfig>) => void;
  setMoe: (partial: Partial<MoeConfig>) => void;
  setTraining: (partial: Partial<TrainingConfig>) => void;
  setInference: (partial: Partial<InferenceConfig>) => void;
  setPreset: (preset: PresetId) => void;
  run: () => void;
  recomputeOverview: () => void;
}

// Helpers
const B = 1e9;

function computePerLayerAttentionParams(
  dModel: number,
  dHead: number,
  nHeads: number,
  nKvHeads: number
) {
  // Q, K, V, O projections with GQA adjustment as per spec.
  const qProj = dModel * (dHead * nHeads);
  const kProj = dModel * (dHead * nKvHeads);
  const vProj = dModel * (dHead * nKvHeads);
  const oProj = (dHead * nHeads) * dModel;
  return qProj + kProj + vProj + oProj;
}

/** MLA attention params per layer from actual structure (DeepSeek). */
function computePerLayerMLAAttentionParams(
  dModel: number,
  nHeads: number,
  qLoraRank: number,
  kvLoraRank: number,
  qkRopeHeadDim: number,
  qkNopeHeadDim: number,
  vHeadDim: number
) {
  const qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
  // q_a: d_model -> q_lora_rank; q_b: q_lora_rank -> n_heads * q_head_dim
  const qA = dModel * qLoraRank;
  const qB = qLoraRank * (nHeads * qHeadDim);
  // kv_a: d_model -> kv_lora_rank + qk_rope_head_dim (MQA for K rotation)
  const kvA = dModel * (kvLoraRank + qkRopeHeadDim);
  // kv_b: kv_lora_rank -> n_heads * (qk_nope + v_head_dim)
  const kvB = kvLoraRank * (nHeads * (qkNopeHeadDim + vHeadDim));
  // o: n_heads * v_head_dim -> d_model
  const oProj = (nHeads * vHeadDim) * dModel;
  return qA + qB + kvA + kvB + oProj;
}

function computePerExpertFfnParams(dModel: number, dFf: number) {
  // Assume SwiGLU: gate + up + down = 3 * d_model * d_ff
  return 3 * dModel * dFf;
}

export function computeOverview(
  model: ModelArchitectureConfig,
  moe: MoeConfig
): DerivedOverviewMetrics {
  const { dModel, layers, nHeads, nKvHeads, dHead, vocabSize } = model;
  const { numExperts, numSharedExperts, topK, dFf, dSharedFf } = moe;
  const numDenseLayers = model.firstKDenseReplace ?? 0;
  const moeLayers = model.layersMoe ?? Math.max(0, layers - numDenseLayers);
  const denseDff = model.denseIntermediateSize ?? dFf;

  const embeddingParams = vocabSize * dModel;
  const hasMLAParams =
    model.useMLA &&
    (model.mlaQLoraRank ?? 0) > 0 &&
    (model.mlaKvLoraRank ?? 0) > 0 &&
    (model.mlaQkRopeHeadDim ?? 0) > 0 &&
    (model.mlaQkNopeHeadDim ?? 0) > 0 &&
    (model.mlaVHeadDim ?? 0) > 0;
  const perLayerAttention = hasMLAParams
    ? computePerLayerMLAAttentionParams(
        dModel,
        nHeads,
        model.mlaQLoraRank!,
        model.mlaKvLoraRank!,
        model.mlaQkRopeHeadDim!,
        model.mlaQkNopeHeadDim!,
        model.mlaVHeadDim!
      )
    : computePerLayerAttentionParams(dModel, dHead, nHeads, nKvHeads);
  const attentionParams = perLayerAttention * layers;

  const perExpert = computePerExpertFfnParams(dModel, dFf);
  const perShared = computePerExpertFfnParams(dModel, dSharedFf);
  const expertParams = moeLayers * numExperts * perExpert;
  const sharedExpertParams = moeLayers * numSharedExperts * perShared;

  // Dense FFN layers (non-MoE, e.g. first K layers in DeepSeek-V3).
  const denseParams =
    numDenseLayers > 0
      ? numDenseLayers * computePerExpertFfnParams(dModel, denseDff)
      : 0;

  // Parallel residual MLP per MoE layer (Arctic: dense + MoE hybrid).
  const residualMLPDff =
    model.residualMLPIntermediateSize ?? model.denseIntermediateSize ?? dFf;
  const residualMLPParams =
    model.useResidualMLP === true
      ? moeLayers * computePerExpertFfnParams(dModel, residualMLPDff)
      : 0;

  const outputHeadParams =
    model.tieWordEmbeddings === true ? 0 : vocabSize * dModel;

  // RMSNorm / LayerNorm parameters: assume two norms per layer (pre-attn, pre-ffn).
  const normParamsPerLayer = 2 * dModel;
  const normParams = layers * normParamsPerLayer;

  // Gating parameters: linear router d_model -> E in each MoE layer.
  const gatingParamsPerLayer = dModel * numExperts;
  const gatingParams = moeLayers * gatingParamsPerLayer;

  const totalParams =
    embeddingParams +
    attentionParams +
    expertParams +
    sharedExpertParams +
    denseParams +
    residualMLPParams +
    outputHeadParams +
    normParams +
    gatingParams;

  const activeExpertParams = moeLayers * topK * perExpert;
  const activeSharedParams = sharedExpertParams; // shared experts always active
  const activeNormParams = normParams; // norms always active
  const activeGatingParams = gatingParams; // gating always active
  // Dense layers are always active.
  const activeDenseParams = denseParams;
  // MLA: full low-rank structure is used in forward; standard: all attention active.
  const activeAttentionParams = attentionParams;

  const activeParams =
    embeddingParams +
    activeAttentionParams +
    activeExpertParams +
    activeSharedParams +
    activeDenseParams +
    residualMLPParams +
    outputHeadParams +
    activeNormParams +
    activeGatingParams;

  return {
    totalParams,
    embeddingParams,
    attentionParams,
    expertParams,
    sharedExpertParams,
    denseParams,
    residualMLPParams,
    gatingParams,
    normParams,
    outputHeadParams,
    activeParams
  };
}

// Hardcoded paper reference metrics (starting set).
const PAPER_PRESETS: Record<PresetId, { config: Partial<ConfigState>; ref?: PaperReferenceMetrics }> =
  {
    "deepseek-v2": {
      config: {
        model: {
          totalParamsB: 236,
          dModel: 5120,
          layers: 60,
          layersMoe: 59,
          firstKDenseReplace: 1,
          denseIntermediateSize: 12288,
          useResidualMLP: false,
          useMLA: true,
          mlaQLoraRank: 1536,
          mlaKvLoraRank: 512,
          mlaQkRopeHeadDim: 64,
          mlaQkNopeHeadDim: 128,
          mlaVHeadDim: 128,
          tieWordEmbeddings: false,
          nHeads: 128,
          nKvHeads: 128,
          dHead: 40,
          vocabSize: 102400,
          maxSeqLen: 16384
        },
        moe: {
          numExperts: 160,
          topK: 6,
          numSharedExperts: 2,
          dFf: 1536,
          dSharedFf: 1536,
          expertGranularity: "fine"
        },
        training: {
          globalBatchTokens: 4096 * 4096,
          totalTrainingTokens: 8e12,
          precision: "bf16",
          optimizer: "adam",
          gradPrecision: "fp32",
          activationCheckpointing: "selective",
          useFlashAttention: true,
          trainingGpuType: "H800-80G",
          mfu: 0.4,
          tp: 8,
          // DeepSeek-V2 paper (Infrastructure): routed experts deployed on D=8 devices.
          ep: 8,
          pp: 4,
          dp: 16
        },
        inference: {
          batchSize: 8,
          inputSeqLen: 8192,
          outputSeqLen: 256,
          precision: "fp16",
          gpuType: "A100-80G",
          // DeepSeek-V2 HF README: BF16 inference requires 80GB × 8 GPUs.
          tp: 8,
          ep: 1,
          pp: 1,
          dp: 1
        }
      },
      ref: {
        name: "DeepSeek-V2",
        totalParamsB: 236,
        activeParamsB: 21,
        kvCachePerTokenKB: 67.5,
        gpuHoursReported: undefined,
        embeddingParamsM: 524.3
      }
    },
    "deepseek-v3": {
      config: {
        model: {
          totalParamsB: 671,
          dModel: 7168,
          layers: 61,
          layersMoe: 58,
          firstKDenseReplace: 3,
          denseIntermediateSize: 18432,
          useResidualMLP: false,
          useMLA: true,
          mlaQLoraRank: 1536,
          mlaKvLoraRank: 512,
          mlaQkRopeHeadDim: 64,
          mlaQkNopeHeadDim: 128,
          mlaVHeadDim: 128,
          tieWordEmbeddings: false,
          nHeads: 128,
          nKvHeads: 128,
          dHead: 128,
          vocabSize: 129280,
          maxSeqLen: 4096
        },
        moe: {
          numExperts: 256,
          topK: 8,
          numSharedExperts: 1,
          dFf: 2048,
          dSharedFf: 2048,
          expertGranularity: "fine"
        },
        training: {
          globalBatchTokens: 4096 * 8192,
          totalTrainingTokens: 14.8e12,
          precision: "fp8",
          optimizer: "adam",
          gradPrecision: "fp32",
          activationCheckpointing: "selective",
          useFlashAttention: true,
          trainingGpuType: "H800-80G",
          mfu: 0.55,
          tp: 8,
          ep: 32,
          pp: 8,
          dp: 1  // Paper: 2048 H800 GPUs = 8×32×8×1
        },
        inference: {
          batchSize: 8,
          inputSeqLen: 4096,
          outputSeqLen: 256,
          precision: "fp16",
          gpuType: "H100-80G",
          tp: 8,
          ep: 4,
          pp: 1,
          dp: 1
        }
      },
      ref: {
        name: "DeepSeek-V3",
        totalParamsB: 671,
        activeParamsB: 37,
        gpuHoursReported: 2.788e6,
        kvCachePerTokenKB: 68.625,
        trainingCostUSD: 5_576_000,
        numGpus: 2048,
        embeddingParamsM: 926.7
      }
    },
    "mixtral-8x7b": {
      config: {
        model: {
          totalParamsB: 46.7,
          dModel: 4096,
          layers: 32,
          layersMoe: 32,
          firstKDenseReplace: 0,
          denseIntermediateSize: undefined,
          useResidualMLP: false,
          useMLA: false,
          tieWordEmbeddings: false,
          nHeads: 32,
          nKvHeads: 8,
          dHead: 128,
          vocabSize: 32000,
          maxSeqLen: 32768
        },
        moe: {
          numExperts: 8,
          topK: 2,
          numSharedExperts: 0,
          dFf: 14336,
          dSharedFf: 14336,
          expertGranularity: "coarse"
        },
        training: {
          globalBatchTokens: 4096 * 1024,
          totalTrainingTokens: 2e12,
          precision: "bf16",
          optimizer: "adam",
          gradPrecision: "fp32",
          activationCheckpointing: "selective",
          useFlashAttention: false,
          trainingGpuType: "H100-80G",
          mfu: 0.35,
          tp: 8,
          ep: 1,
          pp: 4,
          dp: 8
        },
        inference: {
          batchSize: 4,
          inputSeqLen: 8192,
          outputSeqLen: 256,
          precision: "fp16",
          gpuType: "A100-80G",
          tp: 2,
          ep: 1,
          pp: 1,
          dp: 1
        }
      },
      ref: {
        name: "Mixtral 8x7B",
        totalParamsB: 46.7,
        activeParamsB: 12.9,
        inferenceWeightsGB: 94,
        embeddingParamsM: 131.1
      }
    },
    "mixtral-8x22b": {
      config: {
        model: {
          totalParamsB: 141,
          dModel: 6144,
          layers: 56,
          layersMoe: 56,
          firstKDenseReplace: 0,
          denseIntermediateSize: undefined,
          useResidualMLP: false,
          useMLA: false,
          tieWordEmbeddings: false,
          nHeads: 48,
          nKvHeads: 8,
          dHead: 128,
          vocabSize: 32000,
          maxSeqLen: 65536
        },
        moe: {
          numExperts: 8,
          topK: 2,
          numSharedExperts: 0,
          dFf: 16384,
          dSharedFf: 16384,
          expertGranularity: "coarse"
        },
        training: {
          globalBatchTokens: 4096 * 2048,
          totalTrainingTokens: 4e12,
          precision: "bf16",
          optimizer: "adam",
          gradPrecision: "fp32",
          activationCheckpointing: "selective",
          useFlashAttention: true,
          trainingGpuType: "H100-80G",
          mfu: 0.35,
          tp: 8,
          ep: 1,
          pp: 8,
          dp: 16
        },
        inference: {
          batchSize: 4,
          inputSeqLen: 8192,
          outputSeqLen: 256,
          precision: "fp16",
          gpuType: "A100-80G",
          tp: 4,
          ep: 1,
          pp: 1,
          dp: 1
        }
      },
      ref: {
        name: "Mixtral 8x22B",
        totalParamsB: 141,
        activeParamsB: 39,
        inferenceWeightsGB: 283,
        embeddingParamsM: 196.6
      }
    },
    custom: {
      config: {},
      ref: undefined
    }
  };

const DEFAULT_MODEL: ModelArchitectureConfig = {
  totalParamsB: 671,
  dModel: 7168,
  layers: 61,
  layersMoe: 58,
  firstKDenseReplace: 3,
  denseIntermediateSize: 18432,
  useMLA: true,
  mlaQLoraRank: 1536,
  mlaKvLoraRank: 512,
  mlaQkRopeHeadDim: 64,
  mlaQkNopeHeadDim: 128,
  mlaVHeadDim: 128,
  nHeads: 128,
  nKvHeads: 128,
  dHead: 128,
  vocabSize: 129280,
  maxSeqLen: 4096
};

const DEFAULT_MOE: MoeConfig = {
  numExperts: 256,
  topK: 8,
  numSharedExperts: 1,
  dFf: 2048,
  dSharedFf: 2048,
  expertGranularity: "fine"
};

const DEFAULT_TRAINING: TrainingConfig = {
  globalBatchTokens: 4096 * 8192,
  totalTrainingTokens: 14.8e12,
  precision: "fp8",
  optimizer: "adam",
  gradPrecision: "fp32",
  activationCheckpointing: "selective",
  useFlashAttention: true,
  trainingGpuType: "H800-80G",
  mfu: 0.55,
  tp: 8,
  ep: 32,
  pp: 8,
  dp: 1  // DeepSeek-V3: 2048 GPUs
};

const DEFAULT_INFERENCE: InferenceConfig = {
  batchSize: 8,
  inputSeqLen: 4096,
  outputSeqLen: 256,
  precision: "fp16",
  gpuType: "H100-80G",
  tp: 8,
  ep: 4,
  pp: 1,
  dp: 1
};

const DEFAULT_STATE: Omit<
  ConfigState,
  "setModel" | "setMoe" | "setTraining" | "setInference" | "setPreset" | "run" | "recomputeOverview"
> = {
  model: { ...DEFAULT_MODEL },
  moe: { ...DEFAULT_MOE },
  training: { ...DEFAULT_TRAINING },
  inference: { ...DEFAULT_INFERENCE },
  draftModel: { ...DEFAULT_MODEL },
  draftMoe: { ...DEFAULT_MOE },
  draftTraining: { ...DEFAULT_TRAINING },
  draftInference: { ...DEFAULT_INFERENCE },
  preset: "deepseek-v3",
  paperReference: PAPER_PRESETS["deepseek-v3"].ref,
  overview: {
    totalParams: 0,
    embeddingParams: 0,
    attentionParams: 0,
    expertParams: 0,
    sharedExpertParams: 0,
    denseParams: 0,
    residualMLPParams: 0,
    gatingParams: 0,
    normParams: 0,
    outputHeadParams: 0,
    activeParams: 0
  }
};

export const useConfigStore = create<ConfigState>((set, get) => ({
  ...DEFAULT_STATE,
  overview: computeOverview(DEFAULT_STATE.model, DEFAULT_STATE.moe),

  setModel: (partial) => {
    set((state) => ({
      draftModel: { ...state.draftModel, ...partial },
      preset: "custom"
    }));
  },
  setMoe: (partial) => {
    set((state) => ({
      draftMoe: { ...state.draftMoe, ...partial },
      preset: "custom"
    }));
  },
  setTraining: (partial) => {
    set((state) => ({
      draftTraining: { ...state.draftTraining, ...partial },
      preset: "custom"
    }));
  },
  setInference: (partial) => {
    set((state) => ({
      draftInference: { ...state.draftInference, ...partial },
      preset: "custom"
    }));
  },
  setPreset: (presetId) => {
    const preset = PAPER_PRESETS[presetId];
    if (!preset) {
      set({ preset: "custom", paperReference: undefined });
      return;
    }
    set((state) => {
      const model = { ...state.model, ...(preset.config.model ?? {}) };
      const moe = { ...state.moe, ...(preset.config.moe ?? {}) };
      const training = { ...state.training, ...(preset.config.training ?? {}) };
      const inference = { ...state.inference, ...(preset.config.inference ?? {}) };
      const overview = computeOverview(model, moe);
      return {
        model,
        moe,
        training,
        inference,
        draftModel: { ...model },
        draftMoe: { ...moe },
        draftTraining: { ...training },
        draftInference: { ...inference },
        preset: presetId,
        paperReference: preset.ref,
        overview
      };
    });
  },
  run: () => {
    set((state) => {
      const model = { ...state.draftModel };
      const moe = { ...state.draftMoe };
      const training = { ...state.draftTraining };
      const inference = { ...state.draftInference };
      const overview = computeOverview(model, moe);
      return {
        model,
        moe,
        training,
        inference,
        overview
      };
    });
  },
  recomputeOverview: () => {
    const { model, moe } = get();
    set({ overview: computeOverview(model, moe) });
  }
}));

export function formatBigNumber(num: number): string {
  const abs = Math.abs(num);
  if (abs >= 1e12) return (num / 1e12).toFixed(2) + "T";
  if (abs >= 1e9) return (num / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (num / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (num / 1e3).toFixed(2) + "K";
  return num.toFixed(2);
}

