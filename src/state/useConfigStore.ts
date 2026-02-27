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
  tp: number;
  ep: number;
  pp: number;
  dp: number;
}

export type GpuType =
  | "A100-80G"
  | "H100-80G"
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
}

export type PresetId =
  | "deepseek-v2"
  | "deepseek-v3"
  | "mixtral-8x7b"
  | "mixtral-8x22b"
  | "switch-transformer"
  | "snowflake-arctic"
  | "custom";

export interface PaperReferenceMetrics {
  name: string;
  totalParamsB?: number;
  activeParamsB?: number;
  trainingFlopsPerTokenTF?: number;
  totalTrainingFlopsPF?: number;
  trainingMemoryGB?: number;
  kvCachePerTokenKB?: number;
  gpuHoursReported?: number;
}

export interface DerivedOverviewMetrics {
  totalParams: number;
  embeddingParams: number;
  attentionParams: number;
  expertParams: number;
  sharedExpertParams: number;
  outputHeadParams: number;
  activeParams: number;
}

export interface ConfigState {
  // Raw configs
  model: ModelArchitectureConfig;
  moe: MoeConfig;
  training: TrainingConfig;
  inference: InferenceConfig;
  preset: PresetId;

  // Paper reference for validation tab
  paperReference?: PaperReferenceMetrics;

  // Computed overview metrics
  overview: DerivedOverviewMetrics;

  // Actions
  setModel: (partial: Partial<ModelArchitectureConfig>) => void;
  setMoe: (partial: Partial<MoeConfig>) => void;
  setTraining: (partial: Partial<TrainingConfig>) => void;
  setInference: (partial: Partial<InferenceConfig>) => void;
  setPreset: (preset: PresetId) => void;
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

function computePerExpertFfnParams(dModel: number, dFf: number) {
  // Assume SwiGLU: gate + up + down = 3 * d_model * d_ff
  return 3 * dModel * dFf;
}

function computeOverview(
  model: ModelArchitectureConfig,
  moe: MoeConfig
): DerivedOverviewMetrics {
  const { dModel, layers, nHeads, nKvHeads, dHead, vocabSize } = model;
  const { numExperts, numSharedExperts, topK, dFf, dSharedFf } = moe;

  const embeddingParams = vocabSize * dModel;
  const perLayerAttention = computePerLayerAttentionParams(
    dModel,
    dHead,
    nHeads,
    nKvHeads
  );
  const attentionParams = perLayerAttention * layers;

  const perExpert = computePerExpertFfnParams(dModel, dFf);
  const perShared = computePerExpertFfnParams(dModel, dSharedFf);

  // Assume every layer is a MoE layer for now (L_moe = L).
  const moeLayers = layers;
  const expertParams = moeLayers * numExperts * perExpert;
  const sharedExpertParams = moeLayers * numSharedExperts * perShared;

  const outputHeadParams = vocabSize * dModel; // separate from embedding even if tied

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
    outputHeadParams +
    normParams +
    gatingParams;

  const activeExpertParams = moeLayers * topK * perExpert;
  const activeSharedParams = sharedExpertParams; // shared experts always active
  const activeNormParams = normParams; // norms always active
  const activeGatingParams = gatingParams; // gating always active

  const activeParams =
    embeddingParams +
    attentionParams +
    activeExpertParams +
    activeSharedParams +
    outputHeadParams +
    activeNormParams +
    activeGatingParams;

  return {
    totalParams,
    embeddingParams,
    attentionParams,
    expertParams,
    sharedExpertParams,
    outputHeadParams,
    activeParams
  };
}

// Hardcoded paper reference metrics (starting set).
const PAPER_PRESETS: Record<PresetId, { config: Partial<ConfigState>; ref?: PaperReferenceMetrics }> =
  {
    "deepseek-v2": {
      config: {},
      ref: undefined
    },
    "deepseek-v3": {
      config: {
        model: {
          totalParamsB: 671,
          dModel: 7168,
          layers: 61,
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
          tp: 8,
          ep: 32,
          pp: 8,
          dp: 16
        },
        inference: {
          batchSize: 8,
          inputSeqLen: 4096,
          outputSeqLen: 256,
          precision: "fp16",
          gpuType: "H100-80G"
        }
      },
      ref: {
        name: "DeepSeek-V3",
        totalParamsB: 671,
        activeParamsB: 37,
        gpuHoursReported: 2.788e6
      }
    },
    "mixtral-8x7b": {
      config: {
        model: {
          totalParamsB: 46.7,
          dModel: 4096,
          layers: 32,
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
          gpuType: "A100-80G"
        }
      },
      ref: {
        name: "Mixtral 8x7B",
        totalParamsB: 46.7,
        activeParamsB: 12.9
      }
    },
    "mixtral-8x22b": {
      config: {},
      ref: undefined
    },
    "switch-transformer": {
      config: {},
      ref: undefined
    },
    "snowflake-arctic": {
      config: {},
      ref: undefined
    },
    custom: {
      config: {},
      ref: undefined
    }
  };

const DEFAULT_STATE: Omit<ConfigState, "setModel" | "setMoe" | "setTraining" | "setInference" | "setPreset" | "recomputeOverview"> =
  {
    model: {
      totalParamsB: 671,
      dModel: 7168,
      layers: 61,
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
      tp: 8,
      ep: 32,
      pp: 8,
      dp: 16
    },
    inference: {
      batchSize: 8,
      inputSeqLen: 4096,
      outputSeqLen: 256,
      precision: "fp16",
      gpuType: "H100-80G"
    },
    preset: "deepseek-v3",
    paperReference: PAPER_PRESETS["deepseek-v3"].ref,
    overview: {
      totalParams: 0,
      embeddingParams: 0,
      attentionParams: 0,
      expertParams: 0,
      sharedExpertParams: 0,
      outputHeadParams: 0,
      activeParams: 0
    }
  };

export const useConfigStore = create<ConfigState>((set, get) => ({
  ...DEFAULT_STATE,
  overview: computeOverview(DEFAULT_STATE.model, DEFAULT_STATE.moe),

  setModel: (partial) => {
    set((state) => {
      const model = { ...state.model, ...partial };
      return { model, overview: computeOverview(model, state.moe), preset: "custom" };
    });
  },
  setMoe: (partial) => {
    set((state) => {
      const moe = { ...state.moe, ...partial };
      return { moe, overview: computeOverview(state.model, moe), preset: "custom" };
    });
  },
  setTraining: (partial) => {
    set((state) => ({
      training: { ...state.training, ...partial },
      preset: "custom"
    }));
  },
  setInference: (partial) => {
    set((state) => ({
      inference: { ...state.inference, ...partial },
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
        preset: presetId,
        paperReference: preset.ref,
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

