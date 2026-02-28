import React from "react";
import {
  GpuType,
  PrecisionInference,
  PrecisionTraining,
  PresetId,
  TrainingOptimizer,
  useConfigStore
} from "../state/useConfigStore";

const PRESET_LABELS: Record<PresetId, string> = {
  "deepseek-v2": "DeepSeek-V2",
  "deepseek-v3": "DeepSeek-V3",
  "mixtral-8x7b": "Mixtral 8x7B",
  "mixtral-8x22b": "Mixtral 8x22B",
  "snowflake-arctic": "Snowflake Arctic",
  custom: "Custom"
};

const gpuOptions: { id: GpuType; label: string }[] = [
  { id: "A100-80G", label: "A100-80G" },
  { id: "H100-80G", label: "H100-80G" },
  { id: "H800-80G", label: "H800-80G" },
  { id: "H200-141G", label: "H200-141G" },
  { id: "B200-192G", label: "B200-192G" },
  { id: "RTX-4090", label: "RTX 4090" },
  { id: "M4-Max", label: "Apple M4 Max" }
];

export const Sidebar: React.FC = () => {
  const { draftModel, draftMoe, draftTraining, draftInference, preset } =
    useConfigStore();
  const { setModel, setMoe, setTraining, setInference, setPreset, run } =
    useConfigStore();

  const handleNumberChange =
    <T extends object>(setter: (partial: Partial<T>) => void, key: keyof T) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      const numeric = value === "" ? 0 : Number(value);
      if (Number.isNaN(numeric)) return;
      setter({ [key]: numeric } as Partial<T>);
    };

  return (
    <aside className="w-[320px] shrink-0 h-screen bg-sidebar border-r border-borderSoft/80 px-4 py-3 flex flex-col gap-3 overflow-y-auto">
      <div className="flex items-center justify-between mb-1">
        <div>
          <div className="text-[11px] uppercase tracking-[0.2em] text-textMuted">
            MoE Analysis
          </div>
          <div className="text-sm font-semibold">Global Configuration</div>
        </div>
      </div>

      {/* Preset Loader */}
      <div className="sidebar-section">
        <div className="sidebar-label flex items-center justify-between">
          <span>Preset</span>
        </div>
        <select
          className="sidebar-select"
          value={preset}
          onChange={(e) => setPreset(e.target.value as PresetId)}
        >
          {Object.entries(PRESET_LABELS).map(([id, label]) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={run}
          className="mt-2 w-full py-2 px-3 rounded-md bg-accent hover:bg-accent/90 text-accent-foreground text-sm font-medium transition-colors"
        >
          Run
        </button>
      </div>

      {/* Model Architecture */}
      <div className="sidebar-section">
        <div className="sidebar-group-title">Model Architecture</div>
        <LabeledInput
          label="Total parameters (B)"
          value={draftModel.totalParamsB}
          onChange={handleNumberChange<typeof draftModel>(setModel, "totalParamsB")}
        />
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Hidden dim (d_model)"
            value={draftModel.dModel}
            onChange={handleNumberChange<typeof draftModel>(setModel, "dModel")}
          />
          <LabeledInput
            label="Layers (L)"
            value={draftModel.layers}
            onChange={handleNumberChange<typeof draftModel>(setModel, "layers")}
          />
          <LabeledInput
            label="MoE layers (L_moe)"
            value={draftModel.layersMoe ?? draftModel.layers}
            onChange={(e) => {
              const v = e.target.value;
              const n = v === "" ? undefined : Number(v);
              if (n !== undefined && Number.isNaN(n)) return;
              setModel({ layersMoe: n });
            }}
          />
          <LabeledInput
            label="First K dense layers"
            value={draftModel.firstKDenseReplace ?? 0}
            onChange={(e) => {
              const v = e.target.value;
              const n = v === "" ? 0 : Number(v);
              if (Number.isNaN(n)) return;
              setModel({ firstKDenseReplace: n > 0 ? n : undefined });
            }}
          />
          <LabeledInput
            label="Dense d_ff"
            value={draftModel.denseIntermediateSize ?? draftMoe.dFf}
            onChange={(e) => {
              const v = e.target.value;
              const n = v === "" ? undefined : Number(v);
              if (n !== undefined && Number.isNaN(n)) return;
              setModel({ denseIntermediateSize: n });
            }}
          />
        </div>
        <div className="mt-1.5 space-y-1.5">
          <label className="flex items-center gap-2 text-[11px] text-textMuted cursor-pointer select-none">
            <input
              type="checkbox"
              className="h-3 w-3 rounded border-borderSoft bg-card"
              checked={draftModel.useMLA ?? false}
              onChange={(e) => setModel({ useMLA: e.target.checked })}
            />
            <span>MLA (Multi-head Latent Attention) — DeepSeek-V2/V3</span>
          </label>
          {draftModel.useMLA && (
            <div className="grid grid-cols-2 gap-2 pl-5 border-l-2 border-borderSoft/50">
              <LabeledInput
                label="Q LoRA rank"
                value={draftModel.mlaQLoraRank ?? 0}
                onChange={(e) => {
                  const v = e.target.value;
                  const n = v === "" ? undefined : Number(v);
                  if (n !== undefined && Number.isNaN(n)) return;
                  setModel({ mlaQLoraRank: n });
                }}
              />
              <LabeledInput
                label="KV LoRA rank"
                value={draftModel.mlaKvLoraRank ?? 0}
                onChange={(e) => {
                  const v = e.target.value;
                  const n = v === "" ? undefined : Number(v);
                  if (n !== undefined && Number.isNaN(n)) return;
                  setModel({ mlaKvLoraRank: n });
                }}
              />
              <LabeledInput
                label="QK RoPE head dim"
                value={draftModel.mlaQkRopeHeadDim ?? 0}
                onChange={(e) => {
                  const v = e.target.value;
                  const n = v === "" ? undefined : Number(v);
                  if (n !== undefined && Number.isNaN(n)) return;
                  setModel({ mlaQkRopeHeadDim: n });
                }}
              />
              <LabeledInput
                label="QK Nope head dim"
                value={draftModel.mlaQkNopeHeadDim ?? 0}
                onChange={(e) => {
                  const v = e.target.value;
                  const n = v === "" ? undefined : Number(v);
                  if (n !== undefined && Number.isNaN(n)) return;
                  setModel({ mlaQkNopeHeadDim: n });
                }}
              />
              <LabeledInput
                label="V head dim"
                value={draftModel.mlaVHeadDim ?? 0}
                onChange={(e) => {
                  const v = e.target.value;
                  const n = v === "" ? undefined : Number(v);
                  if (n !== undefined && Number.isNaN(n)) return;
                  setModel({ mlaVHeadDim: n });
                }}
              />
            </div>
          )}
          <label className="flex items-center gap-2 text-[11px] text-textMuted cursor-pointer select-none">
            <input
              type="checkbox"
              className="h-3 w-3 rounded border-borderSoft bg-card"
              checked={draftModel.tieWordEmbeddings ?? false}
              onChange={(e) =>
                setModel({ tieWordEmbeddings: e.target.checked })
              }
            />
            <span>Tied embeddings & output head</span>
          </label>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <LabeledInput
            label="Heads"
            value={draftModel.nHeads}
            onChange={handleNumberChange<typeof draftModel>(setModel, "nHeads")}
          />
          <LabeledInput
            label="KV heads"
            value={draftModel.nKvHeads}
            onChange={handleNumberChange<typeof draftModel>(setModel, "nKvHeads")}
          />
          <LabeledInput
            label="Head dim"
            value={draftModel.dHead}
            onChange={handleNumberChange<typeof draftModel>(setModel, "dHead")}
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Vocab size (V)"
            value={draftModel.vocabSize}
            onChange={handleNumberChange<typeof draftModel>(setModel, "vocabSize")}
          />
          <LabeledInput
            label="Max seq len (S)"
            value={draftModel.maxSeqLen}
            onChange={handleNumberChange<typeof draftModel>(setModel, "maxSeqLen")}
          />
        </div>
      </div>

      {/* MoE Configuration */}
      <div className="sidebar-section">
        <div className="sidebar-group-title">MoE Configuration</div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Experts (E)"
            value={draftMoe.numExperts}
            onChange={handleNumberChange<typeof draftMoe>(setMoe, "numExperts")}
          />
          <LabeledInput
            label="Top-K (K)"
            value={draftMoe.topK}
            onChange={handleNumberChange<typeof draftMoe>(setMoe, "topK")}
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Shared experts"
            value={draftMoe.numSharedExperts}
            onChange={handleNumberChange<typeof draftMoe>(setMoe, "numSharedExperts")}
          />
          <select
            className="sidebar-select mt-auto"
            value={draftMoe.expertGranularity}
            onChange={(e) =>
              setMoe({ expertGranularity: e.target.value as typeof draftMoe.expertGranularity })
            }
          >
            <option value="fine">Fine-grained</option>
            <option value="coarse">Coarse</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Expert d_ff"
            value={draftMoe.dFf}
            onChange={handleNumberChange<typeof draftMoe>(setMoe, "dFf")}
          />
          <LabeledInput
            label="Shared d_ff"
            value={draftMoe.dSharedFf}
            onChange={handleNumberChange<typeof draftMoe>(setMoe, "dSharedFf")}
          />
        </div>
      </div>

      {/* Training Configuration */}
      <div className="sidebar-section">
        <div className="sidebar-group-title">Training Configuration</div>
        <LabeledInput
          label="Global batch (tokens)"
          value={draftTraining.globalBatchTokens}
          onChange={handleNumberChange<typeof draftTraining>(setTraining, "globalBatchTokens")}
        />
        <LabeledInput
          label="Total training tokens (T)"
          value={draftTraining.totalTrainingTokens}
          onChange={handleNumberChange<typeof draftTraining>(setTraining, "totalTrainingTokens")}
        />
        <div className="grid grid-cols-2 gap-2">
          <LabeledSelect<PrecisionTraining>
            label="Precision"
            value={draftTraining.precision}
            options={[
              { id: "fp32", label: "fp32" },
              { id: "bf16", label: "bf16" },
              { id: "fp8", label: "fp8" }
            ]}
            onChange={(value) => setTraining({ precision: value })}
          />
          <LabeledSelect<TrainingOptimizer>
            label="Optimizer"
            value={draftTraining.optimizer}
            options={[
              { id: "adam", label: "Adam" },
              { id: "adafactor", label: "AdaFactor" }
            ]}
            onChange={(value) => setTraining({ optimizer: value })}
          />
        </div>
        <div className="grid grid-cols-2 gap-2 mt-1.5">
          <LabeledSelect<"fp32" | "bf16">
            label="Grad precision"
            value={draftTraining.gradPrecision}
            options={[
              { id: "fp32", label: "fp32" },
              { id: "bf16", label: "bf16" }
            ]}
            onChange={(value) => setTraining({ gradPrecision: value })}
          />
          <LabeledSelect<"none" | "sqrt" | "selective">
            label="Checkpointing"
            value={draftTraining.activationCheckpointing}
            options={[
              { id: "none", label: "None" },
              { id: "sqrt", label: "Uniform (√L)" },
              { id: "selective", label: "Selective" }
            ]}
            onChange={(value) =>
              setTraining({ activationCheckpointing: value })
            }
          />
        </div>
        <div className="grid grid-cols-2 gap-2 mt-1.5">
          <LabeledSelect<GpuType>
            label="Training GPU (Fgpu)"
            value={draftTraining.trainingGpuType ?? "H800-80G"}
            options={gpuOptions}
            onChange={(value) => setTraining({ trainingGpuType: value })}
          />
          <label className="block">
            <div className="sidebar-label">MFU (U) — {((draftTraining.mfu ?? 0.55) * 100).toFixed(0)}%</div>
            <input
              type="number"
              className="sidebar-input"
              min={0.2}
              max={0.7}
              step={0.01}
              value={draftTraining.mfu ?? 0.55}
              onChange={(e) => {
                const v = e.target.value;
                const n = v === "" ? 0.55 : Number(v);
                if (Number.isNaN(n)) return;
                setTraining({ mfu: Math.max(0.2, Math.min(0.7, n)) });
              }}
            />
          </label>
        </div>
        <label className="mt-1.5 flex items-center gap-2 text-[11px] text-textMuted cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-3 w-3 rounded border-borderSoft bg-card"
            checked={draftTraining.useFlashAttention}
            onChange={(e) =>
              setTraining({ useFlashAttention: e.target.checked })
            }
          />
          <span>Use FlashAttention (no attn activations stored)</span>
        </label>
        <div className="grid grid-cols-4 gap-1.5">
          <LabeledInput
            label="TP"
            value={draftTraining.tp}
            onChange={handleNumberChange<typeof draftTraining>(setTraining, "tp")}
          />
          <LabeledInput
            label="EP"
            value={draftTraining.ep}
            onChange={handleNumberChange<typeof draftTraining>(setTraining, "ep")}
          />
          <LabeledInput
            label="PP"
            value={draftTraining.pp}
            onChange={handleNumberChange<typeof draftTraining>(setTraining, "pp")}
          />
          <LabeledInput
            label="DP"
            value={draftTraining.dp}
            onChange={handleNumberChange<typeof draftTraining>(setTraining, "dp")}
          />
        </div>
      </div>

      {/* Inference Configuration */}
      <div className="sidebar-section mb-6">
        <div className="sidebar-group-title">Inference Configuration</div>
        <div className="grid grid-cols-3 gap-2">
          <LabeledInput
            label="Batch size"
            value={draftInference.batchSize}
            onChange={handleNumberChange<typeof draftInference>(setInference, "batchSize")}
          />
          <LabeledInput
            label="Input len"
            value={draftInference.inputSeqLen}
            onChange={handleNumberChange<typeof draftInference>(setInference, "inputSeqLen")}
          />
          <LabeledInput
            label="Output len"
            value={draftInference.outputSeqLen}
            onChange={handleNumberChange<typeof draftInference>(setInference, "outputSeqLen")}
          />
        </div>
        <div className="grid grid-cols-2 gap-2 mt-1.5">
          <LabeledSelect<PrecisionInference>
            label="Precision"
            value={draftInference.precision}
            options={[
              { id: "fp16", label: "fp16" },
              { id: "bf16", label: "bf16" },
              { id: "fp8", label: "fp8" },
              { id: "int8", label: "int8" },
              { id: "int4", label: "int4" }
            ]}
            onChange={(value) => setInference({ precision: value })}
          />
          <LabeledSelect<GpuType>
            label="GPU type"
            value={draftInference.gpuType}
            options={gpuOptions}
            onChange={(value) => setInference({ gpuType: value })}
          />
        </div>
      </div>
    </aside>
  );
};

interface LabeledInputProps {
  label: string;
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const LabeledInput: React.FC<LabeledInputProps> = ({ label, value, onChange }) => (
  <label className="block">
    <div className="sidebar-label">{label}</div>
    <input
      type="number"
      className="sidebar-input"
      value={Number.isFinite(value) ? value : 0}
      onChange={onChange}
    />
  </label>
);

interface LabeledSelectOption<T extends string> {
  id: T;
  label: string;
}

interface LabeledSelectProps<T extends string> {
  label: string;
  value: T;
  options: LabeledSelectOption<T>[];
  onChange: (value: T) => void;
}

const LabeledSelect = <T extends string>({
  label,
  value,
  options,
  onChange
}: LabeledSelectProps<T>) => (
  <label className="block">
    <div className="sidebar-label">{label}</div>
    <select
      className="sidebar-select"
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {options.map((opt) => (
        <option key={opt.id} value={opt.id}>
          {opt.label}
        </option>
      ))}
    </select>
  </label>
);

