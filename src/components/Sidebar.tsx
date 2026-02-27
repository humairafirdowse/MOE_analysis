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
  "switch-transformer": "Switch Transformer",
  "snowflake-arctic": "Snowflake Arctic",
  custom: "Custom"
};

const gpuOptions: { id: GpuType; label: string }[] = [
  { id: "A100-80G", label: "A100-80G" },
  { id: "H100-80G", label: "H100-80G" },
  { id: "H200-141G", label: "H200-141G" },
  { id: "B200-192G", label: "B200-192G" },
  { id: "RTX-4090", label: "RTX 4090" },
  { id: "M4-Max", label: "Apple M4 Max" }
];

export const Sidebar: React.FC = () => {
  const { model, moe, training, inference, preset } = useConfigStore();
  const { setModel, setMoe, setTraining, setInference, setPreset } = useConfigStore();

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
      </div>

      {/* Model Architecture */}
      <div className="sidebar-section">
        <div className="sidebar-group-title">Model Architecture</div>
        <LabeledInput
          label="Total parameters (B)"
          value={model.totalParamsB}
          onChange={handleNumberChange<typeof model>(setModel, "totalParamsB")}
        />
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Hidden dim (d_model)"
            value={model.dModel}
            onChange={handleNumberChange<typeof model>(setModel, "dModel")}
          />
          <LabeledInput
            label="Layers (L)"
            value={model.layers}
            onChange={handleNumberChange<typeof model>(setModel, "layers")}
          />
        </div>
        <div className="grid grid-cols-3 gap-2">
          <LabeledInput
            label="Heads"
            value={model.nHeads}
            onChange={handleNumberChange<typeof model>(setModel, "nHeads")}
          />
          <LabeledInput
            label="KV heads"
            value={model.nKvHeads}
            onChange={handleNumberChange<typeof model>(setModel, "nKvHeads")}
          />
          <LabeledInput
            label="Head dim"
            value={model.dHead}
            onChange={handleNumberChange<typeof model>(setModel, "dHead")}
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Vocab size (V)"
            value={model.vocabSize}
            onChange={handleNumberChange<typeof model>(setModel, "vocabSize")}
          />
          <LabeledInput
            label="Max seq len (S)"
            value={model.maxSeqLen}
            onChange={handleNumberChange<typeof model>(setModel, "maxSeqLen")}
          />
        </div>
      </div>

      {/* MoE Configuration */}
      <div className="sidebar-section">
        <div className="sidebar-group-title">MoE Configuration</div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Experts (E)"
            value={moe.numExperts}
            onChange={handleNumberChange<typeof moe>(setMoe, "numExperts")}
          />
          <LabeledInput
            label="Top-K (K)"
            value={moe.topK}
            onChange={handleNumberChange<typeof moe>(setMoe, "topK")}
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Shared experts"
            value={moe.numSharedExperts}
            onChange={handleNumberChange<typeof moe>(setMoe, "numSharedExperts")}
          />
          <select
            className="sidebar-select mt-auto"
            value={moe.expertGranularity}
            onChange={(e) =>
              setMoe({ expertGranularity: e.target.value as typeof moe.expertGranularity })
            }
          >
            <option value="fine">Fine-grained</option>
            <option value="coarse">Coarse</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledInput
            label="Expert d_ff"
            value={moe.dFf}
            onChange={handleNumberChange<typeof moe>(setMoe, "dFf")}
          />
          <LabeledInput
            label="Shared d_ff"
            value={moe.dSharedFf}
            onChange={handleNumberChange<typeof moe>(setMoe, "dSharedFf")}
          />
        </div>
      </div>

      {/* Training Configuration */}
      <div className="sidebar-section">
        <div className="sidebar-group-title">Training Configuration</div>
        <LabeledInput
          label="Global batch (tokens)"
          value={training.globalBatchTokens}
          onChange={handleNumberChange<typeof training>(setTraining, "globalBatchTokens")}
        />
        <LabeledInput
          label="Total training tokens (T)"
          value={training.totalTrainingTokens}
          onChange={handleNumberChange<typeof training>(setTraining, "totalTrainingTokens")}
        />
        <div className="grid grid-cols-2 gap-2">
          <LabeledSelect<PrecisionTraining>
            label="Precision"
            value={training.precision}
            options={[
              { id: "fp32", label: "fp32" },
              { id: "bf16", label: "bf16" },
              { id: "fp8", label: "fp8" }
            ]}
            onChange={(value) => setTraining({ precision: value })}
          />
          <LabeledSelect<TrainingOptimizer>
            label="Optimizer"
            value={training.optimizer}
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
            value={training.gradPrecision}
            options={[
              { id: "fp32", label: "fp32" },
              { id: "bf16", label: "bf16" }
            ]}
            onChange={(value) => setTraining({ gradPrecision: value })}
          />
          <LabeledSelect<"none" | "sqrt" | "selective">
            label="Checkpointing"
            value={training.activationCheckpointing}
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
        <label className="mt-1.5 flex items-center gap-2 text-[11px] text-textMuted cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-3 w-3 rounded border-borderSoft bg-card"
            checked={training.useFlashAttention}
            onChange={(e) =>
              setTraining({ useFlashAttention: e.target.checked })
            }
          />
          <span>Use FlashAttention (no attn activations stored)</span>
        </label>
        <div className="grid grid-cols-4 gap-1.5">
          <LabeledInput
            label="TP"
            value={training.tp}
            onChange={handleNumberChange<typeof training>(setTraining, "tp")}
          />
          <LabeledInput
            label="EP"
            value={training.ep}
            onChange={handleNumberChange<typeof training>(setTraining, "ep")}
          />
          <LabeledInput
            label="PP"
            value={training.pp}
            onChange={handleNumberChange<typeof training>(setTraining, "pp")}
          />
          <LabeledInput
            label="DP"
            value={training.dp}
            onChange={handleNumberChange<typeof training>(setTraining, "dp")}
          />
        </div>
      </div>

      {/* Inference Configuration */}
      <div className="sidebar-section mb-6">
        <div className="sidebar-group-title">Inference Configuration</div>
        <div className="grid grid-cols-3 gap-2">
          <LabeledInput
            label="Batch size"
            value={inference.batchSize}
            onChange={handleNumberChange<typeof inference>(setInference, "batchSize")}
          />
          <LabeledInput
            label="Input len"
            value={inference.inputSeqLen}
            onChange={handleNumberChange<typeof inference>(setInference, "inputSeqLen")}
          />
          <LabeledInput
            label="Output len"
            value={inference.outputSeqLen}
            onChange={handleNumberChange<typeof inference>(setInference, "outputSeqLen")}
          />
        </div>
        <div className="grid grid-cols-2 gap-2 mt-1.5">
          <LabeledSelect<PrecisionInference>
            label="Precision"
            value={inference.precision}
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
            value={inference.gpuType}
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

