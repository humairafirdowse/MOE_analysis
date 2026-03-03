import React, { useState } from "react";
import {
  AdamMomentPrecision,
  GpuType,
  PrecisionInference,
  PrecisionTraining,
  PresetId,
  TrainingOptimizer,
  ZeroStage,
  useConfigStore
} from "../state/useConfigStore";

const PRESET_LABELS: Record<PresetId, string> = {
  "deepseek-v2": "DeepSeek-V2",
  "deepseek-v3": "DeepSeek-V3",
  "mixtral-8x7b": "Mixtral 8x7B",
  "mixtral-8x22b": "Mixtral 8x22B",
  custom: "Custom"
};

const gpuOptions: { id: GpuType; label: string }[] = [
  { id: "A100-80G", label: "A100 80G" },
  { id: "H100-80G", label: "H100 80G" },
  { id: "H800-80G", label: "H800 80G" },
  { id: "H200-141G", label: "H200 141G" },
  { id: "B200-192G", label: "B200 192G" },
  { id: "RTX-4090", label: "RTX 4090" },
  { id: "M4-Max", label: "M4 Max" }
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
    <aside className="w-[300px] shrink-0 h-screen bg-sidebar border-r border-borderSoft/40 flex flex-col overflow-hidden">
      {/* ─── Header + Preset ─── */}
      <div className="px-4 pt-4 pb-3 border-b border-borderSoft/30">
        <div className="text-[10px] uppercase tracking-[0.25em] text-textMuted/60 font-medium">
          MoE Analysis
        </div>
        <div className="text-sm font-bold mt-0.5 tracking-tight">Configuration</div>

        <div className="mt-3 space-y-2">
          <select
            className="sidebar-select text-[13px] font-semibold py-2"
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
            className="w-full py-2 px-3 rounded-lg bg-accent hover:bg-accent/90
                       text-[#050816] text-xs font-bold uppercase tracking-wider
                       transition-colors"
          >
            Apply Configuration
          </button>
        </div>
      </div>

      {/* ─── Scrollable sections ─── */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {/* ── Model Architecture ── */}
        <CollapsibleGroup
          label="Model Architecture"
          color="bg-sectionArch"
          defaultOpen
        >
          <Row2>
            <LabeledInput
              label="d_model"
              value={draftModel.dModel}
              onChange={handleNumberChange<typeof draftModel>(setModel, "dModel")}
            />
            <LabeledInput
              label="Layers (L)"
              value={draftModel.layers}
              onChange={handleNumberChange<typeof draftModel>(setModel, "layers")}
            />
          </Row2>
          <Row2>
            <LabeledInput
              label="MoE layers"
              value={draftModel.layersMoe ?? draftModel.layers}
              onChange={(e) => {
                const v = e.target.value;
                const n = v === "" ? undefined : Number(v);
                if (n !== undefined && Number.isNaN(n)) return;
                setModel({ layersMoe: n });
              }}
            />
            <LabeledInput
              label="Dense first-K"
              value={draftModel.firstKDenseReplace ?? 0}
              onChange={(e) => {
                const v = e.target.value;
                const n = v === "" ? 0 : Number(v);
                if (Number.isNaN(n)) return;
                setModel({ firstKDenseReplace: n > 0 ? n : undefined });
              }}
            />
          </Row2>
          <LabeledInput
            label="Dense FFN d_ff"
            value={draftModel.denseIntermediateSize ?? draftMoe.dFf}
            onChange={(e) => {
              const v = e.target.value;
              const n = v === "" ? undefined : Number(v);
              if (n !== undefined && Number.isNaN(n)) return;
              setModel({ denseIntermediateSize: n });
            }}
          />

          <Row3>
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
          </Row3>
          <Row2>
            <LabeledInput
              label="Vocab (V)"
              value={draftModel.vocabSize}
              onChange={handleNumberChange<typeof draftModel>(setModel, "vocabSize")}
            />
            <LabeledInput
              label="Max seq len"
              value={draftModel.maxSeqLen}
              onChange={handleNumberChange<typeof draftModel>(setModel, "maxSeqLen")}
            />
          </Row2>

          <Checkbox
            label="Tied embeddings & output head"
            checked={draftModel.tieWordEmbeddings ?? false}
            onChange={(c) => setModel({ tieWordEmbeddings: c })}
          />
        </CollapsibleGroup>

        {/* ── MLA (nested under Arch) ── */}
        <CollapsibleGroup
          label="MLA Attention"
          color="bg-sectionArch/60"
          tag="DeepSeek"
          defaultOpen={!!draftModel.useMLA}
        >
          <Checkbox
            label="Enable Multi-head Latent Attention"
            checked={draftModel.useMLA ?? false}
            onChange={(c) => setModel({ useMLA: c })}
          />
          {draftModel.useMLA && (
            <>
              <Row2>
                <LabeledInput
                  label="Q LoRA rank"
                  value={draftModel.mlaQLoraRank ?? 0}
                  onChange={(e) => {
                    const n = e.target.value === "" ? undefined : Number(e.target.value);
                    if (n !== undefined && Number.isNaN(n)) return;
                    setModel({ mlaQLoraRank: n });
                  }}
                />
                <LabeledInput
                  label="KV LoRA rank"
                  value={draftModel.mlaKvLoraRank ?? 0}
                  onChange={(e) => {
                    const n = e.target.value === "" ? undefined : Number(e.target.value);
                    if (n !== undefined && Number.isNaN(n)) return;
                    setModel({ mlaKvLoraRank: n });
                  }}
                />
              </Row2>
              <Row3>
                <LabeledInput
                  label="RoPE dim"
                  value={draftModel.mlaQkRopeHeadDim ?? 0}
                  onChange={(e) => {
                    const n = e.target.value === "" ? undefined : Number(e.target.value);
                    if (n !== undefined && Number.isNaN(n)) return;
                    setModel({ mlaQkRopeHeadDim: n });
                  }}
                />
                <LabeledInput
                  label="Nope dim"
                  value={draftModel.mlaQkNopeHeadDim ?? 0}
                  onChange={(e) => {
                    const n = e.target.value === "" ? undefined : Number(e.target.value);
                    if (n !== undefined && Number.isNaN(n)) return;
                    setModel({ mlaQkNopeHeadDim: n });
                  }}
                />
                <LabeledInput
                  label="V dim"
                  value={draftModel.mlaVHeadDim ?? 0}
                  onChange={(e) => {
                    const n = e.target.value === "" ? undefined : Number(e.target.value);
                    if (n !== undefined && Number.isNaN(n)) return;
                    setModel({ mlaVHeadDim: n });
                  }}
                />
              </Row3>
            </>
          )}
        </CollapsibleGroup>

        {/* ── MoE Configuration ── */}
        <CollapsibleGroup
          label="MoE Experts"
          color="bg-sectionMoe"
          defaultOpen
        >
          <Row2>
            <LabeledInput
              label="Experts (E)"
              value={draftMoe.numExperts}
              onChange={handleNumberChange<typeof draftMoe>(setMoe, "numExperts")}
            />
            <LabeledInput
              label="Top-K"
              value={draftMoe.topK}
              onChange={handleNumberChange<typeof draftMoe>(setMoe, "topK")}
            />
          </Row2>
          <Row2>
            <LabeledInput
              label="Shared experts"
              value={draftMoe.numSharedExperts}
              onChange={handleNumberChange<typeof draftMoe>(setMoe, "numSharedExperts")}
            />
            <div>
              <div className="sidebar-label">Granularity</div>
              <select
                className="sidebar-select"
                value={draftMoe.expertGranularity}
                onChange={(e) =>
                  setMoe({
                    expertGranularity: e.target.value as typeof draftMoe.expertGranularity
                  })
                }
              >
                <option value="fine">Fine-grained</option>
                <option value="coarse">Coarse</option>
              </select>
            </div>
          </Row2>
          <Row2>
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
          </Row2>
        </CollapsibleGroup>

        {/* ── Training Configuration ── */}
        <CollapsibleGroup
          label="Training"
          color="bg-sectionTrain"
          defaultOpen={false}
        >
          <LabeledInput
            label="Global batch (tokens)"
            value={draftTraining.globalBatchTokens}
            onChange={handleNumberChange<typeof draftTraining>(
              setTraining,
              "globalBatchTokens"
            )}
          />
          <LabeledInput
            label="Total training tokens (T)"
            value={draftTraining.totalTrainingTokens}
            onChange={handleNumberChange<typeof draftTraining>(
              setTraining,
              "totalTrainingTokens"
            )}
          />

          <Row2>
            <LabeledSelect<PrecisionTraining>
              label="Precision"
              value={draftTraining.precision}
              options={[
                { id: "fp32", label: "FP32" },
                { id: "bf16", label: "BF16" },
                { id: "fp8", label: "FP8" }
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
          </Row2>
          <Row2>
            <LabeledSelect<"fp32" | "bf16">
              label="Grad precision"
              value={draftTraining.gradPrecision}
              options={[
                { id: "fp32", label: "FP32" },
                { id: "bf16", label: "BF16" }
              ]}
              onChange={(value) => setTraining({ gradPrecision: value })}
            />
            <LabeledSelect<AdamMomentPrecision>
              label="Adam m/v prec"
              value={draftTraining.adamMomentPrecision}
              options={[
                { id: "fp32", label: "FP32" },
                { id: "bf16", label: "BF16" }
              ]}
              onChange={(value) => setTraining({ adamMomentPrecision: value })}
            />
          </Row2>
          <Row2>
            <LabeledSelect<ZeroStage>
              label="ZeRO stage"
              value={draftTraining.zeroStage}
              options={[
                { id: 0, label: "0 – None" },
                { id: 1, label: "1 – Opt states" },
                { id: 2, label: "2 – + Grads" },
                { id: 3, label: "3 – + Params" }
              ]}
              onChange={(value) => setTraining({ zeroStage: value })}
            />
            <LabeledInput
              label="Micro-batch seqs"
              value={draftTraining.microBatchSeqCount}
              onChange={handleNumberChange<typeof draftTraining>(
                setTraining,
                "microBatchSeqCount"
              )}
            />
          </Row2>
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

          <Checkbox
            label="FlashAttention (skip attn activation storage)"
            checked={draftTraining.useFlashAttention}
            onChange={(c) => setTraining({ useFlashAttention: c })}
          />

          <div className="pt-1 border-t border-borderSoft/20">
            <div className="text-[9px] uppercase tracking-widest text-textMuted/50 font-bold mb-2 mt-1">
              Hardware & Parallelism
            </div>
            <Row2>
              <LabeledSelect<GpuType>
                label="GPU"
                value={draftTraining.trainingGpuType ?? "H800-80G"}
                options={gpuOptions}
                onChange={(value) => setTraining({ trainingGpuType: value })}
              />
              <div>
                <div className="sidebar-label">
                  MFU — {((draftTraining.mfu ?? 0.55) * 100).toFixed(0)}%
                </div>
                <input
                  type="number"
                  className="sidebar-input"
                  min={0.05}
                  max={0.7}
                  step={0.01}
                  value={draftTraining.mfu ?? 0.55}
                  onChange={(e) => {
                    const v = e.target.value;
                    const n = v === "" ? 0.55 : Number(v);
                    if (Number.isNaN(n)) return;
                    setTraining({ mfu: Math.max(0.05, Math.min(0.7, n)) });
                  }}
                />
              </div>
            </Row2>
            <div className="grid grid-cols-4 gap-1.5">
              <LabeledInput
                label="TP"
                value={draftTraining.tp}
                onChange={handleNumberChange<typeof draftTraining>(
                  setTraining,
                  "tp"
                )}
              />
              <LabeledInput
                label="EP"
                value={draftTraining.ep}
                onChange={handleNumberChange<typeof draftTraining>(
                  setTraining,
                  "ep"
                )}
              />
              <LabeledInput
                label="PP"
                value={draftTraining.pp}
                onChange={handleNumberChange<typeof draftTraining>(
                  setTraining,
                  "pp"
                )}
              />
              <LabeledInput
                label="DP"
                value={draftTraining.dp}
                onChange={handleNumberChange<typeof draftTraining>(
                  setTraining,
                  "dp"
                )}
              />
            </div>
          </div>
        </CollapsibleGroup>

        {/* ── Inference Configuration ── */}
        <CollapsibleGroup
          label="Inference"
          color="bg-sectionInfer"
          defaultOpen={false}
        >
          <Row3>
            <LabeledInput
              label="Batch"
              value={draftInference.batchSize}
              onChange={handleNumberChange<typeof draftInference>(
                setInference,
                "batchSize"
              )}
            />
            <LabeledInput
              label="In len"
              value={draftInference.inputSeqLen}
              onChange={handleNumberChange<typeof draftInference>(
                setInference,
                "inputSeqLen"
              )}
            />
            <LabeledInput
              label="Out len"
              value={draftInference.outputSeqLen}
              onChange={handleNumberChange<typeof draftInference>(
                setInference,
                "outputSeqLen"
              )}
            />
          </Row3>
          <Row2>
            <LabeledSelect<PrecisionInference>
              label="Precision"
              value={draftInference.precision}
              options={[
                { id: "fp16", label: "FP16" },
                { id: "bf16", label: "BF16" },
                { id: "fp8", label: "FP8" },
                { id: "int8", label: "INT8" },
                { id: "int4", label: "INT4" }
              ]}
              onChange={(value) => setInference({ precision: value })}
            />
            <LabeledSelect<GpuType>
              label="GPU"
              value={draftInference.gpuType}
              options={gpuOptions}
              onChange={(value) => setInference({ gpuType: value })}
            />
          </Row2>

          <div className="pt-1 border-t border-borderSoft/20">
            <div className="text-[9px] uppercase tracking-widest text-textMuted/50 font-bold mb-2 mt-1">
              Parallelism
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              <LabeledInput
                label="TP"
                value={draftInference.tp}
                onChange={handleNumberChange<typeof draftInference>(
                  setInference,
                  "tp"
                )}
              />
              <LabeledInput
                label="EP"
                value={draftInference.ep}
                onChange={handleNumberChange<typeof draftInference>(
                  setInference,
                  "ep"
                )}
              />
              <LabeledInput
                label="PP"
                value={draftInference.pp}
                onChange={handleNumberChange<typeof draftInference>(
                  setInference,
                  "pp"
                )}
              />
              <LabeledInput
                label="DP"
                value={draftInference.dp}
                onChange={handleNumberChange<typeof draftInference>(
                  setInference,
                  "dp"
                )}
              />
            </div>
          </div>
        </CollapsibleGroup>
      </div>
    </aside>
  );
};

/* ─── Collapsible Group ─────────────── */

interface CollapsibleGroupProps {
  label: string;
  color: string;
  tag?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const ChevronIcon: React.FC<{ open: boolean }> = ({ open }) => (
  <svg
    className={`sidebar-group-chevron ${open ? "open" : ""}`}
    viewBox="0 0 20 20"
    fill="currentColor"
  >
    <path
      fillRule="evenodd"
      d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
      clipRule="evenodd"
    />
  </svg>
);

const CollapsibleGroup: React.FC<CollapsibleGroupProps> = ({
  label,
  color,
  tag,
  defaultOpen = true,
  children
}) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="sidebar-group">
      <div
        className="sidebar-group-header"
        onClick={() => setOpen(!open)}
      >
        <span className={`sidebar-group-dot ${color}`} />
        <span className="sidebar-group-label">{label}</span>
        {tag && (
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-borderSoft/40 text-textMuted/70 font-medium tracking-wide">
            {tag}
          </span>
        )}
        <ChevronIcon open={open} />
      </div>
      {open && <div className="sidebar-group-body">{children}</div>}
    </div>
  );
};

/* ─── Primitive inputs ──────────────── */

const Row2: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="grid grid-cols-2 gap-2">{children}</div>
);

const Row3: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="grid grid-cols-3 gap-2">{children}</div>
);

interface LabeledInputProps {
  label: string;
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const LabeledInput: React.FC<LabeledInputProps> = ({
  label,
  value,
  onChange
}) => (
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

interface LabeledSelectOption<T extends string | number> {
  id: T;
  label: string;
}

interface LabeledSelectProps<T extends string | number> {
  label: string;
  value: T;
  options: LabeledSelectOption<T>[];
  onChange: (value: T) => void;
}

const LabeledSelect = <T extends string | number>({
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
      onChange={(e) => {
        const raw = e.target.value;
        onChange((typeof value === "number" ? Number(raw) : raw) as T);
      }}
    >
      {options.map((opt) => (
        <option key={opt.id} value={opt.id}>
          {opt.label}
        </option>
      ))}
    </select>
  </label>
);

interface CheckboxProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

const Checkbox: React.FC<CheckboxProps> = ({ label, checked, onChange }) => (
  <label className="flex items-center gap-2 text-[11px] text-textMuted cursor-pointer select-none py-0.5">
    <input
      type="checkbox"
      className="h-3.5 w-3.5 rounded border-borderSoft/60 bg-background/60 accent-accent"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
    />
    <span>{label}</span>
  </label>
);
