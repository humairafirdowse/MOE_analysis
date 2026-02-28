import React, { useMemo } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from "recharts";
import { useConfigStore, formatBigNumber, computeOverview } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import { computeInferenceMetrics } from "../lib/metrics";

export const TabInference: React.FC = () => {
  const { draftModel, draftMoe, draftInference } = useConfigStore();
  const draftOverview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );
  const gpu = getGpuSpec(draftInference.gpuType);

  const m = computeInferenceMetrics(
    draftModel,
    draftMoe,
    draftOverview,
    draftInference,
    gpu
  );

  const fmtMs = (ms: number) => {
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (ms >= 1) return `${ms.toFixed(1)} ms`;
    return `${(ms * 1000).toFixed(0)} us`;
  };

  return (
    <div className="flex flex-col gap-5">
      {/* GPU & parallelism summary */}
      <div className="section-card">
        <SectionTitle
          title="Inference Configuration"
          subtitle="GPU cluster and parallelism for inference serving"
          right={
            <div className="text-right text-[11px] text-textMuted/60">
              <span className="font-semibold text-white/60">{gpu.name}</span>
              <div className="mt-0.5">
                {draftInference.precision.toUpperCase()} · Batch {draftInference.batchSize}
              </div>
            </div>
          }
        />
        <div className="kpi-grid mt-4">
          <KpiCard
            label="Total GPUs"
            value={`${m.totalGpus}`}
            sub={`TP=${m.tp} EP=${m.ep} PP=${m.pp} DP=${m.dp}`}
            accent="border-l-accent"
          />
          <KpiCard
            label="Weights / GPU"
            value={`${(m.weightsPerGpuBytes / 1e9).toFixed(1)} GB`}
            sub={`${(m.weightsBytes / 1e9).toFixed(0)} GB total / ${m.tp * m.ep * m.pp} GPUs`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="KV cache / GPU"
            value={`${(m.kvPerGpuBytes / 1e9).toFixed(2)} GB`}
            sub={`Sharded across TP=${m.tp}`}
            accent="border-l-kpiPurple"
          />
          <KpiCard
            label="Total / GPU"
            value={`${(m.totalPerGpuBytes / 1e9).toFixed(1)} GB`}
            sub={`GPU VRAM: ${gpu.hbmGB} GB`}
            accent="border-l-kpiRose"
          />
        </div>
      </div>

      {/* Latency & throughput */}
      <div className="section-card">
        <SectionTitle
          title="Latency & Throughput"
          subtitle="Prefill (compute-bound) and decode (memory-bound) performance"
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Prefill latency"
            value={fmtMs(m.prefillLatencyMs)}
            sub={`${draftInference.batchSize} x ${draftInference.inputSeqLen} tokens`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Prefill throughput"
            value={`${(m.prefillTokensPerSec / 1000).toFixed(1)}K tok/s`}
            sub="Compute-bound"
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Decode per token"
            value={fmtMs(m.decodeLatencyMsPerToken)}
            sub="Memory-bandwidth bound"
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Decode throughput"
            value={`${m.decodeTokensPerSec.toFixed(1)} tok/s`}
            sub={`Batch=${draftInference.batchSize} (per replica)`}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Max batch (by KV)"
            value={
              m.maxBatchSizeByMemory > 0
                ? `${m.maxBatchSizeByMemory}`
                : "N/A"
            }
            sub={`Seq len = ${draftInference.inputSeqLen + draftInference.outputSeqLen}`}
            accent="border-l-kpiAmber"
          />
        </div>
      </div>

      {/* End-to-end timing */}
      <div className="section-card">
        <SectionTitle
          title="End-to-End Generation Timing"
          subtitle="TTFT, inter-token latency, and total generation time"
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="TTFT"
            value={fmtMs(m.ttftMs)}
            sub="Time to first token (= prefill)"
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Inter-token latency"
            value={fmtMs(m.interTokenLatencyMs)}
            sub="Time between consecutive decode tokens"
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Total decode time"
            value={fmtMs(m.totalDecodeTimeMs)}
            sub={`${draftInference.outputSeqLen} output tokens`}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Total generation time"
            value={fmtMs(m.totalGenerationTimeMs)}
            sub="TTFT + all decode steps"
            accent="border-l-kpiRose"
          />
        </div>
      </div>

      {/* Memory */}
      <div className="section-card">
        <SectionTitle
          title="Inference Memory"
          subtitle="Model weights vs KV cache for chosen batch and sequence lengths"
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Model weights"
            value={`${(m.weightsBytes / 1e9).toFixed(1)} GB`}
            sub={formatBigNumber(draftOverview.totalParams) + " params"}
            accent="border-l-kpiPurple"
          />
          <KpiCard
            label="KV cache"
            value={`${(m.kvTotalBytes / 1e9).toFixed(2)} GB`}
            sub={`B=${draftInference.batchSize} L=${draftModel.layers}`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Total memory"
            value={`${(m.totalBytes / 1e9).toFixed(1)} GB`}
            sub="Weights + KV cache (cluster)"
            accent="border-l-kpiRose"
          />
        </div>
      </div>

      {/* KV cache + timing chart vs sequence length */}
      <div className="section-card">
        <SectionTitle
          title="Latency vs Sequence Length"
          subtitle="KV cache, decode latency, TTFT, and total time as functions of context length"
        />

        <div className="h-72 mt-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={m.seqSamples} margin={{ top: 8, right: 48, bottom: 8, left: 48 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
              <XAxis
                dataKey="seqLen"
                stroke="#8494b0"
                fontSize={11}
                tickFormatter={(v) => `${v}`}
                label={{ value: "Sequence length", position: "insideBottomRight", offset: -6, fill: "#8494b0", fontSize: 10 }}
              />
              <YAxis
                yAxisId="left"
                stroke="#34d399"
                fontSize={11}
                tickFormatter={(v) => `${v.toFixed(1)} GB`}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#38bdf8"
                fontSize={11}
                tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${v.toFixed(0)}ms`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0a0f1e",
                  border: "1px solid #1a2540",
                  borderRadius: "0.75rem",
                  fontSize: 11,
                  boxShadow: "0 12px 30px rgba(0,0,0,0.5)"
                }}
                formatter={(value: number, name: string) =>
                  name === "KV cache (GB)"
                    ? [`${value.toFixed(2)} GB`, name]
                    : [`${value >= 1000 ? (value / 1000).toFixed(2) + " s" : value.toFixed(1) + " ms"}`, name]
                }
              />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="kvGB"
                name="KV cache (GB)"
                stroke="#34d399"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="decodeMs"
                name="Decode / token (ms)"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="ttftMs"
                name="TTFT (ms)"
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="totalTimeMs"
                name="Total gen time (ms)"
                stroke="#f472b6"
                strokeWidth={2}
                strokeDasharray="4 2"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const SectionTitle: React.FC<{
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}> = ({ title, subtitle, right }) => (
  <div className="flex items-start justify-between">
    <div>
      <div className="section-header text-accent/90">{title}</div>
      {subtitle && <div className="section-subtitle">{subtitle}</div>}
    </div>
    {right && <div className="mt-0.5">{right}</div>}
  </div>
);

interface KpiCardProps {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}

const KpiCard: React.FC<KpiCardProps> = ({ label, value, sub, accent = "border-l-accent" }) => (
  <div className={`kpi-card border-l-2 ${accent}`}>
    <div className="kpi-label">{label}</div>
    <div className="kpi-value">{value}</div>
    {sub && <div className="kpi-subvalue">{sub}</div>}
  </div>
);
