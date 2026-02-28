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

  const metrics = computeInferenceMetrics(
    draftModel,
    draftMoe,
    draftOverview,
    draftInference,
    gpu
  );

  const kvGB = metrics.kvTotalBytes / 1e9;
  const weightsGB = metrics.weightsBytes / 1e9;

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Inference Compute & Latency
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Prefill and decode FLOPs, latency, and throughput for the selected GPU.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            GPU: <span className="font-semibold">{gpu.name}</span>
            <div>
              Precision: {draftInference.precision.toUpperCase()} · Batch size:{" "}
              {draftInference.batchSize}
            </div>
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Prefill FLOPs per token"
            value={`${formatBigNumber(metrics.prefillFlopsPerToken)} FLOPs`}
            sub={`Seq len = ${draftInference.inputSeqLen}`}
          />
          <KpiCard
            label="Decode FLOPs per token"
            value={`${formatBigNumber(metrics.decodeFlopsPerToken)} FLOPs`}
            sub="Similar structure, different KV access pattern"
          />
          <KpiCard
            label="Prefill latency (batch)"
            value={`${metrics.prefillLatencyMs.toFixed(1)} ms`}
            sub={`${draftInference.batchSize} × ${draftInference.inputSeqLen} tokens`}
          />
          <KpiCard
            label="Decode latency per token"
            value={`${metrics.decodeLatencyMsPerToken.toFixed(2)} ms`}
            sub="Memory-bandwidth-bound estimate"
          />
          <KpiCard
            label="Estimated tokens / second"
            value={`${metrics.tokensPerSecond.toFixed(1)} tok/s`}
            sub="Single stream, memory-bound"
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Inference Memory
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Model weights vs KV cache footprint for the chosen batch size and sequence
              lengths.
            </div>
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Model weights"
            value={`${weightsGB.toFixed(2)} GB`}
            sub={`${formatBigNumber(draftOverview.totalParams)} params`}
          />
          <KpiCard
            label="KV cache"
            value={`${kvGB.toFixed(2)} GB`}
            sub={`B = ${draftInference.batchSize}, L = ${draftModel.layers}, n_kv = ${draftModel.nKvHeads}`}
          />
          <KpiCard
            label="Total inference memory"
            value={`${(metrics.totalBytes / 1e9).toFixed(2)} GB`}
            sub={`GPU VRAM: ${gpu.hbmGB} GB`}
          />
          <KpiCard
            label="Max batch size (by KV)"
            value={
              metrics.maxBatchSizeByMemory > 0
                ? `${metrics.maxBatchSizeByMemory}`
                : "N/A"
            }
            sub={`For seq len = ${
              draftInference.inputSeqLen + draftInference.outputSeqLen
            }`}
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              KV cache vs sequence length
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              KV cache grows linearly with sequence length; decode latency also grows as
              more KV is touched.
            </div>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics.seqSamples} margin={{ top: 8, right: 40, bottom: 8, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="seqLen"
                stroke="#9ca3af"
                tickFormatter={(v) => `${v}`}
                label={{ value: "Sequence length", position: "insideBottomRight", offset: -6, fill: "#9ca3af", fontSize: 11 }}
              />
              <YAxis
                yAxisId="left"
                stroke="#34d399"
                tickFormatter={(v) => `${v.toFixed(1)} GB`}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#38bdf8"
                tickFormatter={(v) => `${v.toFixed(0)} ms`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#020617",
                  border: "1px solid #1e293b",
                  borderRadius: "0.5rem",
                  fontSize: 11
                }}
                formatter={(value: number, name) =>
                  name === "KV cache (GB)"
                    ? [`${value.toFixed(2)} GB`, name]
                    : [`${value.toFixed(1)} ms`, name]
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
                name="Decode latency (ms/token)"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

interface KpiCardProps {
  label: string;
  value: string;
  sub?: string;
}

const KpiCard: React.FC<KpiCardProps> = ({ label, value, sub }) => (
  <div className="kpi-card">
    <div className="kpi-label">{label}</div>
    <div className="kpi-value">{value}</div>
    {sub && <div className="kpi-subvalue">{sub}</div>}
  </div>
);

