import React, { useMemo } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { useConfigStore, formatBigNumber, computeOverview } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import { computeTrainingMemoryMetrics } from "../lib/metrics";

export const TabTrainingMemory: React.FC = () => {
  const { draftModel, draftMoe, draftTraining } = useConfigStore();
  const draftOverview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );
  const gpu = getGpuSpec(draftTraining.trainingGpuType ?? "H800-80G");

  const mem = computeTrainingMemoryMetrics(
    draftModel,
    draftMoe,
    draftOverview,
    draftTraining
  );

  const segments = [
    { name: "Parameters", bytes: mem.paramsBytes },
    { name: "Optimizer states", bytes: mem.optimizerBytes },
    { name: "Gradients", bytes: mem.gradientBytes },
    { name: "Activations", bytes: mem.activationsBytes }
  ];

  const totalGB = mem.peakBytes / 1e9;
  const perGpuGB = mem.perGpuBytes / 1e9;

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Training Memory
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Parameter, optimizer, gradient and activation memory at training time.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            GPU: <span className="font-semibold">{gpu.name}</span>
            <div>
              VRAM: {gpu.hbmGB} GB · Precision: {draftTraining.precision.toUpperCase()}
            </div>
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Model parameters"
            value={`${(mem.paramsBytes / 1e9).toFixed(2)} GB`}
            sub={`${formatBigNumber(draftOverview.totalParams)} params`}
          />
          <KpiCard
            label="Optimizer state"
            value={`${(mem.optimizerBytes / 1e9).toFixed(2)} GB`}
            sub={draftTraining.optimizer === "adam" ? "Adam (m, v, master)" : "AdaFactor-style"}
          />
          <KpiCard
            label="Gradients"
            value={`${(mem.gradientBytes / 1e9).toFixed(2)} GB`}
            sub="Typically FP32 gradients"
          />
          <KpiCard
            label="Activations"
            value={`${(mem.activationsBytes / 1e9).toFixed(2)} GB`}
            sub="Approx. checkpointed activations"
          />
          <KpiCard
            label="Peak training memory"
            value={`${totalGB.toFixed(2)} GB`}
            sub="Params + optimizer + grads + activations"
          />
          <KpiCard
            label="Memory per GPU"
            value={`${perGpuGB.toFixed(2)} GB`}
            sub="After TP/EP/PP sharding"
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Memory breakdown
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Shows why optimizer state and activations usually dominate training memory.
            </div>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={segments} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
              <XAxis
                type="number"
                dataKey="bytes"
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)} GB`}
                stroke="#9ca3af"
              />
              <YAxis type="category" dataKey="name" stroke="#9ca3af" width={120} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#020617",
                  border: "1px solid #1e293b",
                  borderRadius: "0.5rem",
                  fontSize: 11
                }}
                formatter={(value: number) => [`${(value / 1e9).toFixed(2)} GB`, "Memory"]}
              />
              <Legend />
              <Bar dataKey="bytes" name="Memory" fill="#38bdf8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-3 text-[11px] text-textMuted grid grid-cols-2 gap-3">
          <div>
            <div className="font-semibold mb-1">Key formulas</div>
            <ul className="list-disc list-inside space-y-0.5">
              <li>Params = P_total × bytes_per_param</li>
              <li>Adam (fp32 master) ≈ 12 bytes per param</li>
              <li>Gradients ≈ P_total × 4 bytes</li>
              <li>Activations ≈ O(B × S × d_model) per layer</li>
            </ul>
          </div>
          <div>
            <div className="font-semibold mb-1">Per-GPU view</div>
            <div>
              Parameters, optimizer state and gradients are sharded across TP×EP×PP, while
              activations are only sharded across TP×PP. Data parallelism (DP) replicates
              model state across replicas.
            </div>
          </div>
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

