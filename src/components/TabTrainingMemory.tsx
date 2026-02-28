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
    <div className="flex flex-col gap-5">
      <div className="section-card">
        <SectionTitle
          title="Training Memory"
          subtitle="Parameter, optimizer, gradient and activation memory at training time"
          right={
            <div className="text-right text-[11px] text-textMuted/60">
              <span className="font-semibold text-white/60">{gpu.name}</span>
              <div className="mt-0.5">
                VRAM {gpu.hbmGB} GB · {draftTraining.precision.toUpperCase()}
              </div>
            </div>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Model weights"
            value={`${(mem.paramsBytes / 1e9).toFixed(1)} GB`}
            sub={formatBigNumber(draftOverview.totalParams) + " params"}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Optimizer state"
            value={`${(mem.optimizerBytes / 1e9).toFixed(1)} GB`}
            sub={draftTraining.optimizer === "adam" ? "Adam (master + m + v)" : "AdaFactor"}
            accent="border-l-kpiPurple"
          />
          <KpiCard
            label="Gradients"
            value={`${(mem.gradientBytes / 1e9).toFixed(1)} GB`}
            sub={`${draftTraining.gradPrecision.toUpperCase()} precision`}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Activations"
            value={`${(mem.activationsBytes / 1e9).toFixed(1)} GB`}
            sub={draftTraining.activationCheckpointing + " checkpointing"}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Peak total"
            value={totalGB >= 1000 ? `${(totalGB / 1000).toFixed(2)} TB` : `${totalGB.toFixed(1)} GB`}
            sub="Params + optimizer + grads + acts"
            accent="border-l-kpiRose"
          />
          <KpiCard
            label="Per GPU"
            value={`${perGpuGB.toFixed(1)} GB`}
            sub="After TP×EP×PP sharding"
            accent="border-l-accent"
          />
        </div>
      </div>

      <div className="section-card">
        <SectionTitle
          title="Memory Breakdown"
          subtitle="Optimizer state and activations usually dominate training memory"
        />

        <div className="h-64 mt-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={segments} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 130 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" horizontal={false} />
              <XAxis
                type="number"
                dataKey="bytes"
                tickFormatter={(v) => `${(v / 1e9).toFixed(0)} GB`}
                stroke="#8494b0"
                fontSize={11}
              />
              <YAxis type="category" dataKey="name" stroke="#8494b0" width={120} fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0a0f1e",
                  border: "1px solid #1a2540",
                  borderRadius: "0.75rem",
                  fontSize: 11,
                  boxShadow: "0 12px 30px rgba(0,0,0,0.5)"
                }}
                formatter={(value: number) => [`${(value / 1e9).toFixed(1)} GB`, "Memory"]}
              />
              <Legend />
              <Bar dataKey="bytes" name="Memory (bytes)" fill="#a78bfa" radius={[0, 6, 6, 0]} maxBarSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Key formulas</div>
            <ul className="formula-list">
              <li>Weights = P_total x bytes_per_param</li>
              <li>Adam (mixed) = 12 bytes/param (master+m+v)</li>
              <li>Gradients = P_total x 4 bytes (FP32)</li>
              <li>Activations = O(B x S x d) per layer</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Sharding</div>
            <div>
              Weights, optimizer and gradients shard across TP x EP x PP.
              Activations shard across TP x PP only. Data parallelism replicates model state.
            </div>
          </div>
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
