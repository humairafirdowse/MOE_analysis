import React from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useConfigStore, formatBigNumber } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import { computeTrainingComputeMetrics } from "../lib/metrics";

export const TabTrainingCompute: React.FC = () => {
  const { draftModel, draftMoe, draftTraining } = useConfigStore();
  const trainingGpuType = draftTraining.trainingGpuType ?? "H800-80G";
  const mfu = draftTraining.mfu ?? 0.55;
  const trainingGpu = getGpuSpec(trainingGpuType);

  const metrics = computeTrainingComputeMetrics(
    draftModel,
    draftMoe,
    draftTraining,
    trainingGpu,
    mfu
  );

  const chartData = [
    { component: "Attention", flops: metrics.attentionFlopsPerToken },
    { component: "MoE FFN (active)", flops: metrics.moeFfnFlopsPerToken },
    ...(metrics.sharedFfnFlopsPerToken > 0
      ? [{ component: "Shared FFN", flops: metrics.sharedFfnFlopsPerToken }]
      : []),
    ...(metrics.denseFfnFlopsPerToken > 0
      ? [{ component: "Dense FFN", flops: metrics.denseFfnFlopsPerToken }]
      : []),
    ...(metrics.residualMlpFfnFlopsPerToken > 0
      ? [{ component: "Residual MLP", flops: metrics.residualMlpFfnFlopsPerToken }]
      : []),
    { component: "Gating / routing", flops: metrics.gatingFlopsPerToken }
  ];

  const denseRatio =
    metrics.denseEquivalentFlopsPerToken > 0
      ? metrics.totalFlopsPerToken / metrics.denseEquivalentFlopsPerToken
      : 1;

  const formatGpuHours = (h: number) =>
    h >= 1e6 ? (h / 1e6).toFixed(2) + "M" : h >= 1e3 ? (h / 1e3).toFixed(1) + "K" : h.toFixed(0);

  const fmtUSD = (v: number) =>
    v >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M" : v >= 1e3 ? "$" + (v / 1e3).toFixed(0) + "K" : "$" + v.toFixed(0);

  return (
    <div className="flex flex-col gap-5">
      <div className="section-card">
        <SectionTitle
          title="Training Compute"
          subtitle="Per-token FLOPs breakdown and end-to-end training cost"
          right={
            <div className="text-right text-[11px] text-textMuted/60">
              <span className="font-semibold text-white/60">{trainingGpu.name}</span>
              <div className="mt-0.5">
                {draftTraining.precision.toUpperCase()} · MFU {(mfu * 100).toFixed(0)}% · {metrics.totalGpus} GPUs
              </div>
            </div>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Forward FLOPs / token"
            value={formatBigNumber(metrics.forwardFlopsPerToken)}
            sub="Single token, forward only"
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Total FLOPs / token"
            value={formatBigNumber(metrics.totalFlopsPerToken)}
            sub="Fwd + bwd (3x forward)"
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Total training FLOPs"
            value={formatBigNumber(metrics.totalTrainingFlops)}
            sub={`T = ${formatBigNumber(draftTraining.totalTrainingTokens)} tokens`}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="GPU-hours"
            value={formatGpuHours(metrics.gpuHoursApprox) + " GPU·h"}
            sub={`${metrics.totalGpus} GPUs · ${(mfu * 100).toFixed(0)}% MFU`}
            accent="border-l-kpiRose"
          />
          <KpiCard
            label="Training cost"
            value={fmtUSD(metrics.trainingCostUSD)}
            sub={`${formatGpuHours(metrics.gpuHoursApprox)} GPU·h × $${trainingGpu.costPerHourUSD}/h`}
            accent="border-l-kpiPurple"
          />
        </div>
      </div>

      <div className="section-card">
        <SectionTitle
          title="FLOPs Breakdown"
          subtitle="Relative compute contribution per component per token"
        />

        <div className="h-72 mt-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 8, right: 24, bottom: 8, left: 140 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" horizontal={false} />
              <XAxis
                type="number"
                dataKey="flops"
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)}B`}
                stroke="#8494b0"
                fontSize={11}
                domain={[0, "auto"]}
              />
              <YAxis
                type="category"
                dataKey="component"
                stroke="#8494b0"
                width={130}
                fontSize={11}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0a0f1e",
                  border: "1px solid #1a2540",
                  borderRadius: "0.75rem",
                  fontSize: 11,
                  boxShadow: "0 12px 30px rgba(0,0,0,0.5)"
                }}
                formatter={(value: number) => [`${formatBigNumber(value)} FLOPs`, "FLOPs"]}
              />
              <Legend />
              <Bar dataKey="flops" name="FLOPs / token" fill="#38bdf8" maxBarSize={40} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Key formulas</div>
            <ul className="formula-list">
              <li>Attention: ~8·d² + 4·S·d per layer</li>
              <li>MoE FFN: K · 6·d·d_ff per MoE layer</li>
              <li>Gating: 2·d·E per MoE layer</li>
              <li>Total train ≈ 3 · F_fwd/token · T</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Dense equivalent</div>
            <div>
              Dense fires <span className="font-semibold text-white/70">all experts</span> per token.
              MoE reaches similar quality at{" "}
              <span className="text-accent font-semibold">{(denseRatio * 100).toFixed(1)}%</span> of dense compute.
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
