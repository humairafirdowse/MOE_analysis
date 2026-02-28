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
    {
      component: "Attention",
      flops: metrics.attentionFlopsPerToken
    },
    {
      component: "MoE FFN (active experts)",
      flops: metrics.moeFfnFlopsPerToken
    },
    {
      component: "Shared FFN",
      flops: metrics.sharedFfnFlopsPerToken
    },
    {
      component: "Gating / routing",
      flops: metrics.gatingFlopsPerToken
    }
  ];

  const denseRatio =
    metrics.denseEquivalentFlopsPerToken > 0
      ? metrics.totalFlopsPerToken / metrics.denseEquivalentFlopsPerToken
      : 1;

  const formatGpuHours = (h: number) =>
    h >= 1e6 ? (h / 1e6).toFixed(2) + "M" : h >= 1e3 ? (h / 1e3).toFixed(1) + "K" : h.toFixed(0);

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Training Compute
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Per-token FLOPs breakdown and end-to-end training cost.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            Training GPU: <span className="font-semibold">{trainingGpu.name}</span>
            <div>
              Precision: {draftTraining.precision.toUpperCase()} · MFU: {(mfu * 100).toFixed(0)}% ·{" "}
              {metrics.totalGpus} GPUs (TP×EP×PP×DP)
            </div>
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Per-token forward FLOPs"
            value={`${formatBigNumber(metrics.forwardFlopsPerToken)} FLOPs`}
            sub="Single token, forward only"
          />
          <KpiCard
            label="Per-token total FLOPs"
            value={`${formatBigNumber(metrics.totalFlopsPerToken)} FLOPs`}
            sub="Forward + backward (~3× forward)"
          />
          <KpiCard
            label="Total training FLOPs"
            value={`${formatBigNumber(metrics.totalTrainingFlops)} FLOPs`}
            sub={`T = ${formatBigNumber(draftTraining.totalTrainingTokens)} tokens`}
          />
          <KpiCard
            label="Approx. GPU-hours"
            value={formatGpuHours(metrics.gpuHoursApprox) + " GPU·h"}
            sub={`${metrics.totalGpus} GPUs · ${(mfu * 100).toFixed(0)}% MFU`}
          />
          <KpiCard
            label="Dense equivalent FLOPs/token"
            value={`${formatBigNumber(metrics.denseEquivalentFlopsPerToken)} FLOPs`}
            sub={`${(denseRatio * 100).toFixed(1)}% of dense`}
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              FLOPs breakdown per token
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Relative compute contribution of attention, MoE FFN, shared FFN and gating.
            </div>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 8, right: 16, bottom: 8, left: 180 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
              <XAxis
                type="number"
                dataKey="flops"
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)}B`}
                stroke="#9ca3af"
                domain={[0, "auto"]}
              />
              <YAxis
                type="category"
                dataKey="component"
                stroke="#9ca3af"
                width={170}
                tick={{ fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#020617",
                  border: "1px solid #1e293b",
                  borderRadius: "0.5rem",
                  fontSize: 11
                }}
                formatter={(value: number) => [`${formatBigNumber(value)} FLOPs`, "FLOPs"]}
              />
              <Legend />
              <Bar dataKey="flops" name="FLOPs per token" fill="#38bdf8" maxBarSize={48} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-3 text-[11px] text-textMuted grid grid-cols-2 gap-3">
          <div>
            <div className="font-semibold mb-1">Key formulas</div>
            <ul className="list-disc list-inside space-y-0.5">
              <li>Attention: ~8·d² + 4·S·d per layer per token</li>
              <li>MoE FFN: K · 6·d·d_ff per MoE layer per token</li>
              <li>Gating: 2·d·E per MoE layer per token</li>
              <li>Total train FLOPs ≈ 3·F_fwd/token · T</li>
            </ul>
          </div>
          <div>
            <div className="font-semibold mb-1">Dense equivalent</div>
            <div>
              Dense baseline fires <span className="font-semibold">all experts</span>{" "}
              per token instead of top-K. The MoE model reaches similar quality at{" "}
              <span className="text-accent">
                {(denseRatio * 100).toFixed(1)}%
              </span>{" "}
              of dense compute.
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

