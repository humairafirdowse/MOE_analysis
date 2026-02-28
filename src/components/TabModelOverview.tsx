import React, { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { formatBigNumber, useConfigStore, computeOverview } from "../state/useConfigStore";

export const TabModelOverview: React.FC = () => {
  const { draftModel, draftMoe } = useConfigStore();
  const draftOverview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );

  const activeRatio =
    draftOverview.totalParams > 0
      ? draftOverview.activeParams / draftOverview.totalParams
      : 0;

  const denseEquivalentParams = draftOverview.activeParams;

  const chartData = [
    {
      component: "Embedding",
      total: draftOverview.embeddingParams,
      active: draftOverview.embeddingParams
    },
    {
      component: "Attention",
      total: draftOverview.attentionParams,
      active: draftOverview.attentionParams
    },
    {
      component: "Routed experts",
      total: draftOverview.expertParams,
      active: draftOverview.expertParams
        ? (draftOverview.expertParams * draftMoe.topK) / Math.max(draftMoe.numExperts, 1)
        : 0
    },
    {
      component: "Shared experts",
      total: draftOverview.sharedExpertParams,
      active: draftOverview.sharedExpertParams
    },
    {
      component: "Output head",
      total: draftOverview.outputHeadParams,
      active: draftOverview.outputHeadParams
    }
  ];

  const totalParamsB = draftOverview.totalParams / 1e9;
  const activeParamsB = draftOverview.activeParams / 1e9;

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Model Overview
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              What this MoE model looks like at a glance.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            d_model={draftModel.dModel} · L={draftModel.layers} · E={draftMoe.numExperts} · K=
            {draftMoe.topK}
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Total parameters"
            value={`${formatBigNumber(draftOverview.totalParams)} params`}
            sub={`${totalParamsB.toFixed(1)}B total`}
          />
          <KpiCard
            label="Active params per token"
            value={`${formatBigNumber(draftOverview.activeParams)} params`}
            sub={`${activeParamsB.toFixed(1)}B active`}
          />
          <KpiCard
            label="Active / Total ratio"
            value={(activeRatio * 100).toFixed(2) + "%"}
            sub={`${activeParamsB.toFixed(1)}B / ${totalParamsB.toFixed(1)}B`}
          />
          <KpiCard
            label="Equivalent dense model"
            value={`${activeParamsB.toFixed(1)}B params`}
            sub="Dense model with same active parameters"
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Architecture Parameter Breakdown
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Horizontal stacked bar showing how parameters are distributed across
              components. Active parameters are solid; inactive expert capacity is ghosted.
            </div>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 8, right: 16, bottom: 8, left: 80 }}
              stackOffset="none"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
              <XAxis
                type="number"
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)}B`}
                stroke="#9ca3af"
              />
              <YAxis
                type="category"
                dataKey="component"
                stroke="#9ca3af"
                width={80}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#020617",
                  border: "1px solid #1e293b",
                  borderRadius: "0.5rem",
                  fontSize: 11
                }}
                formatter={(value: number, key) => {
                  if (key === "active") {
                    return [`${formatBigNumber(value)} active`, "Active params"];
                  }
                  if (key === "inactive") {
                    return [`${formatBigNumber(value)} inactive`, "Inactive params"];
                  }
                  if (key === "total") {
                    return [`${formatBigNumber(value)} total`, "Total params"];
                  }
                  return [value, key];
                }}
              />
              <Legend />
              <Bar
                dataKey="active"
                stackId="params"
                name="Active params"
                fill="#38bdf8"
              />
              <Bar
                dataKey={(d) => Math.max(0, (d as any).total - (d as any).active)}
                stackId="params"
                name="Inactive params"
                fill="#0f172a"
                opacity={0.6}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3 text-[11px] text-textMuted">
          <div>
            <div className="font-semibold mb-1">Key formulas</div>
            <ul className="list-disc list-inside space-y-0.5">
              <li>
                P_total = P_embed + P_attn + P_experts + P_shared + P_output
              </li>
              <li>
                P_active = P_embed + P_attn + K × P_single_expert × L + N_shared ×
                P_shared_expert × L + P_output
              </li>
            </ul>
          </div>
          <div>
            <div className="font-semibold mb-1">Dense baseline (ghost)</div>
            <div>
              We treat a dense model with{" "}
              <span className="text-accent">{denseEquivalentParams / 1e9}</span>B
              parameters as the equivalent dense baseline for all reference
              comparisons.
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

