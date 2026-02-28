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
    ...(draftOverview.denseParams > 0
      ? [
          {
            component: "Dense FFN layers",
            total: draftOverview.denseParams,
            active: draftOverview.denseParams
          }
        ]
      : []),
    ...(draftOverview.residualMLPParams > 0
      ? [
          {
            component: "Residual MLP",
            total: draftOverview.residualMLPParams,
            active: draftOverview.residualMLPParams
          }
        ]
      : []),
    {
      component: "Gating / routing",
      total: draftOverview.gatingParams,
      active: draftOverview.gatingParams
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
    <div className="flex flex-col gap-5">
      {/* KPIs */}
      <div className="section-card">
        <SectionTitle
          title="Model Overview"
          subtitle="Parameter summary and architecture at a glance"
          right={
            <span className="text-[11px] text-textMuted/60 font-mono">
              d={draftModel.dModel} L={draftModel.layers} E={draftMoe.numExperts} K={draftMoe.topK}
            </span>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Total parameters"
            value={`${totalParamsB.toFixed(1)}B`}
            sub={formatBigNumber(draftOverview.totalParams) + " params"}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Active per token"
            value={`${activeParamsB.toFixed(1)}B`}
            sub={formatBigNumber(draftOverview.activeParams) + " params"}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Active / Total"
            value={(activeRatio * 100).toFixed(1) + "%"}
            sub={`${activeParamsB.toFixed(1)}B of ${totalParamsB.toFixed(1)}B`}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Dense equivalent"
            value={`${activeParamsB.toFixed(1)}B`}
            sub="Model with same active params"
            accent="border-l-kpiPurple"
          />
        </div>
      </div>

      {/* Chart */}
      <div className="section-card">
        <SectionTitle
          title="Parameter Breakdown"
          subtitle="Distribution across components — active (solid) vs inactive expert capacity (ghost)"
        />

        <div className="h-72 mt-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 8, right: 24, bottom: 8, left: 100 }}
              stackOffset="none"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" horizontal={false} />
              <XAxis
                type="number"
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)}B`}
                stroke="#8494b0"
                fontSize={11}
              />
              <YAxis
                type="category"
                dataKey="component"
                stroke="#8494b0"
                width={95}
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
                formatter={(value: number, key) => {
                  if (key === "active")
                    return [`${formatBigNumber(value)} active`, "Active"];
                  if (key === "inactive")
                    return [`${formatBigNumber(value)} inactive`, "Inactive"];
                  if (key === "total")
                    return [`${formatBigNumber(value)} total`, "Total"];
                  return [value, key];
                }}
              />
              <Legend />
              <Bar
                dataKey="active"
                stackId="params"
                name="Active params"
                fill="#38bdf8"
                radius={[0, 4, 4, 0]}
              />
              <Bar
                dataKey={(d) => Math.max(0, (d as any).total - (d as any).active)}
                stackId="params"
                name="Inactive params"
                fill="#1a2540"
                opacity={0.7}
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Key formulas</div>
            <ul className="formula-list">
              <li>P_total = P_embed + P_attn + P_experts + P_shared + P_output</li>
              <li>P_active = P_embed + P_attn + K x P_expert x L + N_shared x P_shared x L + P_output</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Dense baseline</div>
            <div>
              A dense model with{" "}
              <span className="text-accent font-semibold">{(denseEquivalentParams / 1e9).toFixed(1)}B</span>{" "}
              parameters is the equivalent dense baseline for compute comparisons.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─── Shared sub-components ─────────── */

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
