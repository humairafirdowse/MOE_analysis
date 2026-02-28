import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { useConfigStore, computeOverview } from "../state/useConfigStore";
import { computeRoutingEfficiencyMetrics } from "../lib/metrics";

export const TabRouting: React.FC = () => {
  const { draftModel, draftMoe, draftTraining } = useConfigStore();
  const draftOverview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );
  const metrics = computeRoutingEfficiencyMetrics(
    draftModel,
    draftMoe,
    draftOverview,
    draftTraining
  );

  const moeActivePct = metrics.activeParamRatio * 100;
  const radarOption = buildRadarOption(metrics);

  return (
    <div className="flex flex-col gap-5">
      {/* KPIs */}
      <div className="section-card">
        <SectionTitle
          title="Routing & MoE Efficiency"
          subtitle="How well the MoE design uses parameters compared to dense baselines"
          right={
            <span className="text-[11px] text-textMuted/60 font-mono">
              E={draftMoe.numExperts} K={draftMoe.topK} L={draftModel.layers}
            </span>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Expert activation ratio"
            value={`${moeActivePct.toFixed(1)}%`}
            sub={`K/E = ${draftMoe.topK}/${draftMoe.numExperts || 1}`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Gating overhead"
            value={`${(metrics.gatingOverheadPct * 100).toFixed(2)}%`}
            sub="% of total forward FLOPs"
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Expert capacity"
            value={metrics.expertCapacityTokens.toFixed(0)}
            sub="Tokens/expert (CF=1.25)"
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Token drop rate"
            value={`${(metrics.theoreticalDropRate * 100).toFixed(2)}%`}
            sub="Theoretical overflow probability"
            accent="border-l-kpiRose"
          />
        </div>
      </div>

      {/* Comparison table */}
      <div className="section-card">
        <SectionTitle
          title="MoE vs Dense Comparison"
          subtitle="Per-token FLOPs across model configurations"
        />

        <div className="overflow-x-auto mt-3">
          <table className="min-w-full text-[12px]">
            <thead>
              <tr className="border-b border-borderSoft/40 text-left">
                <th className="py-2 pr-4 font-bold text-[10px] uppercase tracking-wider text-textMuted">Model</th>
                <th className="py-2 pr-4 font-bold text-[10px] uppercase tracking-wider text-textMuted">Total Params</th>
                <th className="py-2 pr-4 font-bold text-[10px] uppercase tracking-wider text-textMuted">Active Params</th>
                <th className="py-2 pr-4 font-bold text-[10px] uppercase tracking-wider text-textMuted">FLOPs / token</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-borderSoft/20">
                <td className="py-2.5 pr-4 font-medium text-accent/80">This MoE model</td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(draftOverview.totalParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(draftOverview.activeParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-2.5 pr-4 font-semibold">
                  {(metrics.flopsMoePerToken / 1e9).toFixed(2)} B
                </td>
              </tr>
              <tr className="border-b border-borderSoft/20">
                <td className="py-2.5 pr-4 text-textMuted/80">Dense (same total)</td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(draftOverview.totalParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(draftOverview.totalParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(metrics.flopsDenseTotalPerToken / 1e9).toFixed(2)} B
                </td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 text-textMuted/80">Dense (same active)</td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(draftOverview.activeParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(draftOverview.activeParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-2.5 pr-4 text-textMuted">
                  {(metrics.flopsDenseActivePerToken / 1e9).toFixed(2)} B
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Radar */}
      <div className="section-card">
        <SectionTitle
          title="Radar: MoE vs Dense"
          subtitle="Qualitative comparison across compute, memory, utilization, communication and inference speed"
        />
        <div className="h-72 mt-2">
          <ReactECharts
            style={{ width: "100%", height: "100%" }}
            option={radarOption}
            notMerge
          />
        </div>
      </div>
    </div>
  );
};

function buildRadarOption(metrics: ReturnType<typeof computeRoutingEfficiencyMetrics>) {
  const utilMoe = metrics.activeParamRatio;
  const utilDense = 1;

  const computeEffMoe =
    metrics.flopsDenseTotalPerToken > 0
      ? metrics.flopsDenseTotalPerToken / metrics.flopsMoePerToken
      : 1;
  const computeEffDenseTotal = 1;
  const computeEffDenseActive =
    metrics.flopsDenseTotalPerToken > 0
      ? metrics.flopsDenseTotalPerToken / metrics.flopsDenseActivePerToken
      : 1;

  const memEffMoe =
    metrics.flopsDenseTotalPerToken > 0
      ? metrics.flopsDenseTotalPerToken / metrics.flopsMoePerToken
      : 1;
  const memEffDenseTotal = 1;
  const memEffDenseActive = 1.2;

  const commOverMoe = 0.6;
  const commOverDenseTotal = 0.4;
  const commOverDenseActive = 0.3;

  const infSpeedMoe = computeEffMoe * 1.1;
  const infSpeedDenseTotal = 1;
  const infSpeedDenseActive = computeEffDenseActive * 1.05;

  const maxVal = 2;

  return {
    backgroundColor: "transparent",
    tooltip: { confine: true },
    legend: {
      data: ["MoE", "Dense (total)", "Dense (active)"],
      textStyle: { color: "#8494b0", fontSize: 11 }
    },
    radar: {
      indicator: [
        { name: "Compute eff.", max: maxVal },
        { name: "Memory eff.", max: maxVal },
        { name: "Param util.", max: maxVal },
        { name: "Comm. (lower=better)", max: maxVal },
        { name: "Inference speed", max: maxVal }
      ],
      axisName: { color: "#8494b0", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1a2540" } },
      splitArea: { areaStyle: { color: ["#06080f", "#0a0f1e"] } },
      axisLine: { lineStyle: { color: "#1a2540" } }
    },
    series: [
      {
        type: "radar",
        data: [
          {
            value: [
              Math.min(computeEffMoe, maxVal),
              Math.min(memEffMoe, maxVal),
              Math.min(utilMoe * 2, maxVal),
              Math.min(commOverMoe, maxVal),
              Math.min(infSpeedMoe, maxVal)
            ],
            name: "MoE",
            lineStyle: { color: "#38bdf8" },
            itemStyle: { color: "#38bdf8" },
            areaStyle: { color: "rgba(56,189,248,0.08)" }
          },
          {
            value: [
              computeEffDenseTotal,
              memEffDenseTotal,
              utilDense * 2,
              commOverDenseTotal,
              infSpeedDenseTotal
            ],
            name: "Dense (total)",
            lineStyle: { color: "#a78bfa" },
            itemStyle: { color: "#a78bfa" },
            areaStyle: { color: "rgba(167,139,250,0.06)" }
          },
          {
            value: [
              Math.min(computeEffDenseActive, maxVal),
              memEffDenseActive,
              utilDense * 2,
              commOverDenseActive,
              Math.min(infSpeedDenseActive, maxVal)
            ],
            name: "Dense (active)",
            lineStyle: { color: "#34d399" },
            itemStyle: { color: "#34d399" },
            areaStyle: { color: "rgba(52,211,153,0.06)" }
          }
        ]
      }
    ]
  };
}

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
