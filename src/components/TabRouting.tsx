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
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Routing & MoE Efficiency
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              How well the MoE design uses parameters compared to dense baselines.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            E={draftMoe.numExperts} · K={draftMoe.topK} · L={draftModel.layers}
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Active/Total parameter ratio"
            value={`${moeActivePct.toFixed(2)}%`}
            sub={`K / E = ${draftMoe.topK} / ${draftMoe.numExperts || 1}`}
          />
          <KpiCard
            label="Gating compute overhead"
            value={`${(metrics.gatingOverheadPct * 100).toFixed(2)}%`}
            sub="Gating FLOPs as % of total forward FLOPs"
          />
          <KpiCard
            label="Expert capacity (tokens/expert)"
            value={metrics.expertCapacityTokens.toFixed(0)}
            sub="Capacity factor ≈ 1.25 · E⁻¹ · B·S·K"
          />
          <KpiCard
            label="Theoretical token drop rate"
            value={`${(metrics.theoreticalDropRate * 100).toFixed(2)}%`}
            sub="Approximate probability of overflow at capacity"
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              MoE vs Dense comparison
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Per-token FLOPs for this MoE model vs dense equivalents.
            </div>
          </div>
        </div>

        <div className="overflow-x-auto text-[12px]">
          <table className="min-w-full">
            <thead>
              <tr className="border-b border-borderSoft/60 text-left">
                <th className="py-1.5 pr-3 font-medium text-textMuted">Model</th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">Total Params</th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">Active Params</th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">FLOPs / token</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-borderSoft/30">
                <td className="py-1.5 pr-3">This MoE model</td>
                <td className="py-1.5 pr-3">
                  {(draftOverview.totalParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-1.5 pr-3">
                  {(draftOverview.activeParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-1.5 pr-3">
                  {(metrics.flopsMoePerToken / 1e9).toFixed(2)} B
                </td>
              </tr>
              <tr className="border-b border-borderSoft/30">
                <td className="py-1.5 pr-3">Dense (same total params)</td>
                <td className="py-1.5 pr-3">
                  {(draftOverview.totalParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-1.5 pr-3">
                  {(draftOverview.totalParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-1.5 pr-3">
                  {(metrics.flopsDenseTotalPerToken / 1e9).toFixed(2)} B
                </td>
              </tr>
              <tr>
                <td className="py-1.5 pr-3">Dense (same active params)</td>
                <td className="py-1.5 pr-3">
                  {(draftOverview.activeParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-1.5 pr-3">
                  {(draftOverview.activeParams / 1e9).toFixed(1)} B
                </td>
                <td className="py-1.5 pr-3">
                  {(metrics.flopsDenseActivePerToken / 1e9).toFixed(2)} B
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section-card">
        <div className="mb-3">
          <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
            Radar: MoE vs dense baselines
          </div>
          <div className="text-[13px] text-textMuted mt-0.5">
            Qualitative comparison across compute efficiency, memory, parameter utilization,
            communication overhead, and inference speed.
          </div>
        </div>
        <div className="h-64">
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
  const utilMoe = metrics.activeParamRatio; // 0..1
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
  const memEffDenseActive = 1.2; // more compact model

  const commOverMoe = 0.6; // more EP → more comm
  const commOverDenseTotal = 0.4;
  const commOverDenseActive = 0.3;

  const infSpeedMoe = computeEffMoe * 1.1;
  const infSpeedDenseTotal = 1;
  const infSpeedDenseActive = computeEffDenseActive * 1.05;

  const maxVal = 2; // for all indicators

  return {
    backgroundColor: "transparent",
    tooltip: {
      confine: true
    },
    legend: {
      data: ["MoE", "Dense (total)", "Dense (active)"],
      textStyle: { color: "#e5e7eb", fontSize: 11 }
    },
    radar: {
      indicator: [
        { name: "Compute efficiency", max: maxVal },
        { name: "Memory efficiency", max: maxVal },
        { name: "Parameter utilization", max: maxVal },
        { name: "Communication overhead (lower better)", max: maxVal },
        { name: "Inference speed", max: maxVal }
      ],
      axisName: { color: "#9ca3af", fontSize: 11 },
      splitLine: { lineStyle: { color: "#1f2937" } },
      splitArea: { areaStyle: { color: ["#020617", "#020617"] } },
      axisLine: { lineStyle: { color: "#1f2937" } }
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
            name: "MoE"
          },
          {
            value: [
              computeEffDenseTotal,
              memEffDenseTotal,
              utilDense * 2,
              commOverDenseTotal,
              infSpeedDenseTotal
            ],
            name: "Dense (total)"
          },
          {
            value: [
              Math.min(computeEffDenseActive, maxVal),
              memEffDenseActive,
              utilDense * 2,
              commOverDenseActive,
              Math.min(infSpeedDenseActive, maxVal)
            ],
            name: "Dense (active)"
          }
        ],
        areaStyle: { opacity: 0.1 }
      }
    ]
  };
}

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

