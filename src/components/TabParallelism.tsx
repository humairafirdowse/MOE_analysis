import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from "recharts";
import { useConfigStore, computeOverview } from "../state/useConfigStore";
import { computeParallelismMetrics } from "../lib/metrics";

export const TabParallelism: React.FC = () => {
  const { draftModel, draftMoe, draftTraining } = useConfigStore();
  const draftOverview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );
  const metrics = computeParallelismMetrics(
    draftModel,
    draftMoe,
    draftOverview,
    draftTraining
  );

  const data = [
    {
      name: "All-to-All (EP)",
      bytes: metrics.allToAllBytesPerStep
    },
    {
      name: "AllReduce (TP)",
      bytes: metrics.allReduceTpBytesPerStep
    },
    {
      name: "P2P (PP)",
      bytes: metrics.p2pBytesPerStep
    },
    {
      name: "AllReduce (DP)",
      bytes: metrics.allReduceDpBytesPerStep
    }
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Parallelism Layout
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              How TP, EP, PP and DP combine into total GPU count and local workloads.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            TP={draftTraining.tp} · EP={draftTraining.ep} · PP={draftTraining.pp} · DP={draftTraining.dp}
            <div>
              Total GPUs:{" "}
              <span className="font-semibold">{metrics.totalGpus}</span>
            </div>
          </div>
        </div>

        <div className="kpi-grid">
          <KpiCard
            label="Total GPUs"
            value={`${metrics.totalGpus}`}
            sub="TP × EP × PP × DP"
          />
          <KpiCard
            label="Experts per GPU"
            value={metrics.expertsPerGpu.toFixed(2)}
            sub={`E = ${draftMoe.numExperts}`}
          />
          <KpiCard
            label="Layers per GPU"
            value={metrics.layersPerGpu.toFixed(2)}
            sub={`L = ${draftModel.layers}`}
          />
          <KpiCard
            label="Total communication / step"
            value={`${(metrics.totalBytesPerStep / 1e9).toFixed(2)} GB`}
            sub="All collective operations combined"
          />
        </div>
      </div>

      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Communication volume by strategy
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Compares All-to-All (expert parallel), AllReduce (tensor/data parallel), and
              P2P traffic for pipeline stages.
            </div>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 160 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
              <XAxis
                type="number"
                dataKey="bytes"
                stroke="#9ca3af"
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)} GB`}
              />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#9ca3af"
                width={160}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#020617",
                  border: "1px solid #1e293b",
                  borderRadius: "0.5rem",
                  fontSize: 11
                }}
                formatter={(value: number) => [`${(value / 1e9).toFixed(2)} GB`, "Bytes / step"]}
              />
              <Legend />
              <Bar dataKey="bytes" name="Bytes / step" fill="#38bdf8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-3 text-[11px] text-textMuted grid grid-cols-2 gap-3">
          <div>
            <div className="font-semibold mb-1">Key formulas</div>
            <ul className="list-disc list-inside space-y-0.5">
              <li>All-to-All (EP): 2·B_micro·S·d·bytes·(EP−1)/EP</li>
              <li>AllReduce (TP): 4·B_micro·S·d·bytes·(TP−1)/TP</li>
              <li>P2P (PP): B_micro·S·d·bytes per stage boundary</li>
              <li>AllReduce (DP): 2·P_total·bytes·(DP−1)/DP</li>
            </ul>
          </div>
          <div>
            <div className="font-semibold mb-1">Interpreting the chart</div>
            <div>
              If All-to-All (EP) dominates, expert parallelism is the main bottleneck;
              if AllReduce (TP/DP) dominates, tensor or data parallelism is the limiting
              factor. This view is useful for deciding which degree of parallelism to
              reduce or increase.
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

