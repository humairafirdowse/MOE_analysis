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
    { name: "All-to-All (EP)", bytes: metrics.allToAllBytesPerStep },
    { name: "AllReduce (TP)", bytes: metrics.allReduceTpBytesPerStep },
    { name: "P2P (PP)", bytes: metrics.p2pBytesPerStep },
    { name: "AllReduce (DP)", bytes: metrics.allReduceDpBytesPerStep }
  ];

  return (
    <div className="flex flex-col gap-5">
      <div className="section-card">
        <SectionTitle
          title="Parallelism Layout"
          subtitle="How TP, EP, PP and DP combine into total GPU count and local workloads"
          right={
            <div className="text-right text-[11px] text-textMuted/60 font-mono">
              TP={draftTraining.tp} EP={draftTraining.ep} PP={draftTraining.pp} DP={draftTraining.dp}
            </div>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Total GPUs"
            value={`${metrics.totalGpus}`}
            sub="TP x EP x PP x DP"
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Experts / GPU"
            value={metrics.expertsPerGpu.toFixed(1)}
            sub={`E = ${draftMoe.numExperts}`}
            accent="border-l-kpiPurple"
          />
          <KpiCard
            label="Layers / GPU"
            value={metrics.layersPerGpu.toFixed(1)}
            sub={`L = ${draftModel.layers}`}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Comm / step"
            value={`${(metrics.totalBytesPerStep / 1e9).toFixed(1)} GB`}
            sub="All collectives combined"
            accent="border-l-kpiAmber"
          />
        </div>
      </div>

      <div className="section-card">
        <SectionTitle
          title="Communication Volume"
          subtitle="All-to-All (EP), AllReduce (TP/DP), and P2P (PP) traffic per step"
        />

        <div className="h-64 mt-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 150 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" horizontal={false} />
              <XAxis
                type="number"
                dataKey="bytes"
                stroke="#8494b0"
                fontSize={11}
                tickFormatter={(v) => `${(v / 1e9).toFixed(1)} GB`}
              />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#8494b0"
                width={140}
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
                formatter={(value: number) => [`${(value / 1e9).toFixed(2)} GB`, "Bytes / step"]}
              />
              <Legend />
              <Bar dataKey="bytes" name="Bytes / step" fill="#fbbf24" maxBarSize={40} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Key formulas</div>
            <ul className="formula-list">
              <li>All-to-All (EP): 2·B·S·d·bytes·(EP-1)/EP</li>
              <li>AllReduce (TP): 4·B·S·d·bytes·(TP-1)/TP</li>
              <li>P2P (PP): B·S·d·bytes per boundary</li>
              <li>AllReduce (DP): 2·P·bytes·(DP-1)/DP</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Interpretation</div>
            <div>
              If All-to-All (EP) dominates, expert parallelism is the main bottleneck.
              If AllReduce (TP/DP) dominates, tensor or data parallelism is the limiting factor.
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
