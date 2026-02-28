import React, { useMemo } from "react";
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from "recharts";
import { useConfigStore, computeOverview } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import {
  computeHardwareEfficiencyMetrics,
  computeParallelismMetrics,
  type HardwareEfficiencyMetrics
} from "../lib/metrics";

export const TabHardwareEfficiency: React.FC = () => {
  const { draftModel, draftMoe, draftTraining, draftInference } =
    useConfigStore();

  const overview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );

  const trainingGpu = getGpuSpec(draftTraining.trainingGpuType ?? "H800-80G");
  const inferenceGpu = getGpuSpec(draftInference.gpuType);
  const mfu = draftTraining.mfu ?? 0.55;

  const parallelism = useMemo(
    () => computeParallelismMetrics(draftModel, draftMoe, overview, draftTraining),
    [draftModel, draftMoe, overview, draftTraining]
  );

  const hw = useMemo(
    () =>
      computeHardwareEfficiencyMetrics(
        draftModel,
        draftMoe,
        overview,
        draftTraining,
        trainingGpu,
        mfu,
        draftInference,
        inferenceGpu,
        parallelism
      ),
    [draftModel, draftMoe, overview, draftTraining, trainingGpu, mfu, draftInference, inferenceGpu, parallelism]
  );

  return (
    <div className="flex flex-col gap-5">
      {/* ── Section A: Training Efficiency ── */}
      <div className="section-card">
        <SectionTitle
          title="Training Efficiency"
          subtitle="MFU, HFU, pipeline overhead, compute-comm overlap, and effective throughput"
          right={
            <span className="text-[11px] text-textMuted/60 font-mono">
              {trainingGpu.name.replace("NVIDIA ", "")} · {draftTraining.precision.toUpperCase()}
            </span>
          }
        />

        {/* Big MFU + HFU gauges */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          <MfuGauge label="MFU" sublabel="Model FLOPs Utilization" value={hw.mfu} />
          <MfuGauge label="HFU" sublabel="Hardware FLOPs Utilization" value={hw.hfu} />
        </div>

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Recompute multiplier"
            value={`${hw.recomputeMultiplier.toFixed(2)}x`}
            sub={`HFU/MFU — ${draftTraining.activationCheckpointing} checkpointing`}
            accent="border-l-kpiPurple"
          />
          <KpiCard
            label="Pipeline bubble"
            value={`${hw.pipelineBubblePct.toFixed(2)}%`}
            sub={`PP=${draftTraining.pp}, ${hw.numMicrobatches} microbatches`}
            accent={hw.pipelineBubblePct > 5 ? "border-l-kpiRose" : "border-l-kpiGreen"}
          />
          <KpiCard
            label="Effective throughput"
            value={`${hw.effectiveThroughputTokensPerSecPerGpu.toFixed(0)} tok/s/GPU`}
            sub="After all parallelism overheads"
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Compute-comm overlap"
            value={`${hw.computeCommOverlapPct.toFixed(1)}%`}
            sub={hw.computeCommOverlapPct >= 80
              ? "Communication well hidden"
              : hw.computeCommOverlapPct >= 50
              ? "Partial overlap — some comm exposed"
              : "Communication bottleneck"}
            accent={hw.computeCommOverlapPct >= 80
              ? "border-l-kpiGreen"
              : hw.computeCommOverlapPct >= 50
              ? "border-l-kpiAmber"
              : "border-l-kpiRose"}
          />
        </div>

        <div className="grid grid-cols-2 gap-4 mt-3 text-[11px] text-textMuted">
          <div className="bg-card/60 rounded-lg px-3 py-2 border border-borderSoft/20">
            <span className="font-bold text-white/60 text-[10px] uppercase tracking-wider">Compute time / step</span>
            <div className="mt-0.5 font-mono">{hw.computeTimePerStepMs.toFixed(1)} ms</div>
          </div>
          <div className="bg-card/60 rounded-lg px-3 py-2 border border-borderSoft/20">
            <span className="font-bold text-white/60 text-[10px] uppercase tracking-wider">Comm time / step</span>
            <div className="mt-0.5 font-mono">{hw.commTimePerStepMs.toFixed(1)} ms</div>
          </div>
        </div>

        <div className="mt-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4 grid grid-cols-2 gap-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Key formulas</div>
            <ul className="formula-list">
              <li>MFU = model_FLOPs / (peak_FLOPs x time x GPUs)</li>
              <li>HFU = (model_FLOPs + recompute) / (peak x time x GPUs)</li>
              <li>Bubble = (PP - 1) / num_microbatches (1F1B)</li>
              <li>Throughput = effective_TFLOPS / FLOPs_per_token</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Interpretation</div>
            <ul className="formula-list">
              <li>HFU &gt; MFU shows checkpointing recomputation overhead</li>
              <li>Low overlap % = communication is not hidden behind compute</li>
              <li>High bubble % = too few microbatches for PP depth</li>
            </ul>
          </div>
        </div>
      </div>

      {/* ── Section B: Inference Roofline ── */}
      <div className="section-card">
        <SectionTitle
          title="Inference Roofline Analysis"
          subtitle="Log-log plot of achievable TFLOPS vs arithmetic intensity — see where prefill and decode operate"
          right={
            <span className="text-[11px] text-textMuted/60 font-mono">
              {inferenceGpu.name.replace("NVIDIA ", "")} · {draftInference.precision.toUpperCase()}
            </span>
          }
        />

        <div className="h-80 mt-3">
          <RooflinePlot hw={hw} gpuName={inferenceGpu.name} />
        </div>

        {/* Verdict badges */}
        <div className="grid grid-cols-2 gap-3 mt-4">
          <BoundVerdict
            phase="Prefill"
            bound={hw.prefillBound}
            ai={hw.prefillArithmeticIntensity}
            util={hw.prefillGpuUtil}
          />
          <BoundVerdict
            phase="Decode"
            bound={hw.decodeBound}
            ai={hw.decodeArithmeticIntensity}
            util={hw.decodeGpuUtil}
          />
        </div>

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Prefill arith. intensity"
            value={`${hw.prefillArithmeticIntensity.toFixed(1)} FLOPs/byte`}
            sub={`Ridge point: ${hw.ridgePointFlopsPerByte.toFixed(1)} FLOPs/byte`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Decode arith. intensity"
            value={`${hw.decodeArithmeticIntensity.toFixed(1)} FLOPs/byte`}
            sub={hw.decodeBound === "memory" ? "Below ridge → memory-bound" : "Above ridge → compute-bound"}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Prefill GPU utilization"
            value={`${hw.prefillGpuUtil.toFixed(1)}%`}
            sub={`${hw.prefillAchievableTflops.toFixed(0)} / ${hw.peakTflops.toFixed(0)} TFLOPS`}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Decode GPU utilization"
            value={`${hw.decodeGpuUtil.toFixed(1)}%`}
            sub={`${hw.decodeAchievableTflops.toFixed(1)} / ${hw.peakTflops.toFixed(0)} TFLOPS`}
            accent={hw.decodeGpuUtil < 5 ? "border-l-kpiRose" : "border-l-kpiAmber"}
          />
        </div>

        <div className="mt-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4 grid grid-cols-2 gap-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Roofline model</div>
            <ul className="formula-list">
              <li>AI = FLOPs / bytes_accessed</li>
              <li>Achievable = min(peak_TFLOPS, memBW x AI)</li>
              <li>Ridge = peak_TFLOPS / memBW (transition point)</li>
              <li>Left of ridge = memory-bound, right = compute-bound</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Why decode is slow</div>
            <ul className="formula-list">
              <li>Decode processes 1 token at a time → low AI</li>
              <li>Must read all active weights per token from HBM</li>
              <li>GPU compute units are mostly idle waiting for memory</li>
              <li>Batching increases AI and improves decode utilization</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─── MFU Gauge ─────────────────────── */

const MfuGauge: React.FC<{
  label: string;
  sublabel: string;
  value: number;
}> = ({ label, sublabel, value }) => {
  const pct = value * 100;
  const color =
    pct >= 50 ? "text-emerald-400" : pct >= 30 ? "text-amber-400" : "text-rose-400";
  const bg =
    pct >= 50
      ? "bg-emerald-500/5 border-emerald-500/20"
      : pct >= 30
      ? "bg-amber-500/5 border-amber-500/20"
      : "bg-rose-500/5 border-rose-500/20";

  return (
    <div className={`rounded-xl border px-5 py-4 text-center ${bg}`}>
      <div className={`text-4xl font-black tracking-tight ${color}`}>
        {pct.toFixed(1)}%
      </div>
      <div className="text-sm font-bold mt-1">{label}</div>
      <div className="text-[10px] text-textMuted mt-0.5">{sublabel}</div>
    </div>
  );
};

/* ─── Bound Verdict Badge ───────────── */

const BoundVerdict: React.FC<{
  phase: string;
  bound: "compute" | "memory";
  ai: number;
  util: number;
}> = ({ phase, bound, ai, util }) => {
  const isCompute = bound === "compute";
  return (
    <div
      className={`rounded-xl border px-4 py-3 flex items-center gap-3 ${
        isCompute
          ? "bg-blue-500/5 border-blue-500/20"
          : "bg-amber-500/5 border-amber-500/20"
      }`}
    >
      <div className={`text-2xl ${isCompute ? "text-blue-400" : "text-amber-400"}`}>
        {isCompute ? "\u26A1" : "\uD83D\uDCBE"}
      </div>
      <div>
        <div className="font-bold text-[13px]">
          {phase}:{" "}
          <span className={isCompute ? "text-blue-400" : "text-amber-400"}>
            {isCompute ? "COMPUTE-BOUND" : "MEMORY-BOUND"}
          </span>
        </div>
        <div className="text-[10px] text-textMuted mt-0.5">
          AI = {ai.toFixed(1)} FLOPs/byte · GPU util {util.toFixed(1)}%
        </div>
      </div>
    </div>
  );
};

/* ─── Roofline Plot ─────────────────── */

const RooflinePlot: React.FC<{
  hw: HardwareEfficiencyMetrics;
  gpuName: string;
}> = ({ hw, gpuName }) => {
  const curveData = hw.rooflineCurve;

  const prefillPoint = hw.rooflinePoints.find((p) => p.label === "Prefill")!;
  const decodePoint = hw.rooflinePoints.find((p) => p.label === "Decode")!;

  const scatterData = [
    { ai: prefillPoint.ai, tflops: prefillPoint.tflops, label: "Prefill" },
    { ai: decodePoint.ai, tflops: decodePoint.tflops, label: "Decode" }
  ];

  const COLORS = ["#38bdf8", "#fbbf24"];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart
        data={curveData}
        margin={{ top: 16, right: 40, bottom: 24, left: 50 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
        <XAxis
          dataKey="ai"
          type="number"
          scale="log"
          domain={["auto", "auto"]}
          tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}K` : v >= 1 ? v.toFixed(0) : v.toFixed(1)}
          stroke="#8494b0"
          fontSize={10}
          label={{
            value: "Arithmetic Intensity (FLOPs/byte)",
            position: "insideBottom",
            offset: -12,
            fill: "#8494b0",
            fontSize: 10
          }}
        />
        <YAxis
          dataKey="tflops"
          type="number"
          scale="log"
          domain={["auto", "auto"]}
          tickFormatter={(v) => v >= 1 ? `${v.toFixed(0)}` : v.toFixed(2)}
          stroke="#8494b0"
          fontSize={10}
          label={{
            value: "TFLOPS",
            angle: -90,
            position: "insideLeft",
            offset: -5,
            fill: "#8494b0",
            fontSize: 10
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0a0f1e",
            border: "1px solid #1a2540",
            borderRadius: "0.75rem",
            fontSize: 11,
            boxShadow: "0 12px 30px rgba(0,0,0,0.5)"
          }}
          formatter={(value: number, name: string) => {
            if (name === "Roofline") return [`${value.toFixed(1)} TFLOPS`, "Roofline"];
            return [`${value.toFixed(2)} TFLOPS`, name];
          }}
          labelFormatter={(v) => `AI: ${Number(v).toFixed(1)} FLOPs/byte`}
        />
        <Legend />
        <ReferenceLine
          x={hw.ridgePointFlopsPerByte}
          stroke="#6366f1"
          strokeDasharray="4 4"
          strokeWidth={1}
          label={{
            value: "Ridge",
            position: "top",
            fill: "#6366f1",
            fontSize: 9
          }}
        />
        <Line
          dataKey="tflops"
          name="Roofline"
          stroke="#34d399"
          strokeWidth={2.5}
          dot={false}
          activeDot={false}
          isAnimationActive={false}
        />
        <Scatter
          data={scatterData}
          name="Operating points"
          dataKey="tflops"
          isAnimationActive={false}
        >
          {scatterData.map((_entry, index) => (
            <Cell key={index} fill={COLORS[index]} r={7} />
          ))}
        </Scatter>
      </ComposedChart>
    </ResponsiveContainer>
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
