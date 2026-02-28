import React, { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import { useConfigStore, computeOverview, formatBigNumber } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import {
  computeTrainingComputeMetrics,
  computeCostAnalysisMetrics
} from "../lib/metrics";

function fmtUSD(v: number): string {
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(1) + "K";
  if (v >= 1) return "$" + v.toFixed(2);
  if (v >= 0.001) return "$" + v.toFixed(4);
  return "$" + v.toExponential(2);
}

function fmtDuration(hours: number): string {
  if (hours < 24) return `${hours.toFixed(1)} hours`;
  const days = hours / 24;
  if (days < 7) return `${days.toFixed(1)} days`;
  const weeks = days / 7;
  if (weeks < 8) return `${weeks.toFixed(1)} weeks`;
  const months = days / 30;
  return `${months.toFixed(1)} months`;
}

export const TabCostAnalysis: React.FC = () => {
  const { draftModel, draftMoe, draftTraining, draftInference, paperReference } =
    useConfigStore();

  const overview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );

  const trainingGpu = getGpuSpec(draftTraining.trainingGpuType ?? "H800-80G");
  const inferenceGpu = getGpuSpec(draftInference.gpuType);
  const mfu = draftTraining.mfu ?? 0.55;

  const trainingCompute = useMemo(
    () => computeTrainingComputeMetrics(draftModel, draftMoe, draftTraining, trainingGpu, mfu),
    [draftModel, draftMoe, draftTraining, trainingGpu, mfu]
  );

  const cost = useMemo(
    () =>
      computeCostAnalysisMetrics(
        draftModel, draftMoe, overview, draftTraining,
        trainingGpu, mfu, draftInference, inferenceGpu, trainingCompute
      ),
    [draftModel, draftMoe, overview, draftTraining, trainingGpu, mfu, draftInference, inferenceGpu, trainingCompute]
  );

  // Monthly serving estimation — local state
  const [reqPerDay, setReqPerDay] = useState(100000);
  const [avgInputTok, setAvgInputTok] = useState(1000);
  const [avgOutputTok, setAvgOutputTok] = useState(500);

  const monthlyServing = useMemo(() => {
    const inputTokensPerDay = reqPerDay * avgInputTok;
    const outputTokensPerDay = reqPerDay * avgOutputTok;
    const dailyInputCost = (inputTokensPerDay / 1e6) * cost.costPer1MInputTokens;
    const dailyOutputCost = (outputTokensPerDay / 1e6) * cost.costPer1MOutputTokens;
    const dailyCost = dailyInputCost + dailyOutputCost;
    const monthlyCost = dailyCost * 30;

    const secPerInputToken = cost.prefillTokensPerSec > 0 ? 1 / cost.prefillTokensPerSec : 0;
    const secPerOutputToken = cost.decodeTokensPerSec > 0 ? 1 / cost.decodeTokensPerSec : 0;
    const secPerReq = secPerInputToken * avgInputTok + secPerOutputToken * avgOutputTok;
    const reqPerSec = reqPerDay / 86400;
    const gpusNeeded = Math.ceil(reqPerSec * secPerReq);

    return { dailyCost, monthlyCost, gpusNeeded, secPerReq };
  }, [reqPerDay, avgInputTok, avgOutputTok, cost]);

  const fmtGpuHours = (h: number) =>
    h >= 1e6 ? (h / 1e6).toFixed(2) + "M" : h >= 1e3 ? (h / 1e3).toFixed(1) + "K" : h.toFixed(0);

  // Batch sweep chart data
  const batchChartData = cost.batchSweep.map((b) => ({
    batch: `B=${b.batchSize}`,
    "Output $/1M tok": Number(b.costPer1MOutput.toFixed(4)),
    "tok/s": Math.round(b.decodeTokensPerSec)
  }));

  return (
    <div className="flex flex-col gap-5">
      {/* ══════════════ Section A: Training Cost ══════════════ */}
      <div className="section-card">
        <SectionTitle
          title="Training Cost"
          subtitle="GPU-hours, wall-clock time, and total cost for the full training run"
          right={
            <div className="text-right text-[11px] text-textMuted/60">
              <span className="font-semibold text-white/60">
                {trainingGpu.name.replace("NVIDIA ", "")}
              </span>
              <div className="mt-0.5">${cost.costPerGpuHour}/GPU-hr</div>
            </div>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="GPU-hours"
            value={fmtGpuHours(cost.gpuHours) + " GPU·h"}
            sub={`${cost.totalGpus.toLocaleString()} GPUs`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Total training cost"
            value={fmtUSD(cost.totalTrainingCost)}
            sub={`${fmtGpuHours(cost.gpuHours)} h × $${cost.costPerGpuHour}/h`}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Cost / trillion tokens"
            value={fmtUSD(cost.costPerTrillionTokens)}
            sub={`${formatBigNumber(cost.totalTrainingTokens)} training tokens`}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Wall-clock time"
            value={fmtDuration(cost.wallClockHours)}
            sub={`${cost.wallClockDays.toFixed(1)} days on ${cost.totalGpus.toLocaleString()} GPUs`}
            accent="border-l-kpiPurple"
          />
        </div>

        {/* Grounding comparison */}
        <div className="mt-4 rounded-xl border border-accent/20 bg-accent/5 px-4 py-3">
          <div className="text-[12px]">
            <span className="font-bold text-accent">Real-world context:</span>{" "}
            Training{" "}
            <span className="font-semibold text-white/80">
              {paperReference?.name ?? "this model"}
            </span>{" "}
            ({formatBigNumber(overview.totalParams)} params) on{" "}
            <span className="font-semibold text-white/80">
              {cost.totalGpus.toLocaleString()} {trainingGpu.name.replace("NVIDIA ", "")}s
            </span>{" "}
            for{" "}
            <span className="font-semibold text-white/80">
              {fmtDuration(cost.wallClockHours)}
            </span>{" "}
            costs roughly{" "}
            <span className="font-bold text-accent text-[14px]">
              {fmtUSD(cost.totalTrainingCost)}
            </span>{" "}
            at ${cost.costPerGpuHour}/GPU-hr.
          </div>
          {paperReference?.trainingCostUSD != null && (
            <div className="text-[11px] text-textMuted mt-1.5">
              Paper reported: {fmtUSD(paperReference.trainingCostUSD)}
              {" — "}delta:{" "}
              {((cost.totalTrainingCost - paperReference.trainingCostUSD) / paperReference.trainingCostUSD * 100).toFixed(1)}%
            </div>
          )}
        </div>

        <div className="mt-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-3">
          <div className="font-bold text-white/70 mb-1 text-[10px] uppercase tracking-wider">Notes</div>
          <ul className="formula-list">
            <li>Cost = GPU-hours x $/GPU-hr. Rates vary: H100 on AWS ~$4-6/hr, Lambda ~$2-3/hr, reserved ~$1.5/hr</li>
            <li>Wall-clock = GPU-hours / num_GPUs. Assumes perfect linear scaling</li>
            <li>Real costs add networking, storage, engineering, and idle time overhead</li>
          </ul>
        </div>
      </div>

      {/* ══════════════ Section B: Inference Cost ══════════════ */}
      <div className="section-card">
        <SectionTitle
          title="Inference Cost"
          subtitle="Per-token serving cost for prefill (input) and decode (output)"
          right={
            <div className="text-right text-[11px] text-textMuted/60">
              <span className="font-semibold text-white/60">
                {inferenceGpu.name.replace("NVIDIA ", "")}
              </span>
              <div className="mt-0.5">
                {draftInference.precision.toUpperCase()} · B={draftInference.batchSize} · ${inferenceGpu.costPerHourUSD}/hr
              </div>
            </div>
          }
        />

        <div className="kpi-grid mt-4">
          <KpiCard
            label="Cost / 1M input tokens"
            value={fmtUSD(cost.costPer1MInputTokens)}
            sub={`Prefill: ${Math.round(cost.prefillTokensPerSec).toLocaleString()} tok/s`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Cost / 1M output tokens"
            value={fmtUSD(cost.costPer1MOutputTokens)}
            sub={`Decode: ${Math.round(cost.decodeTokensPerSec).toLocaleString()} tok/s`}
            accent="border-l-kpiAmber"
          />
        </div>
      </div>

      {/* ── Batch size sweep ── */}
      <div className="section-card">
        <SectionTitle
          title="Cost vs Batch Size"
          subtitle="Larger batches amortize weight-loading cost — decode gets dramatically cheaper"
        />

        <div className="overflow-x-auto mt-3">
          <table className="min-w-full text-[12px]">
            <thead>
              <tr className="border-b border-borderSoft/40">
                <th className="py-2 pr-4 text-left font-bold text-[10px] uppercase tracking-wider text-textMuted">Batch</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">Decode tok/s</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">Output $/1M tok</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">Input $/1M tok</th>
                <th className="py-2 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">vs B=1</th>
              </tr>
            </thead>
            <tbody>
              {cost.batchSweep.map((b, idx) => {
                const baseline = cost.batchSweep[0];
                const savings = baseline.costPer1MOutput > 0
                  ? baseline.costPer1MOutput / b.costPer1MOutput
                  : 1;
                const isActive = b.batchSize === draftInference.batchSize;
                return (
                  <tr
                    key={b.batchSize}
                    className={`border-b border-borderSoft/15 ${isActive ? "bg-accent/5" : ""}`}
                  >
                    <td className="py-2 pr-4 font-mono font-medium">
                      {b.batchSize}
                      {isActive && (
                        <span className="ml-1.5 text-[9px] text-accent font-bold uppercase">current</span>
                      )}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-textMuted">
                      {Math.round(b.decodeTokensPerSec).toLocaleString()}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono">
                      {fmtUSD(b.costPer1MOutput)}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-textMuted">
                      {fmtUSD(b.costPer1MInput)}
                    </td>
                    <td className="py-2 text-right font-mono text-emerald-400/80">
                      {savings.toFixed(1)}x
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="h-56 mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={batchChartData} margin={{ top: 8, right: 24, bottom: 8, left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
              <XAxis dataKey="batch" stroke="#8494b0" fontSize={10} />
              <YAxis
                stroke="#8494b0"
                fontSize={10}
                tickFormatter={(v) => "$" + v.toFixed(2)}
                label={{
                  value: "$/1M output tokens",
                  angle: -90,
                  position: "insideLeft",
                  offset: -10,
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
              />
              <Legend />
              <Bar
                dataKey="Output $/1M tok"
                name="Output $/1M tokens"
                fill="#fbbf24"
                radius={[4, 4, 0, 0]}
                maxBarSize={50}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Quantization cost comparison ── */}
      <div className="section-card">
        <SectionTitle
          title="Quantization Cost Comparison"
          subtitle="Lower precision = smaller weights = faster decode = cheaper serving"
        />

        <div className="overflow-x-auto mt-3">
          <table className="min-w-full text-[12px]">
            <thead>
              <tr className="border-b border-borderSoft/40">
                <th className="py-2 pr-4 text-left font-bold text-[10px] uppercase tracking-wider text-textMuted">Precision</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">Weights</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">Decode tok/s</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">$/1M output</th>
                <th className="py-2 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted">vs FP16</th>
              </tr>
            </thead>
            <tbody>
              {cost.quantSweep.map((q) => {
                const isActive = q.precision === draftInference.precision;
                return (
                  <tr
                    key={q.precision}
                    className={`border-b border-borderSoft/15 ${isActive ? "bg-accent/5" : ""}`}
                  >
                    <td className="py-2 pr-4 font-bold">
                      {q.label}
                      {isActive && (
                        <span className="ml-1.5 text-[9px] text-accent font-bold uppercase">current</span>
                      )}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-textMuted">
                      {q.weightsGB.toFixed(0)} GB
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-textMuted">
                      {Math.round(q.decodeTokensPerSec).toLocaleString()}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono">
                      {fmtUSD(q.costPer1MOutput)}
                    </td>
                    <td className="py-2 text-right font-mono text-emerald-400/80">
                      {q.relativeToFp16.toFixed(1)}x cheaper
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Monthly serving estimate ── */}
      <div className="section-card">
        <SectionTitle
          title="Monthly Serving Cost Estimate"
          subtitle="Estimate GPU cost for a production workload"
        />

        <div className="grid grid-cols-3 gap-3 mt-4">
          <InlineInput
            label="Requests / day"
            value={reqPerDay}
            onChange={setReqPerDay}
          />
          <InlineInput
            label="Avg input tokens / req"
            value={avgInputTok}
            onChange={setAvgInputTok}
          />
          <InlineInput
            label="Avg output tokens / req"
            value={avgOutputTok}
            onChange={setAvgOutputTok}
          />
        </div>

        <div className="kpi-grid mt-4">
          <KpiCard
            label="GPUs needed"
            value={`${monthlyServing.gpusNeeded}`}
            sub={`${inferenceGpu.name.replace("NVIDIA ", "")} · ${draftInference.precision.toUpperCase()}`}
            accent="border-l-kpiBlue"
          />
          <KpiCard
            label="Daily cost"
            value={fmtUSD(monthlyServing.dailyCost)}
            sub={`${(reqPerDay).toLocaleString()} req/day`}
            accent="border-l-kpiAmber"
          />
          <KpiCard
            label="Monthly cost"
            value={fmtUSD(monthlyServing.monthlyCost)}
            sub={`${monthlyServing.gpusNeeded} GPUs × 30 days`}
            accent="border-l-kpiGreen"
          />
          <KpiCard
            label="Seconds per request"
            value={`${monthlyServing.secPerReq.toFixed(2)}s`}
            sub={`${avgInputTok} in + ${avgOutputTok} out tokens`}
            accent="border-l-kpiPurple"
          />
        </div>

        <div className="mt-4 rounded-xl border border-borderSoft/20 bg-surface/40 px-4 py-3 text-[12px] text-textMuted">
          <span className="font-bold text-white/70">Summary: </span>
          Serving{" "}
          <span className="text-white/80 font-semibold">
            {reqPerDay.toLocaleString()} req/day
          </span>{" "}
          ({avgInputTok} in + {avgOutputTok} out tokens each) on{" "}
          <span className="text-white/80 font-semibold">
            {monthlyServing.gpusNeeded} {inferenceGpu.name.replace("NVIDIA ", "")}
          </span>{" "}
          GPUs at ${inferenceGpu.costPerHourUSD}/hr ={" "}
          <span className="text-accent font-bold text-[14px]">
            {fmtUSD(monthlyServing.monthlyCost)}/month
          </span>
        </div>

        <div className="mt-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-3">
          <div className="font-bold text-white/70 mb-1 text-[10px] uppercase tracking-wider">Assumptions</div>
          <ul className="formula-list">
            <li>Single-stream per GPU (no request batching across users)</li>
            <li>Prefill is compute-bound, decode is memory-bound</li>
            <li>No overprovisioning, queuing, or cold-start overhead</li>
            <li>KV cache fits in GPU memory for the configured sequence lengths</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

/* ─── Inline number input for monthly estimate ── */

const InlineInput: React.FC<{
  label: string;
  value: number;
  onChange: (v: number) => void;
}> = ({ label, value, onChange }) => (
  <label className="block">
    <div className="text-[10px] font-semibold text-textMuted/80 mb-1 uppercase tracking-wider">
      {label}
    </div>
    <input
      type="number"
      className="w-full rounded-lg bg-background/80 border border-borderSoft/50
                 px-2.5 py-1.5 text-xs text-white
                 focus:outline-none focus:ring-1 focus:ring-accent/60 focus:border-accent/60
                 transition-colors"
      value={value}
      onChange={(e) => {
        const n = Number(e.target.value);
        if (!Number.isNaN(n) && n >= 0) onChange(n);
      }}
    />
  </label>
);

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
