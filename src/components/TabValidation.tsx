import React, { useMemo } from "react";
import { formatBigNumber, useConfigStore, computeOverview } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import {
  computeTrainingComputeMetrics,
  computeTrainingMemoryMetrics,
  computeInferenceMetrics,
  forwardFlopsPerToken
} from "../lib/metrics";

interface ValidationRow {
  metric: string;
  category: "params" | "compute" | "memory" | "inference";
  unit: string;
  ourValue: number | null;
  paperValue: number | null;
  formatOur?: (v: number) => string;
  formatPaper?: (v: number) => string;
}

function classifyDelta(delta: number | null): { status: "pass" | "warn" | "fail" | "na"; label: string; color: string } {
  if (delta === null || !Number.isFinite(delta)) {
    return { status: "na", label: "N/A", color: "text-textMuted/50" };
  }
  const pct = Math.abs(delta);
  if (pct <= 0.05) return { status: "pass", label: "PASS", color: "text-emerald-400" };
  if (pct <= 0.15) return { status: "warn", label: "WARN", color: "text-amber-400" };
  return { status: "fail", label: "FAIL", color: "text-rose-400" };
}

function fmtGpuHours(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toFixed(0);
}
function fmtKB(v: number): string { return v.toFixed(1) + " KB"; }
function fmtGB(v: number): string { return v.toFixed(2) + " GB"; }
function fmtTB(v: number): string { return v.toFixed(1) + " TB"; }
function fmtUSD(v: number): string {
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(0) + "K";
  return "$" + v.toFixed(0);
}
function fmtB(v: number): string { return v.toFixed(2) + " B"; }
function fmtM(v: number): string { return v.toFixed(1) + " M"; }
function fmtInt(v: number): string { return v.toLocaleString(); }
function fmtFlops(v: number): string { return formatBigNumber(v) + " FLOPs"; }

const CATEGORY_LABELS: Record<string, { label: string; color: string }> = {
  params: { label: "Parameters", color: "text-sectionArch" },
  compute: { label: "Training Compute", color: "text-sectionTrain" },
  memory: { label: "Training Memory", color: "text-sectionMoe" },
  inference: { label: "Inference", color: "text-sectionInfer" }
};

export const TabValidation: React.FC = () => {
  const {
    paperReference,
    preset,
    draftModel,
    draftMoe,
    draftTraining,
    draftInference
  } = useConfigStore();
  const draftOverview = useMemo(
    () => computeOverview(draftModel, draftMoe),
    [draftModel, draftMoe]
  );
  const trainingGpuType = draftTraining.trainingGpuType ?? "H800-80G";
  const mfu = draftTraining.mfu ?? 0.55;
  const trainingGpu = getGpuSpec(trainingGpuType);
  const inferenceGpu = getGpuSpec(draftInference.gpuType);

  const trainingCompute = useMemo(
    () => computeTrainingComputeMetrics(draftModel, draftMoe, draftTraining, trainingGpu, mfu),
    [draftModel, draftMoe, draftTraining, trainingGpu, mfu]
  );
  const trainingMemory = useMemo(
    () => computeTrainingMemoryMetrics(draftModel, draftMoe, draftOverview, draftTraining),
    [draftModel, draftMoe, draftOverview, draftTraining]
  );
  const inferenceMetrics = useMemo(
    () => computeInferenceMetrics(draftModel, draftMoe, draftOverview, draftInference, inferenceGpu),
    [draftModel, draftMoe, draftOverview, draftInference, inferenceGpu]
  );

  const fwdFlops = useMemo(
    () => forwardFlopsPerToken(draftModel, draftMoe, draftModel.maxSeqLen),
    [draftModel, draftMoe]
  );

  const numDenseLayers = draftModel.firstKDenseReplace ?? 0;
  const moeLayers = draftModel.layersMoe ?? Math.max(0, draftModel.layers - numDenseLayers);
  const perExpertPerLayer = 3 * draftModel.dModel * draftMoe.dFf;
  const expertParamsPerExpertB = (perExpertPerLayer * moeLayers) / 1e9;

  const rows: ValidationRow[] = useMemo(() => {
    const totalParamsB = draftOverview.totalParams / 1e9;
    const activeParamsB = draftOverview.activeParams / 1e9;
    const embeddingParamsM = draftOverview.embeddingParams / 1e6;
    const kvCachePerTokenKB = inferenceMetrics.kvBytesPerToken / 1024;
    const weightsGB = inferenceMetrics.weightsBytes / 1e9;
    const peakTrainingGB = trainingMemory.peakBytes / 1e9;
    const peakTrainingTB = trainingMemory.peakBytes / 1e12;
    const forwardFlopsPerTokenVal = fwdFlops.total;

    const result: ValidationRow[] = [
      {
        metric: "Total parameters",
        category: "params",
        unit: "B",
        ourValue: totalParamsB,
        paperValue: paperReference?.totalParamsB ?? null,
        formatOur: fmtB,
        formatPaper: fmtB
      },
      {
        metric: "Active parameters",
        category: "params",
        unit: "B",
        ourValue: activeParamsB,
        paperValue: paperReference?.activeParamsB ?? null,
        formatOur: fmtB,
        formatPaper: fmtB
      },
      {
        metric: "Embedding parameters",
        category: "params",
        unit: "M",
        ourValue: embeddingParamsM,
        paperValue: paperReference?.embeddingParamsM ?? null,
        formatOur: fmtM,
        formatPaper: fmtM
      },
      {
        metric: "Expert params / expert",
        category: "params",
        unit: "B",
        ourValue: expertParamsPerExpertB,
        paperValue: paperReference?.expertParamsPerExpertB ?? null,
        formatOur: fmtB,
        formatPaper: fmtB
      },
      {
        metric: "Number of GPUs",
        category: "compute",
        unit: "",
        ourValue: trainingCompute.totalGpus,
        paperValue: paperReference?.numGpus ?? null,
        formatOur: fmtInt,
        formatPaper: fmtInt
      },
      {
        metric: "GPU-hours (training)",
        category: "compute",
        unit: "GPU·h",
        ourValue: trainingCompute.gpuHoursApprox,
        paperValue: paperReference?.gpuHoursReported ?? null,
        formatOur: fmtGpuHours,
        formatPaper: fmtGpuHours
      },
      {
        metric: "Training cost",
        category: "compute",
        unit: "USD",
        ourValue: trainingCompute.trainingCostUSD,
        paperValue: paperReference?.trainingCostUSD ?? null,
        formatOur: fmtUSD,
        formatPaper: fmtUSD
      },
      {
        metric: "Forward FLOPs / token",
        category: "compute",
        unit: "FLOPs",
        ourValue: forwardFlopsPerTokenVal,
        paperValue: paperReference?.forwardFlopsPerTokenB
          ? paperReference.forwardFlopsPerTokenB * 1e9
          : null,
        formatOur: fmtFlops,
        formatPaper: fmtFlops
      },
      {
        metric: "Total training FLOPs",
        category: "compute",
        unit: "FLOPs",
        ourValue: trainingCompute.totalTrainingFlops,
        paperValue: paperReference?.totalTrainingFlopsPF
          ? paperReference.totalTrainingFlopsPF * 1e15
          : null,
        formatOur: fmtFlops,
        formatPaper: fmtFlops
      }
    ];

    if (paperReference?.trainingMemoryTB != null) {
      result.push({
        metric: "Training memory (peak)",
        category: "memory",
        unit: "TB",
        ourValue: peakTrainingTB,
        paperValue: paperReference.trainingMemoryTB,
        formatOur: fmtTB,
        formatPaper: fmtTB
      });
    } else {
      result.push({
        metric: "Training memory (peak)",
        category: "memory",
        unit: "GB",
        ourValue: peakTrainingGB,
        paperValue: paperReference?.trainingMemoryGB ?? null,
        formatOur: fmtGB,
        formatPaper: fmtGB
      });
    }

    result.push(
      {
        metric: "KV cache / token",
        category: "inference",
        unit: "KB",
        ourValue: kvCachePerTokenKB,
        paperValue: paperReference?.kvCachePerTokenKB ?? null,
        formatOur: fmtKB,
        formatPaper: fmtKB
      },
      {
        metric: "Inference weights",
        category: "inference",
        unit: "GB",
        ourValue: weightsGB,
        paperValue: paperReference?.inferenceWeightsGB ?? null,
        formatOur: fmtGB,
        formatPaper: fmtGB
      }
    );

    return result;
  }, [draftOverview, paperReference, trainingCompute, trainingMemory, inferenceMetrics, fwdFlops, expertParamsPerExpertB]);

  const deltas = rows.map((row) => {
    if (row.ourValue == null || row.paperValue == null || row.paperValue === 0) return null;
    return (row.ourValue - row.paperValue) / row.paperValue;
  });

  const statuses = deltas.map((d) => classifyDelta(d));
  const passCount = statuses.filter((s) => s.status === "pass").length;
  const warnCount = statuses.filter((s) => s.status === "warn").length;
  const failCount = statuses.filter((s) => s.status === "fail").length;
  const totalWithPaper = statuses.filter((s) => s.status !== "na").length;

  const formatOurDisplay = (row: ValidationRow) => {
    if (row.ourValue == null) return "—";
    return row.formatOur ? row.formatOur(row.ourValue) : `${row.ourValue.toFixed(2)} ${row.unit}`;
  };
  const formatPaperDisplay = (row: ValidationRow) => {
    if (row.paperValue == null) return "—";
    return row.formatPaper ? row.formatPaper(row.paperValue) : `${row.paperValue.toFixed(2)} ${row.unit}`;
  };

  const categories = ["params", "compute", "memory", "inference"] as const;

  return (
    <div className="flex flex-col gap-5">
      <div className="section-card">
        <SectionTitle
          title="Validation"
          subtitle="Compare theoretical calculations to reported values from papers and documentation"
          right={
            <div className="text-right text-[11px] text-textMuted/60">
              Preset: <span className="font-semibold text-white/60">{paperReference?.name ?? "N/A"}</span>
            </div>
          }
        />

        {/* Scorecard */}
        <div className={`mt-4 rounded-xl border px-4 py-3 ${
          failCount > 0
            ? "border-rose-500/30 bg-rose-500/5"
            : warnCount > 0
            ? "border-amber-500/30 bg-amber-500/5"
            : totalWithPaper > 0
            ? "border-emerald-500/30 bg-emerald-500/5"
            : "border-borderSoft/30 bg-surface/30"
        }`}>
          {totalWithPaper > 0 ? (
            <div className="flex items-center gap-3 text-[13px]">
              <span className={`text-2xl font-bold ${failCount > 0 ? "text-rose-400" : warnCount > 0 ? "text-amber-400" : "text-emerald-400"}`}>
                {passCount}/{totalWithPaper}
              </span>
              <div>
                <div className="font-bold">
                  {passCount} PASS
                  {warnCount > 0 && <span className="text-amber-400 ml-1.5">| {warnCount} WARN</span>}
                  {failCount > 0 && <span className="text-rose-400 ml-1.5">| {failCount} FAIL</span>}
                  <span className="text-textMuted/50 ml-1.5">| {rows.length - totalWithPaper} no ref</span>
                </div>
                <div className="text-[11px] text-textMuted mt-0.5">
                  {failCount > 0
                    ? "Some metrics deviate >15% from paper values — review architecture settings"
                    : warnCount > 0
                    ? "Small deviations (5-15%) detected — may be rounding or counting differences"
                    : "All checked metrics within 5% of paper / reported values"}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-[12px] text-textMuted">
              Select a preset with known paper values (e.g. DeepSeek-V3) to enable validation.
            </div>
          )}
        </div>

        {/* Table grouped by category */}
        <div className="overflow-x-auto mt-4">
          <table className="min-w-full text-[12px]">
            <thead>
              <tr className="border-b border-borderSoft/40">
                <th className="py-2 pr-4 text-left font-bold text-[10px] uppercase tracking-wider text-textMuted w-[200px]">Metric</th>
                <th className="py-2 pr-4 text-left font-bold text-[10px] uppercase tracking-wider text-textMuted">Calculated</th>
                <th className="py-2 pr-4 text-left font-bold text-[10px] uppercase tracking-wider text-textMuted">Paper / Reported</th>
                <th className="py-2 pr-4 text-right font-bold text-[10px] uppercase tracking-wider text-textMuted w-[80px]">Delta</th>
                <th className="py-2 text-center font-bold text-[10px] uppercase tracking-wider text-textMuted w-[60px]">Status</th>
              </tr>
            </thead>
            <tbody>
              {categories.map((cat) => {
                const catRows = rows
                  .map((r, i) => ({ row: r, idx: i }))
                  .filter(({ row }) => row.category === cat);
                if (catRows.length === 0) return null;
                const catInfo = CATEGORY_LABELS[cat];
                return (
                  <React.Fragment key={cat}>
                    <tr>
                      <td
                        colSpan={5}
                        className={`pt-3.5 pb-1.5 text-[10px] font-bold uppercase tracking-widest ${catInfo.color}`}
                      >
                        {catInfo.label}
                      </td>
                    </tr>
                    {catRows.map(({ row, idx }) => {
                      const delta = deltas[idx];
                      const status = statuses[idx];
                      const deltaPct =
                        delta == null || !Number.isFinite(delta)
                          ? "—"
                          : (delta >= 0 ? "+" : "") + (delta * 100).toFixed(2) + "%";

                      return (
                        <tr
                          key={row.metric}
                          className="border-b border-borderSoft/15 last:border-b-0"
                        >
                          <td className="py-2 pr-4 font-medium text-white/80">{row.metric}</td>
                          <td className="py-2 pr-4 text-textMuted font-mono text-[11px]">
                            {formatOurDisplay(row)}
                          </td>
                          <td className="py-2 pr-4 font-mono text-[11px]">
                            {row.paperValue != null ? (
                              <span className="text-white/70">{formatPaperDisplay(row)}</span>
                            ) : (
                              <span className="text-textMuted/40">—</span>
                            )}
                          </td>
                          <td className="py-2 pr-4 text-right font-mono text-[11px]">
                            <span className={
                              delta == null ? "text-textMuted/40"
                              : Math.abs(delta) <= 0.05 ? "text-emerald-400/80"
                              : Math.abs(delta) <= 0.15 ? "text-amber-400/80"
                              : "text-rose-400/80"
                            }>
                              {deltaPct}
                            </span>
                          </td>
                          <td className="py-2 text-center">
                            <span
                              className={`inline-block px-2 py-0.5 rounded-full text-[9px] font-bold tracking-wide ${status.color} ${
                                status.status === "pass"
                                  ? "bg-emerald-500/10"
                                  : status.status === "warn"
                                  ? "bg-amber-500/10"
                                  : status.status === "fail"
                                  ? "bg-rose-500/10"
                                  : "bg-borderSoft/15"
                              }`}
                            >
                              {status.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-[11px] text-textMuted border-t border-borderSoft/30 pt-4">
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">What we validate</div>
            <ul className="formula-list">
              <li><strong>Params:</strong> Total, active, embedding, per-expert counts</li>
              <li><strong>Training Compute:</strong> GPU count, GPU-hours, training cost, FLOPs</li>
              <li><strong>Training Memory:</strong> Peak training memory (params + opt + grad + act)</li>
              <li><strong>Inference:</strong> KV cache per token, model weights in chosen precision</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-white/70 mb-1.5 text-[10px] uppercase tracking-wider">Deviation sources</div>
            <ul className="formula-list">
              <li>Rounding in reported "B" figures (e.g. 671B is approximate)</li>
              <li>Tied vs untied embedding/output head counting</li>
              <li>MFU estimates differ between theory and cluster reality</li>
              <li>MLA caches (d_c + d_R_h) per layer, not separate K+V</li>
              <li>Training cost depends on GPU rental rate assumptions</li>
            </ul>
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
