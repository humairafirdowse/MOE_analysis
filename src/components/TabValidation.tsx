import React, { useMemo } from "react";
import { formatBigNumber, useConfigStore, computeOverview } from "../state/useConfigStore";
import { getGpuSpec } from "../lib/gpus";
import {
  computeTrainingComputeMetrics,
  computeTrainingMemoryMetrics,
  computeInferenceMetrics
} from "../lib/metrics";

interface ValidationRow {
  metric: string;
  unit: string;
  ourValue: number | null;
  paperValue: number | null;
  /** Optional: format our value differently for display (e.g. "2.79M") */
  formatOur?: (v: number) => string;
  /** Optional: format paper value differently */
  formatPaper?: (v: number) => string;
}

function classifyDelta(delta: number | null): { status: "pass" | "warn" | "fail" | "na"; label: string } {
  if (delta === null || !Number.isFinite(delta)) {
    return { status: "na", label: "N/A" };
  }
  const pct = Math.abs(delta);
  if (pct <= 0.05) return { status: "pass", label: "✅ PASS" };
  if (pct <= 0.15) return { status: "warn", label: "⚠️ WARN" };
  return { status: "fail", label: "❌ FAIL" };
}

function formatGpuHours(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toFixed(0);
}

function formatKB(v: number): string {
  return v.toFixed(1) + " KB";
}

function formatGB(v: number): string {
  return v.toFixed(2) + " GB";
}

function formatTB(v: number): string {
  return v.toFixed(1) + " TB";
}

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
    () =>
      computeTrainingComputeMetrics(
        draftModel,
        draftMoe,
        draftTraining,
        trainingGpu,
        mfu
      ),
    [draftModel, draftMoe, draftTraining, trainingGpu, mfu]
  );
  const trainingMemory = useMemo(
    () =>
      computeTrainingMemoryMetrics(
        draftModel,
        draftMoe,
        draftOverview,
        draftTraining
      ),
    [draftModel, draftMoe, draftOverview, draftTraining]
  );
  const inferenceMetrics = useMemo(
    () =>
      computeInferenceMetrics(
        draftModel,
        draftMoe,
        draftOverview,
        draftInference,
        inferenceGpu
      ),
    [draftModel, draftMoe, draftOverview, draftInference, inferenceGpu]
  );

  const rows: ValidationRow[] = useMemo(() => {
    const totalParamsB = draftOverview.totalParams / 1e9;
    const activeParamsB = draftOverview.activeParams / 1e9;
    const kvCachePerTokenKB = inferenceMetrics.kvBytesPerToken / 1024;
    const weightsGB = inferenceMetrics.weightsBytes / 1e9;
    const peakTrainingGB = trainingMemory.peakBytes / 1e9;
    const peakTrainingTB = trainingMemory.peakBytes / 1e12;

    const result: ValidationRow[] = [
      {
        metric: "Total Parameters",
        unit: "B",
        ourValue: totalParamsB,
        paperValue: paperReference?.totalParamsB ?? null
      },
      {
        metric: "Active Parameters",
        unit: "B",
        ourValue: activeParamsB,
        paperValue: paperReference?.activeParamsB ?? null
      },
      {
        metric: "GPU Hours (training)",
        unit: "GPU·h",
        ourValue: trainingCompute.gpuHoursApprox,
        paperValue: paperReference?.gpuHoursReported ?? null,
        formatOur: formatGpuHours,
        formatPaper: formatGpuHours
      },
      {
        metric: "KV Cache per token",
        unit: "KB",
        ourValue: kvCachePerTokenKB,
        paperValue: paperReference?.kvCachePerTokenKB ?? null,
        formatOur: formatKB,
        formatPaper: formatKB
      },
      {
        metric: "Inference weights",
        unit: "GB",
        ourValue: weightsGB,
        paperValue: paperReference?.inferenceWeightsGB ?? null,
        formatOur: formatGB,
        formatPaper: formatGB
      }
    ];

    if (paperReference?.trainingMemoryTB != null) {
      result.push({
        metric: "Training memory (peak)",
        unit: "TB",
        ourValue: peakTrainingTB,
        paperValue: paperReference.trainingMemoryTB,
        formatOur: formatTB,
        formatPaper: formatTB
      });
    } else if (paperReference?.trainingMemoryGB != null) {
      result.push({
        metric: "Training memory (peak)",
        unit: "GB",
        ourValue: peakTrainingGB,
        paperValue: paperReference.trainingMemoryGB,
        formatOur: formatGB,
        formatPaper: formatGB
      });
    } else {
      result.push({
        metric: "Training memory (peak)",
        unit: "GB",
        ourValue: peakTrainingGB,
        paperValue: paperReference?.trainingMemoryGB ?? null,
        formatOur: formatGB,
        formatPaper: formatGB
      });
    }

    return result;
  }, [draftOverview, paperReference, trainingCompute, trainingMemory, inferenceMetrics]);

  const deltas = rows.map((row) => {
    if (row.ourValue == null || row.paperValue == null || row.paperValue === 0) {
      return null;
    }
    return (row.ourValue - row.paperValue) / row.paperValue;
  });

  const statuses = deltas.map((d) => classifyDelta(d));

  const passCount = statuses.filter((s) => s.status === "pass").length;
  const warnCount = statuses.filter((s) => s.status === "warn").length;
  const failCount = statuses.filter((s) => s.status === "fail").length;
  const totalWithPaper = statuses.filter((s) => s.status !== "na").length;

  const overallStatus =
    failCount > 0
      ? "❌ Some metrics significantly deviate (>15%)."
      : warnCount > 0
      ? "⚠️ Small deviations detected (5–15%)."
      : totalWithPaper > 0
      ? "✅ All checked metrics within 5% of paper."
      : "No paper reference values available for this preset yet.";

  const formatOurDisplay = (row: ValidationRow) => {
    if (row.ourValue == null) return "N/A";
    return row.formatOur ? row.formatOur(row.ourValue) : `${row.ourValue.toFixed(2)} ${row.unit}`;
  };
  const formatPaperDisplay = (row: ValidationRow) => {
    if (row.paperValue == null) return "N/A";
    return row.formatPaper ? row.formatPaper(row.paperValue) : `${row.paperValue.toFixed(2)} ${row.unit}`;
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Validation
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Compare our theoretical calculations to reported values from papers and official sources.
            </div>
          </div>
          <div className="text-right text-[11px] text-textMuted">
            Paper preset:{" "}
            <span className="font-semibold">
              {paperReference?.name ?? "N/A"}
            </span>
            <div className="mt-0.5">
              (Selected via global preset: <span className="uppercase">{preset}</span>)
            </div>
          </div>
        </div>

        <div className="section-card bg-card/40 border-dashed border-borderSoft/50">
          <div className="text-[13px] font-medium mb-1">Accuracy scorecard</div>
          <div className="text-[12px] text-textMuted">
            {totalWithPaper > 0 ? (
              <>
                <span className="font-semibold">
                  {passCount}/{totalWithPaper} metrics PASS
                </span>
                {warnCount > 0 && (
                  <span>
                    , {warnCount} WARN
                    {failCount > 0 && <> , {failCount} FAIL</>}
                  </span>
                )}
                {failCount === 0 && warnCount === 0 && <span>, 0 WARN / 0 FAIL</span>}
                <span className="ml-2">{overallStatus}</span>
              </>
            ) : (
              <span>
                Select a preset with known paper values (e.g. DeepSeek-V3) in the sidebar
                to enable validation.
              </span>
            )}
          </div>
        </div>

        <div className="overflow-x-auto mt-3">
          <table className="min-w-full text-[12px]">
            <thead>
              <tr className="text-left border-b border-borderSoft/60">
                <th className="py-1.5 pr-3 font-medium text-textMuted">Metric</th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">Our Calculation</th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">
                  Paper / Reported
                </th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">Delta (%)</th>
                <th className="py-1.5 pr-3 font-medium text-textMuted">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const delta = deltas[idx];
                const status = statuses[idx];
                const deltaPct =
                  delta == null || !Number.isFinite(delta) ? "N/A" : (delta * 100).toFixed(2);

                return (
                  <tr
                    key={row.metric}
                    className="border-b border-borderSoft/30 last:border-b-0"
                  >
                    <td className="py-1.5 pr-3">{row.metric}</td>
                    <td className="py-1.5 pr-3 text-textMuted">
                      {formatOurDisplay(row)}
                    </td>
                    <td className="py-1.5 pr-3 text-textMuted">
                      {formatPaperDisplay(row)}
                    </td>
                    <td className="py-1.5 pr-3">{deltaPct}</td>
                    <td className="py-1.5 pr-3">
                      <span
                        className={
                          status.status === "pass"
                            ? "text-emerald-400"
                            : status.status === "warn"
                            ? "text-amber-300"
                            : status.status === "fail"
                            ? "text-rose-400"
                            : "text-textMuted"
                        }
                      >
                        {status.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-3 text-[11px] text-textMuted">
          <div className="font-semibold mb-1">Notes on potential deviation</div>
          <ul className="list-disc list-inside space-y-0.5">
            <li>
              Papers sometimes count embedding and output head parameters differently (e.g.,
              tied vs untied); this can introduce a small offset.
            </li>
            <li>
              &quot;Active params&quot; definitions vary: some include all layers, others only
              MoE experts; MLA (DeepSeek) uses low-rank attention in the active path.
            </li>
            <li>
              Hybrid architectures (e.g. Snowflake Arctic) may require formula
              extensions we have not yet implemented.
            </li>
            <li>
              GPU hours: our estimate uses the configured training GPU and MFU; papers report actual cluster usage.
              Inference weights: papers often cite bf16/fp16; our value depends on selected precision.
            </li>
            <li>
              KV cache: DeepSeek-V2 paper reports 93.3% reduction vs dense; our MLA formula
              matches (d_c + d_R_h) × layers. Snowflake Arctic training memory is total model state.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
