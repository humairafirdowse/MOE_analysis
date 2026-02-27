import React, { useMemo } from "react";
import { formatBigNumber, useConfigStore } from "../state/useConfigStore";

interface ValidationRow {
  metric: string;
  unit: string;
  ourValue: number | null;
  paperValue: number | null;
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

export const TabValidation: React.FC = () => {
  const { overview, paperReference, preset } = useConfigStore();

  const rows: ValidationRow[] = useMemo(() => {
    const totalParamsB = overview.totalParams / 1e9;
    const activeParamsB = overview.activeParams / 1e9;

    return [
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
      }
      // Training FLOPs and others can be added once we derive them in later phases.
    ];
  }, [overview, paperReference]);

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

  return (
    <div className="flex flex-col gap-4">
      <div className="section-card">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-xs font-semibold tracking-wide uppercase text-textMuted">
              Validation
            </div>
            <div className="text-[13px] text-textMuted mt-0.5">
              Compare our theoretical calculations to reported values from the paper.
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
                  Paper&apos;s Value
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
                      {row.ourValue == null
                        ? "N/A"
                        : `${row.ourValue.toFixed(2)} ${row.unit}`}
                    </td>
                    <td className="py-1.5 pr-3 text-textMuted">
                      {row.paperValue == null
                        ? "N/A"
                        : `${row.paperValue.toFixed(2)} ${row.unit}`}
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
              Exact FFN structure (SwiGLU vs ReLU) slightly changes both parameter counts and
              FLOP estimates.
            </li>
            <li>
              Later phases will add per-token FLOPs and memory validation once those tabs are
              implemented.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

