import React from "react";

export type TabId =
  | "overview"
  | "training-compute"
  | "training-memory"
  | "inference"
  | "parallelism"
  | "routing"
  | "validation";

const TAB_LABELS: { id: TabId; label: string }[] = [
  { id: "overview", label: "Model Overview" },
  { id: "training-compute", label: "Training Compute" },
  { id: "training-memory", label: "Training Memory" },
  { id: "inference", label: "Inference Analysis" },
  { id: "parallelism", label: "Parallelism & Comm" },
  { id: "routing", label: "Routing & Efficiency" },
  { id: "validation", label: "Validation" }
];

interface Props {
  active: TabId;
  onChange: (id: TabId) => void;
}

export const TabHeader: React.FC<Props> = ({ active, onChange }) => {
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-borderSoft/60 pb-2">
      {TAB_LABELS.map((tab) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            className={`tab-button ${isActive ? "tab-button-active" : ""}`}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
};

