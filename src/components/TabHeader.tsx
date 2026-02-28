import React from "react";

export type TabId =
  | "overview"
  | "training-compute"
  | "training-memory"
  | "inference"
  | "parallelism"
  | "hw-efficiency"
  | "cost"
  | "routing"
  | "validation";

const TAB_LABELS: { id: TabId; label: string; icon: string }[] = [
  { id: "overview", label: "Overview", icon: "cube" },
  { id: "training-compute", label: "Training Compute", icon: "bolt" },
  { id: "training-memory", label: "Training Memory", icon: "database" },
  { id: "inference", label: "Inference", icon: "play" },
  { id: "parallelism", label: "Parallelism", icon: "arrows" },
  { id: "hw-efficiency", label: "HW Efficiency", icon: "gauge" },
  { id: "cost", label: "Cost", icon: "dollar" },
  { id: "routing", label: "Routing", icon: "shuffle" },
  { id: "validation", label: "Validation", icon: "check" }
];

interface Props {
  active: TabId;
  onChange: (id: TabId) => void;
}

const TabIcon: React.FC<{ name: string; className?: string }> = ({
  name,
  className = ""
}) => {
  const cn = `w-3.5 h-3.5 ${className}`;
  switch (name) {
    case "cube":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path d="M10 1L2 5.5v9L10 19l8-4.5v-9L10 1zm0 2.236L15.528 6 10 8.764 4.472 6 10 3.236z" />
        </svg>
      );
    case "bolt":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path d="M11.983 1.907a.75.75 0 00-1.292-.657l-8.5 9.5A.75.75 0 002.75 12h6.572l-1.305 6.093a.75.75 0 001.292.657l8.5-9.5A.75.75 0 0017.25 8h-6.572l1.305-6.093z" />
        </svg>
      );
    case "database":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 1c-3.866 0-7 1.79-7 4s3.134 4 7 4 7-1.79 7-4-3.134-4-7-4zM3 9.45V11c0 2.21 3.134 4 7 4s7-1.79 7-4V9.45C15.33 10.96 12.836 12 10 12s-5.33-1.04-7-2.55zM3 14.45V16c0 2.21 3.134 4 7 4s7-1.79 7-4v-1.55C15.33 15.96 12.836 17 10 17s-5.33-1.04-7-2.55z"
            clipRule="evenodd"
          />
        </svg>
      );
    case "play":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
        </svg>
      );
    case "arrows":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M2 10a.75.75 0 01.75-.75h12.59l-2.1-1.95a.75.75 0 111.02-1.1l3.5 3.25a.75.75 0 010 1.1l-3.5 3.25a.75.75 0 11-1.02-1.1l2.1-1.95H2.75A.75.75 0 012 10z"
            clipRule="evenodd"
          />
        </svg>
      );
    case "gauge":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-11.25a.75.75 0 00-1.5 0v2.5h-2.5a.75.75 0 000 1.5h2.5v2.5a.75.75 0 001.5 0v-2.5h2.5a.75.75 0 000-1.5h-2.5v-2.5z"
            clipRule="evenodd"
          />
        </svg>
      );
    case "dollar":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm.75 4.032V5.75a.75.75 0 00-1.5 0v.282a2.25 2.25 0 00-.972 3.89l1.525 1.17A.75.75 0 019.342 12H8.75a.75.75 0 000 1.5h.5v.75a.75.75 0 001.5 0v-.282a2.25 2.25 0 00.972-3.89l-1.525-1.17A.75.75 0 0110.658 8h.592a.75.75 0 000-1.5h-.5v-.468z" />
        </svg>
      );
    case "shuffle":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M13.47 3.22a.75.75 0 011.06 0l3.25 3.25a.75.75 0 010 1.06l-3.25 3.25a.75.75 0 01-1.06-1.06l1.97-1.97H11a3.5 3.5 0 00-3.5 3.5v.75a.75.75 0 01-1.5 0v-.75a5 5 0 015-5h4.44l-1.97-1.97a.75.75 0 010-1.06z"
            clipRule="evenodd"
          />
        </svg>
      );
    case "check":
      return (
        <svg className={cn} viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z"
            clipRule="evenodd"
          />
        </svg>
      );
    default:
      return null;
  }
};

export const TabHeader: React.FC<Props> = ({ active, onChange }) => {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {TAB_LABELS.map((tab) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            className={`tab-button flex items-center gap-1.5 ${isActive ? "tab-button-active" : ""}`}
            onClick={() => onChange(tab.id)}
          >
            <TabIcon name={tab.icon} />
            {tab.label}
          </button>
        );
      })}
    </div>
  );
};
