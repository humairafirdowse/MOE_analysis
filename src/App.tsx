import React, { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { TabHeader, TabId } from "./components/TabHeader";
import { TabModelOverview } from "./components/TabModelOverview";
import { TabTrainingCompute } from "./components/TabTrainingCompute";
import { TabTrainingMemory } from "./components/TabTrainingMemory";
import { TabInference } from "./components/TabInference";
import { TabParallelism } from "./components/TabParallelism";
import { TabHardwareEfficiency } from "./components/TabHardwareEfficiency";
import { TabCostAnalysis } from "./components/TabCostAnalysis";
import { TabRouting } from "./components/TabRouting";
import { TabValidation } from "./components/TabValidation";

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  return (
    <div className="flex h-screen bg-background text-white">
      <Sidebar />
      <main className="flex-1 h-screen overflow-hidden flex flex-col">
        <header className="px-6 pt-4 pb-3 border-b border-borderSoft/40 flex items-center justify-between bg-background/95 backdrop-blur-sm">
          <div>
            <div className="text-lg font-bold tracking-tight">
              MoE Analysis Dashboard
            </div>
            <div className="text-[11px] text-textMuted/70 mt-0.5">
              Theoretical analysis of Mixture-of-Experts architectures
            </div>
          </div>
          <div className="text-[10px] text-textMuted/50 bg-surface/50 border border-borderSoft/30 rounded-lg px-3 py-1.5 font-medium">
            All tabs update reactively
          </div>
        </header>
        <section className="px-6 pt-4 flex-1 overflow-y-auto pb-8">
          <TabHeader active={activeTab} onChange={setActiveTab} />
          <div className="mt-4">
            {activeTab === "overview" && <TabModelOverview />}
            {activeTab === "training-compute" && <TabTrainingCompute />}
            {activeTab === "training-memory" && <TabTrainingMemory />}
            {activeTab === "inference" && <TabInference />}
            {activeTab === "parallelism" && <TabParallelism />}
            {activeTab === "hw-efficiency" && <TabHardwareEfficiency />}
            {activeTab === "cost" && <TabCostAnalysis />}
            {activeTab === "routing" && <TabRouting />}
            {activeTab === "validation" && <TabValidation />}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
