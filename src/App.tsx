import React, { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { TabHeader, TabId } from "./components/TabHeader";
import { TabModelOverview } from "./components/TabModelOverview";
import { TabTrainingCompute } from "./components/TabTrainingCompute";
import { TabTrainingMemory } from "./components/TabTrainingMemory";
import { TabInference } from "./components/TabInference";
import { TabParallelism } from "./components/TabParallelism";
import { TabRouting } from "./components/TabRouting";
import { TabValidation } from "./components/TabValidation";

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  return (
    <div className="flex h-screen bg-background text-white">
      <Sidebar />
      <main className="flex-1 h-screen overflow-hidden flex flex-col">
        <header className="px-5 pt-3 pb-2 border-b border-borderSoft/80 flex items-center justify-between bg-background/90 backdrop-blur">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-textMuted">
              MoE Theoretical Analysis Tool
            </div>
            <div className="text-sm font-semibold mt-0.5">MoE Analysis Dashboard</div>
          </div>
          <div className="text-[11px] text-textMuted">
            Change any value in the sidebar and all tabs update reactively.
          </div>
        </header>
        <section className="px-5 pt-3 flex-1 overflow-y-auto">
          <TabHeader active={activeTab} onChange={setActiveTab} />
          <div className="mt-3 mb-4">
            {activeTab === "overview" && <TabModelOverview />}
            {activeTab === "training-compute" && <TabTrainingCompute />}
            {activeTab === "training-memory" && <TabTrainingMemory />}
            {activeTab === "inference" && <TabInference />}
            {activeTab === "parallelism" && <TabParallelism />}
            {activeTab === "routing" && <TabRouting />}
            {activeTab === "validation" && <TabValidation />}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;

