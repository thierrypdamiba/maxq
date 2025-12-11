"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { fetchExperiments, runEval, createExperiment, fetchEmbeddingModels, fetchProject } from "@/lib/api";
import { Sliders, CheckCircle, AlertTriangle, Clock, ArrowUp, ArrowDown, Play } from "lucide-react";

export default function TuningPage() {
    const params = useParams();
    const projectId = params.id as string;

    const [experiments, setExperiments] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    // Modal State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [newExpName, setNewExpName] = useState("");
    const [selectedModels, setSelectedModels] = useState<string[]>([]);
    const [selectedStrategies, setSelectedStrategies] = useState<string[]>(["hybrid"]);
    const [availableModels, setAvailableModels] = useState<any[]>([]);

    // Project/Collection config
    const [projectEmbeddingModel, setProjectEmbeddingModel] = useState<string | null>(null);

    useEffect(() => {
        loadExperiments();
        loadModels();
        loadProjectConfig();
    }, []);

    async function loadProjectConfig() {
        try {
            const project = await fetchProject(projectId);
            console.log("Project config loaded:", project);
            console.log("Embedding model:", project.embedding_model);
            setProjectEmbeddingModel(project.embedding_model);
        } catch (error) {
            console.error("Failed to load project config:", error);
        }
    }

    // Poll for running experiments
    useEffect(() => {
        const hasRunning = experiments.some(exp => exp.status === "running");
        if (!hasRunning) return;

        const interval = setInterval(() => {
            loadExperiments();
        }, 2000); // Poll every 2 seconds

        return () => clearInterval(interval);
    }, [experiments]);

    async function loadExperiments() {
        try {
            const data = await fetchExperiments(projectId);
            // Sort by most recent first
            const sorted = data.sort((a: any, b: any) =>
                new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
            );
            setExperiments(sorted);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }

    async function loadModels() {
        try {
            const models = await fetchEmbeddingModels();
            setAvailableModels(models);
            if (models.length > 0) setSelectedModels([models[0].model]);
        } catch (error) {
            console.error("Failed to load models", error);
        }
    }

    async function handleRunEval(experimentId: string) {
        try {
            await runEval(experimentId);
            // Reload to show updated status/metrics
            loadExperiments();
        } catch (error) {
            console.error("Eval failed:", error);
            alert("Failed to start evaluation");
        }
    }

    async function handleCreateExperiment() {
        console.log("handleCreateExperiment called");
        console.log("newExpName:", newExpName);
        console.log("selectedModels:", selectedModels);
        console.log("selectedStrategies:", selectedStrategies);

        if (!newExpName || selectedModels.length === 0 || selectedStrategies.length === 0) {
            console.log("Validation failed - missing required fields");
            alert("Please fill in experiment name and select at least one model and strategy");
            return;
        }

        try {
            console.log("Creating experiments...");
            // Create one experiment per model × strategy combination
            for (const model of selectedModels) {
                for (const strategy of selectedStrategies) {
                    let expName = newExpName;

                    // Add suffixes if multiple selections
                    const needsSuffix = selectedModels.length > 1 || selectedStrategies.length > 1;
                    if (needsSuffix) {
                        const modelSuffix = model.split('/').pop()?.replace(/[.-]/g, '_');
                        const strategySuffix = strategy.replace(/_/g, '');
                        expName = `${newExpName}_${modelSuffix}_${strategySuffix}`;
                    }

                    console.log(`Creating experiment: ${expName} with model ${model} and strategy ${strategy}`);
                    await createExperiment(projectId, expName, model, strategy);
                    console.log(`Experiment created: ${expName}`);
                }
            }
            console.log("All experiments created successfully");
            setIsModalOpen(false);
            setNewExpName("");
            setSelectedModels(availableModels.length > 0 ? [availableModels[0].model] : []);
            setSelectedStrategies(["hybrid"]);
            await loadExperiments();
        } catch (error) {
            console.error("Failed to create experiment:", error);
            alert(`Failed to create experiment: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    return (
        <div className="p-10 max-w-7xl mx-auto">
            <div className="flex justify-between items-center mb-10">
                <div>
                    <h1 className="text-2xl font-bold text-foreground tracking-tight">Tuning Experiments</h1>
                    <p className="text-foreground-muted mt-1 text-sm">Compare configurations and promote the best models.</p>
                </div>
                <button
                    onClick={() => setIsModalOpen(true)}
                    className="btn-miami px-4 py-2 text-sm"
                >
                    New Experiment
                </button>
            </div>

            <div className="space-y-6">
                {loading ? (
                    <div className="text-foreground-muted text-sm animate-pulse">Loading experiments...</div>
                ) : (
                    experiments.map((exp) => (
                        <div key={exp.id} className="glass-panel chrome-border rounded-lg overflow-hidden group">
                            <div className="px-6 py-4 border-b border-white/[0.06] flex justify-between items-center bg-white/[0.02]">
                                <div className="flex items-center">
                                    <Sliders className="w-4 h-4 text-foreground-muted mr-3" />
                                    <h3 className="text-base font-semibold text-foreground">{exp.name}</h3>
                                </div>
                                <div className="flex items-center space-x-3">
                                    <span className="text-xs text-foreground-muted">{new Date(exp.created_at).toLocaleDateString()}</span>
                                    {exp.status === "completed" ? (
                                        <span className="bg-success/10 text-success text-[10px] px-2 py-0.5 rounded-full font-medium flex items-center border border-success/10">
                                            <CheckCircle className="w-3 h-3 mr-1" /> Completed
                                        </span>
                                    ) : exp.status === "running" ? (
                                        <span className="bg-neon-cyan/10 text-neon-cyan text-[10px] px-2 py-0.5 rounded-full font-medium flex items-center border border-neon-cyan/10 animate-pulse">
                                            <Clock className="w-3 h-3 mr-1" /> Running
                                        </span>
                                    ) : (
                                        <span className="bg-white/10 text-foreground-muted text-[10px] px-2 py-0.5 rounded-full font-medium flex items-center border border-white/10">
                                            Pending
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="p-6">
                                {exp.status === "completed" ? (
                                    <div>
                                        <div className="flex items-center mb-6">
                                            <div className="bg-success/5 border border-success/10 rounded-md px-4 py-3 flex-1 flex items-center">
                                                <CheckCircle className="w-4 h-4 text-success mr-3" />
                                                <div>
                                                    <p className="text-sm font-medium text-success">Mission Verdict: Passed</p>
                                                    <p className="text-xs text-success/80 mt-0.5">Winner: {exp.winner || "N/A"}</p>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => handleRunEval(exp.id)}
                                                className="ml-4 px-4 py-2 border border-white/10 rounded-md text-sm font-medium text-foreground-muted hover:text-foreground hover:bg-white/[0.03] transition-all flex items-center"
                                            >
                                                <Play className="w-3 h-3 mr-2" /> Re-run Ragas
                                            </button>
                                            <button className="ml-2 px-4 py-2 bg-foreground text-background rounded-md text-sm font-medium hover:bg-foreground/90 transition-colors">
                                                Promote Winner
                                            </button>
                                        </div>

                                        <div className="grid grid-cols-2 gap-8">
                                            <div>
                                                <h4 className="text-[10px] text-foreground-muted font-medium mb-3 uppercase tracking-wide">QUALITY (nDCG@10)</h4>
                                                <div className="flex items-end">
                                                    <span className="text-2xl font-bold text-foreground tracking-tight">{exp.metrics?.ndcg?.candidate || "0.0"}</span>
                                                    <span className="text-sm text-foreground-muted ml-2 mb-1">vs {exp.metrics?.ndcg?.baseline || "0.0"}</span>
                                                    <span className="ml-4 flex items-center text-success font-medium bg-success/5 px-1.5 py-0.5 rounded border border-success/10 text-xs">
                                                        <ArrowUp className="w-3 h-3 mr-1" /> {exp.metrics?.ndcg?.delta || "0%"}
                                                    </span>
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-[10px] text-foreground-muted font-medium mb-3 uppercase tracking-wide">LATENCY (p95)</h4>
                                                <div className="flex items-end">
                                                    <span className="text-2xl font-bold text-foreground tracking-tight">{exp.metrics?.latency?.candidate || "0ms"}</span>
                                                    <span className="text-sm text-foreground-muted ml-2 mb-1">vs {exp.metrics?.latency?.baseline || "0ms"}</span>
                                                    <span className="ml-4 flex items-center text-danger font-medium bg-danger/5 px-1.5 py-0.5 rounded border border-danger/10 text-xs">
                                                        <ArrowUp className="w-3 h-3 mr-1" /> {exp.metrics?.latency?.delta || "0%"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="py-6 px-6">
                                        {/* Progress Bar */}
                                        <div className="mb-4">
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="text-xs text-foreground-muted font-medium">
                                                    {exp.progress_message || "Running..."}
                                                </span>
                                                <span className="text-xs text-foreground-muted">
                                                    {exp.progress_current || 0} / {exp.progress_total || 0}
                                                </span>
                                            </div>
                                            <div className="w-full bg-black/30 rounded-full h-2 overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-neon-cyan to-neon-magenta rounded-full transition-all duration-500 ease-out"
                                                    style={{
                                                        width: exp.progress_total > 0
                                                            ? `${(exp.progress_current / exp.progress_total) * 100}%`
                                                            : '0%'
                                                    }}
                                                />
                                            </div>
                                        </div>

                                        {/* Time Info */}
                                        <div className="flex justify-between items-center text-xs text-foreground-muted">
                                            <div>
                                                <Clock className="w-3 h-3 inline mr-1" />
                                                Elapsed: {exp.started_at
                                                    ? Math.floor((Date.now() - new Date(exp.started_at).getTime()) / 1000) + 's'
                                                    : '0s'}
                                            </div>
                                            <div>
                                                ETA: {(() => {
                                                    if (!exp.started_at || !exp.progress_current || exp.progress_current === 0) return 'Calculating...';
                                                    const elapsed = Date.now() - new Date(exp.started_at).getTime();
                                                    const avgTimePerStep = elapsed / exp.progress_current;
                                                    const remaining = (exp.progress_total - exp.progress_current) * avgTimePerStep;
                                                    return Math.ceil(remaining / 1000) + 's';
                                                })()}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* New Experiment Modal */}
            {isModalOpen && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                    <div className="glass-panel rounded-lg p-6 w-full max-w-md shadow-2xl">
                        <h2 className="text-lg font-bold mb-4 text-foreground">New Experiment</h2>

                        <div className="mb-4">
                            <label className="block text-xs font-medium text-foreground-muted mb-1">Experiment Name</label>
                            <input
                                type="text"
                                placeholder="e.g., BGE-Large vs Base"
                                className="w-full bg-black/20 border border-white/10 rounded-md px-3 py-2 text-foreground text-sm miami-focus placeholder-foreground-muted/30"
                                value={newExpName}
                                onChange={(e) => setNewExpName(e.target.value)}
                                autoFocus
                            />
                        </div>

                        <div className="mb-4">
                            <label className="block text-xs font-medium text-foreground-muted mb-2">Search Strategies ({selectedStrategies.length} selected)</label>
                            <div className="bg-black/20 border border-white/10 rounded-md p-2 space-y-1">
                                {[
                                    { value: "hybrid", label: "Hybrid (Dense + Sparse)" },
                                    { value: "dense", label: "Dense Only" },
                                    { value: "sparse", label: "Sparse Only" },
                                    { value: "speculative_rag", label: "Speculative RAG (HyDE)" }
                                ].map((strategy) => (
                                    <label
                                        key={strategy.value}
                                        className="flex items-center px-2 py-1.5 hover:bg-white/5 rounded cursor-pointer transition-colors"
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedStrategies.includes(strategy.value)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setSelectedStrategies([...selectedStrategies, strategy.value]);
                                                } else {
                                                    setSelectedStrategies(selectedStrategies.filter(s => s !== strategy.value));
                                                }
                                            }}
                                            className="w-4 h-4 rounded border-white/20 bg-black/30 text-neon-magenta focus:ring-neon-magenta focus:ring-offset-0"
                                        />
                                        <span className="ml-2 text-sm text-foreground">{strategy.label}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="mb-6">
                            <label className="block text-xs font-medium text-foreground-muted mb-2">
                                Embedding Models ({selectedModels.length} selected)
                                {projectEmbeddingModel && (
                                    <span className="ml-2 text-neon-cyan">• Collection uses: {projectEmbeddingModel}</span>
                                )}
                            </label>
                            <div className="max-h-48 overflow-y-auto bg-black/20 border border-white/10 rounded-md p-2 space-y-1">
                                {availableModels.map((model) => {
                                    const isCompatible = !projectEmbeddingModel || model.model === projectEmbeddingModel;
                                    console.log(`Model: ${model.model}, Compatible: ${isCompatible}, ProjectModel: ${projectEmbeddingModel}`);
                                    return (
                                        <label
                                            key={model.model}
                                            className={`flex items-center px-2 py-1.5 rounded transition-colors ${
                                                isCompatible
                                                    ? "hover:bg-white/5 cursor-pointer"
                                                    : "opacity-40 cursor-not-allowed"
                                            }`}
                                            title={
                                                !isCompatible
                                                    ? `This model is incompatible with the collection (uses ${projectEmbeddingModel})`
                                                    : ""
                                            }
                                        >
                                            <input
                                                type="checkbox"
                                                checked={selectedModels.includes(model.model)}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        setSelectedModels([...selectedModels, model.model]);
                                                    } else {
                                                        setSelectedModels(selectedModels.filter(m => m !== model.model));
                                                    }
                                                }}
                                                disabled={!isCompatible}
                                                className="w-4 h-4 rounded border-white/20 bg-black/30 text-neon-cyan focus:ring-neon-cyan focus:ring-offset-0 disabled:opacity-50 disabled:cursor-not-allowed"
                                            />
                                            <span className={`ml-2 text-sm ${isCompatible ? "text-foreground" : "text-foreground-muted line-through"}`}>
                                                {model.model}
                                                {!isCompatible && (
                                                    <span className="ml-2 text-xs text-red-400">(Incompatible)</span>
                                                )}
                                            </span>
                                        </label>
                                    );
                                })}
                            </div>
                        </div>

                        <div className="flex justify-end space-x-3">
                            <button
                                onClick={() => setIsModalOpen(false)}
                                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleCreateExperiment}
                                className="btn-miami px-4 py-2 text-sm"
                                disabled={selectedModels.length === 0 || selectedStrategies.length === 0}
                            >
                                {selectedModels.length * selectedStrategies.length > 1
                                    ? `Create ${selectedModels.length * selectedStrategies.length} Experiments`
                                    : 'Start Experiment'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
