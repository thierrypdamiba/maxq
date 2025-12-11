"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { indexDataset, fetchEmbeddingModels, fetchIndexStatus, IndexedModel } from "@/lib/api";
import { Database, Play, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

export default function IndexingPage() {
    const params = useParams();
    const projectId = params.id as string;

    // Form state
    const [datasetName, setDatasetName] = useState("");
    const [selectedModel, setSelectedModel] = useState("sentence-transformers/all-MiniLM-L6-v2");
    const [limit, setLimit] = useState(500);

    // UI state
    const [models, setModels] = useState<Array<{model: string; dim: number; description: string}>>([]);
    const [indexedModels, setIndexedModels] = useState<IndexedModel[]>([]);
    const [loading, setLoading] = useState(true);
    const [indexing, setIndexing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    useEffect(() => {
        loadData();
    }, [projectId]);

    async function loadData() {
        setLoading(true);
        try {
            const [modelsData, statusData] = await Promise.all([
                fetchEmbeddingModels(),
                fetchIndexStatus(projectId)
            ]);
            setModels(modelsData);
            setIndexedModels(statusData.indexed_models);
            setError(null);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    async function handleIndex() {
        if (!datasetName.trim()) {
            setError("Please enter a HuggingFace dataset name");
            return;
        }

        setIndexing(true);
        setError(null);
        setSuccess(null);

        try {
            const result = await indexDataset(projectId, datasetName, selectedModel, limit);
            setSuccess(`Successfully indexed ${result.points_indexed} documents!`);
            // Refresh indexed models list
            const statusData = await fetchIndexStatus(projectId);
            setIndexedModels(statusData.indexed_models);
            setDatasetName("");
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIndexing(false);
        }
    }

    if (loading) {
        return (
            <div className="flex h-full items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-neon-cyan" />
            </div>
        );
    }

    return (
        <div className="p-8 max-w-4xl mx-auto">
            <h1 className="text-2xl font-bold text-foreground mb-2">Index Data</h1>
            <p className="text-foreground-muted text-sm mb-8">
                Index a HuggingFace dataset into Qdrant for semantic search
            </p>

            {/* Indexed Collections */}
            {indexedModels.length > 0 && (
                <div className="mb-8">
                    <h2 className="text-sm font-medium text-foreground-muted mb-3 uppercase tracking-wide">
                        Indexed Collections
                    </h2>
                    <div className="space-y-2">
                        {indexedModels.map((model) => (
                            <div
                                key={model.collection_name}
                                className="glass-panel chrome-border rounded-lg p-4 flex items-center justify-between"
                            >
                                <div>
                                    <div className="text-sm font-medium text-foreground">
                                        {model.model_name.split('/').pop()}
                                    </div>
                                    <div className="text-xs text-foreground-muted mt-1">
                                        {model.collection_name}
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <span className={`text-xs px-2 py-1 rounded ${
                                        model.point_count > 0
                                            ? "bg-green-500/10 text-green-400 border border-green-500/20"
                                            : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                                    }`}>
                                        {model.point_count.toLocaleString()} vectors
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Index Form */}
            <div className="glass-panel chrome-border rounded-lg p-6">
                <h2 className="text-base font-semibold text-foreground mb-4 flex items-center">
                    <Database className="w-4 h-4 mr-2 text-neon-cyan" />
                    Index New Dataset
                </h2>

                <div className="space-y-4">
                    {/* Dataset Name */}
                    <div>
                        <label className="block text-xs font-medium text-foreground-muted mb-2 uppercase tracking-wide">
                            HuggingFace Dataset
                        </label>
                        <input
                            type="text"
                            value={datasetName}
                            onChange={(e) => setDatasetName(e.target.value)}
                            placeholder="e.g., squad, fka/awesome-chatgpt-prompts"
                            className="w-full bg-black/20 border border-white/10 rounded-md px-3 py-2 text-foreground text-sm miami-focus placeholder-foreground-muted/30"
                        />
                        <p className="text-xs text-foreground-muted mt-1">
                            Enter the full dataset name from huggingface.co/datasets
                        </p>
                    </div>

                    {/* Embedding Model */}
                    <div>
                        <label className="block text-xs font-medium text-foreground-muted mb-2 uppercase tracking-wide">
                            Embedding Model
                        </label>
                        <div className="space-y-2">
                            {models.map((model) => (
                                <button
                                    key={model.model}
                                    onClick={() => setSelectedModel(model.model)}
                                    className={`w-full text-left px-3 py-3 rounded-md border transition-all ${
                                        selectedModel === model.model
                                            ? "bg-neon-cyan/10 border-neon-cyan/30 text-foreground"
                                            : "bg-transparent border-white/[0.06] text-foreground-muted hover:text-foreground hover:bg-white/[0.02]"
                                    }`}
                                >
                                    <div className="text-sm font-medium">{model.model.split('/').pop()}</div>
                                    <div className="text-xs text-foreground-muted/60 mt-0.5">
                                        {model.description} ({model.dim} dimensions)
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Document Limit */}
                    <div>
                        <label className="block text-xs font-medium text-foreground-muted mb-2 uppercase tracking-wide">
                            Document Limit
                        </label>
                        <input
                            type="number"
                            value={limit}
                            onChange={(e) => setLimit(parseInt(e.target.value) || 500)}
                            min={10}
                            max={10000}
                            className="w-32 bg-black/20 border border-white/10 rounded-md px-3 py-2 text-foreground text-sm miami-focus"
                        />
                        <p className="text-xs text-foreground-muted mt-1">
                            Number of documents to index (keep small for testing)
                        </p>
                    </div>

                    {/* Error/Success Messages */}
                    {error && (
                        <div className="p-3 rounded-md bg-red-500/10 border border-red-500/20 flex items-start gap-2">
                            <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                            <p className="text-red-400 text-sm">{error}</p>
                        </div>
                    )}

                    {success && (
                        <div className="p-3 rounded-md bg-green-500/10 border border-green-500/20 flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                            <p className="text-green-400 text-sm">{success}</p>
                        </div>
                    )}

                    {/* Submit Button */}
                    <button
                        onClick={handleIndex}
                        disabled={indexing || !datasetName.trim()}
                        className={`w-full py-3 rounded-md font-medium text-sm flex items-center justify-center gap-2 transition-all ${
                            indexing || !datasetName.trim()
                                ? "bg-white/[0.05] text-foreground-muted cursor-not-allowed"
                                : "btn-miami"
                        }`}
                    >
                        {indexing ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Indexing... (this may take a minute)
                            </>
                        ) : (
                            <>
                                <Play className="w-4 h-4" />
                                Start Indexing
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Help Section */}
            <div className="mt-8 p-4 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                <h3 className="text-sm font-medium text-foreground mb-2">Popular Datasets</h3>
                <div className="flex flex-wrap gap-2">
                    {["squad", "fka/awesome-chatgpt-prompts", "ms_marco", "quora"].map((ds) => (
                        <button
                            key={ds}
                            onClick={() => setDatasetName(ds)}
                            className="text-xs px-2 py-1 rounded bg-white/[0.05] text-foreground-muted hover:text-foreground hover:bg-white/[0.08] transition-colors"
                        >
                            {ds}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
