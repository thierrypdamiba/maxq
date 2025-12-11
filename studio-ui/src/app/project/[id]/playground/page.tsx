"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { searchProject, fetchIndexedModels, IndexedModel } from "@/lib/api";
import { Search, Play, AlertCircle, Database, Cpu, Layers } from "lucide-react";
import Link from "next/link";

export default function PlaygroundPage() {
    const params = useParams();
    const projectId = params.id as string;

    const [query, setQuery] = useState("");
    const [results, setResults] = useState<any[]>([]);
    const [searching, setSearching] = useState(false);
    const [indexedModels, setIndexedModels] = useState<IndexedModel[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("");
    const [useRRF, setUseRRF] = useState(false);
    const [loadingModels, setLoadingModels] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [modelUsed, setModelUsed] = useState<string | null>(null);

    useEffect(() => {
        loadIndexedModels();
    }, [projectId]);

    async function loadIndexedModels() {
        setLoadingModels(true);
        try {
            const data = await fetchIndexedModels(projectId);
            setIndexedModels(data.indexed_models);
            if (data.indexed_models.length > 0) {
                setSelectedModel(data.indexed_models[0].model_name);
            }
            setError(null);
        } catch (err: any) {
            console.error(err);
            setError(err.message);
        } finally {
            setLoadingModels(false);
        }
    }

    async function handleSearch() {
        if (!query.trim()) return;

        setSearching(true);
        setError(null);
        try {
            // Use hybrid strategy (dense + sparse with RRF fusion)
            const data = await searchProject(projectId, query, "hybrid", selectedModel);
            setResults(data.results);
            setModelUsed(data.model_used);
        } catch (err: any) {
            console.error(err);
            setError(err.message);
            setResults([]);
        } finally {
            setSearching(false);
        }
    }

    // Show loading state
    if (loadingModels) {
        return (
            <div className="flex h-full items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin w-8 h-8 border-2 border-neon-cyan border-t-transparent rounded-full mx-auto mb-4"></div>
                    <p className="text-foreground-muted">Loading playground...</p>
                </div>
            </div>
        );
    }

    // Show error if no data indexed
    if (indexedModels.length === 0) {
        return (
            <div className="flex h-full items-center justify-center">
                <div className="text-center max-w-md">
                    <div className="w-16 h-16 rounded-full bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-4">
                        <AlertCircle className="w-8 h-8 text-amber-500" />
                    </div>
                    <h2 className="text-lg font-semibold text-foreground mb-2">No Data Indexed</h2>
                    <p className="text-foreground-muted text-sm mb-6">
                        You need to index some data before you can search. Head over to the Indexing page to get started.
                    </p>
                    <Link
                        href={`/project/${projectId}/indexing`}
                        className="btn-miami px-6 py-2 text-sm inline-flex items-center gap-2"
                    >
                        <Database className="w-4 h-4" />
                        Go to Indexing
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-full">
            {/* Left Panel: Query & Config */}
            <div className="w-1/3 border-r border-white/[0.06] bg-background p-6 flex flex-col">
                <h2 className="text-base font-semibold text-foreground mb-6 flex items-center">
                    <Play className="w-4 h-4 mr-2 text-foreground-muted" />
                    Query Playground
                </h2>

                <div className="space-y-6 flex-1">
                    {/* Search Query */}
                    <div>
                        <label className="block text-xs font-medium text-foreground-muted mb-2 uppercase tracking-wide">Search Query</label>
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="w-full h-32 bg-black/20 border border-white/10 rounded-md p-3 text-foreground text-sm miami-focus resize-none placeholder-foreground-muted/30"
                            placeholder="Enter your test query here..."
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSearch();
                                }
                            }}
                        />
                    </div>

                    {/* Model Selection */}
                    <div>
                        <label className="block text-xs font-medium text-foreground-muted mb-3 uppercase tracking-wide flex items-center">
                            <Cpu className="w-3 h-3 mr-1.5" /> Embedding Model
                        </label>
                        <div className="space-y-2">
                            {indexedModels.map((model) => (
                                <button
                                    key={model.model_name}
                                    onClick={() => {
                                        setSelectedModel(model.model_name);
                                        setUseRRF(false);
                                    }}
                                    className={`w-full text-left px-3 py-2.5 rounded-md border transition-all ${
                                        selectedModel === model.model_name && !useRRF
                                            ? "bg-white/[0.05] border-neon-cyan/30 text-foreground"
                                            : "bg-transparent border-white/[0.06] text-foreground-muted hover:text-foreground hover:bg-white/[0.02]"
                                    }`}
                                >
                                    <div className="text-sm font-medium">{model.model_name.split('/').pop()}</div>
                                    <div className="text-[10px] text-foreground-muted/60 mt-0.5">
                                        {model.point_count.toLocaleString()} vectors
                                    </div>
                                </button>
                            ))}

                            {/* RRF option - only show if multiple models */}
                            {indexedModels.length > 1 && (
                                <button
                                    onClick={() => setUseRRF(true)}
                                    className={`w-full text-left px-3 py-2.5 rounded-md border transition-all ${
                                        useRRF
                                            ? "bg-neon-magenta/10 border-neon-magenta/30 text-foreground"
                                            : "bg-transparent border-white/[0.06] text-foreground-muted hover:text-foreground hover:bg-white/[0.02]"
                                    }`}
                                >
                                    <div className="text-sm font-medium flex items-center gap-2">
                                        <Layers className="w-3.5 h-3.5" />
                                        RRF Fusion
                                    </div>
                                    <div className="text-[10px] text-foreground-muted/60 mt-0.5">
                                        Combine results from all {indexedModels.length} models
                                    </div>
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {/* Error Display */}
                {error && (
                    <div className="mb-4 p-3 rounded-md bg-red-500/10 border border-red-500/20">
                        <p className="text-red-400 text-xs">{error}</p>
                    </div>
                )}

                <div className="mt-6">
                    <button
                        onClick={handleSearch}
                        disabled={searching || !query.trim() || (!selectedModel && !useRRF)}
                        className={`w-full py-2.5 rounded-md font-medium text-sm flex items-center justify-center transition-all ${
                            searching || !query.trim() || (!selectedModel && !useRRF)
                                ? "bg-white/[0.05] text-foreground-muted cursor-not-allowed"
                                : "btn-miami"
                        }`}
                    >
                        {searching ? <span className="animate-pulse">Searching...</span> : <span>Run Search</span>}
                    </button>
                </div>
            </div>

            {/* Right Panel: Results */}
            <div className="flex-1 bg-background/50 p-8 overflow-y-auto">
                <div className="max-w-3xl mx-auto">
                    <div className="flex justify-between items-center mb-6">
                        <div>
                            <h3 className="text-base font-semibold text-foreground">Results</h3>
                            {modelUsed && results.length > 0 && (
                                <p className="text-[10px] text-foreground-muted mt-1">
                                    Using: <span className="text-neon-cyan">{modelUsed}</span>
                                </p>
                            )}
                        </div>
                        {results.length > 0 && (
                            <span className="text-[10px] font-medium text-neon-cyan bg-neon-cyan/5 px-2 py-0.5 rounded border border-neon-cyan/10">{results.length} HITS</span>
                        )}
                    </div>

                    {results.length === 0 ? (
                        <div className="text-center py-20 text-foreground-muted/30">
                            <Search className="w-10 h-10 mx-auto mb-4 opacity-50" />
                            <p className="text-sm">Run a query to see results</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {results.map((hit, i) => (
                                <div key={hit.id} className="glass-panel chrome-border rounded-lg p-5 transition-all hover:bg-white/[0.02] group">
                                    <div className="flex justify-between items-start mb-2">
                                        <span className="bg-white/[0.05] text-foreground-muted text-[10px] px-1.5 py-0.5 rounded border border-white/[0.05] group-hover:border-neon-cyan/20 group-hover:text-neon-cyan transition-colors">
                                            {hit.id}
                                        </span>
                                        <span className="text-xs font-mono font-medium text-foreground-muted group-hover:text-neon-magenta transition-colors">
                                            {hit.score.toFixed(4)}
                                        </span>
                                    </div>
                                    <p className="text-foreground mb-3 leading-relaxed text-sm font-light">
                                        {hit.text}
                                    </p>
                                    {hit.metadata && (
                                        <div className="bg-black/20 rounded p-2 text-[10px] text-foreground-muted font-mono border border-white/[0.05] overflow-x-auto">
                                            {JSON.stringify(hit.metadata, null, 2)}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
