"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import {
    Code,
    Download,
    Share2,
    FileJson,
    Database,
    FileText,
    Copy,
    Check,
    AlertCircle,
    Loader2
} from "lucide-react";
import {
    fetchProject,
    fetchIndexedModels,
    exportProjectConfig,
    createSnapshot,
    getReactComponentUrl,
    IndexedModel
} from "@/lib/api";

interface Project {
    id: string;
    name: string;
    description?: string;
}

export default function ExportPage() {
    const params = useParams();
    const projectId = params.id as string;

    const [project, setProject] = useState<Project | null>(null);
    const [indexedModels, setIndexedModels] = useState<IndexedModel[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Export states
    const [exportingConfig, setExportingConfig] = useState(false);
    const [exportingSnapshot, setExportingSnapshot] = useState(false);
    const [snapshotResult, setSnapshotResult] = useState<any>(null);
    const [copied, setCopied] = useState<string | null>(null);

    useEffect(() => {
        loadData();
    }, [projectId]);

    async function loadData() {
        setLoading(true);
        try {
            const [projectData, modelsData] = await Promise.all([
                fetchProject(projectId),
                fetchIndexedModels(projectId)
            ]);
            setProject(projectData);
            setIndexedModels(modelsData.indexed_models);
            setError(null);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    async function handleExportConfig() {
        setExportingConfig(true);
        try {
            const config = await exportProjectConfig(projectId);
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `maxq-config-${project?.name?.replace(/\s+/g, "-") || projectId}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setExportingConfig(false);
        }
    }

    async function handleCreateSnapshot() {
        setExportingSnapshot(true);
        setSnapshotResult(null);
        try {
            const result = await createSnapshot(projectId);
            setSnapshotResult(result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setExportingSnapshot(false);
        }
    }

    function handleDownloadReact() {
        window.open(getReactComponentUrl(projectId), "_blank");
    }

    function copyToClipboard(text: string, key: string) {
        navigator.clipboard.writeText(text);
        setCopied(key);
        setTimeout(() => setCopied(null), 2000);
    }

    const curlCommand = `curl -X POST http://localhost:8888/playground/search \\
  -H "Content-Type: application/json" \\
  -d '{"project_id": "${projectId}", "query": "your search query", "strategy": "hybrid"}'`;

    if (loading) {
        return (
            <div className="flex h-full items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin w-8 h-8 border-2 border-neon-cyan border-t-transparent rounded-full mx-auto mb-4"></div>
                    <p className="text-foreground-muted">Loading export options...</p>
                </div>
            </div>
        );
    }

    if (indexedModels.length === 0) {
        return (
            <div className="flex h-full items-center justify-center">
                <div className="text-center max-w-md">
                    <div className="w-16 h-16 rounded-full bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-4">
                        <AlertCircle className="w-8 h-8 text-amber-500" />
                    </div>
                    <h2 className="text-lg font-semibold text-foreground mb-2">No Data to Export</h2>
                    <p className="text-foreground-muted text-sm">
                        Index some data first before exporting. Head to the Indexing page to get started.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-8 max-w-5xl mx-auto">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-foreground mb-2">Export & Deploy</h1>
                <p className="text-foreground-muted">
                    Export your search configuration for use in production.
                </p>
            </div>

            {error && (
                <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                    <p className="text-red-400 text-sm">{error}</p>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* React Component */}
                <div className="glass-panel chrome-border rounded-lg p-6 hover:bg-white/[0.02] transition-colors">
                    <div className="flex items-center mb-4">
                        <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mr-3">
                            <Share2 className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <h3 className="text-base font-semibold text-foreground">React Component</h3>
                            <p className="text-xs text-foreground-muted">Embeddable search widget</p>
                        </div>
                    </div>
                    <p className="text-sm text-foreground-muted mb-4">
                        Download a ready-to-use React component with search input, results display, and styling.
                    </p>
                    <button
                        onClick={handleDownloadReact}
                        className="btn-miami px-4 py-2 text-sm w-full flex items-center justify-center gap-2"
                    >
                        <Download className="w-4 h-4" />
                        Download MaxQSearch.jsx
                    </button>
                </div>

                {/* Project Config */}
                <div className="glass-panel chrome-border rounded-lg p-6 hover:bg-white/[0.02] transition-colors">
                    <div className="flex items-center mb-4">
                        <div className="w-10 h-10 rounded-lg bg-green-500/10 border border-green-500/20 flex items-center justify-center mr-3">
                            <FileJson className="w-5 h-5 text-green-400" />
                        </div>
                        <div>
                            <h3 className="text-base font-semibold text-foreground">Project Config</h3>
                            <p className="text-xs text-foreground-muted">JSON configuration</p>
                        </div>
                    </div>
                    <p className="text-sm text-foreground-muted mb-4">
                        Export project settings, indexed models, and configuration as JSON for backup or replication.
                    </p>
                    <button
                        onClick={handleExportConfig}
                        disabled={exportingConfig}
                        className="btn-miami px-4 py-2 text-sm w-full flex items-center justify-center gap-2"
                    >
                        {exportingConfig ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Download className="w-4 h-4" />
                        )}
                        {exportingConfig ? "Exporting..." : "Download Config"}
                    </button>
                </div>

                {/* Qdrant Snapshot */}
                <div className="glass-panel chrome-border rounded-lg p-6 hover:bg-white/[0.02] transition-colors">
                    <div className="flex items-center mb-4">
                        <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mr-3">
                            <Database className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <h3 className="text-base font-semibold text-foreground">Qdrant Snapshot</h3>
                            <p className="text-xs text-foreground-muted">Vector database backup</p>
                        </div>
                    </div>
                    <p className="text-sm text-foreground-muted mb-4">
                        Create a snapshot of your indexed vectors for migration or backup purposes.
                    </p>
                    <button
                        onClick={handleCreateSnapshot}
                        disabled={exportingSnapshot}
                        className="btn-miami px-4 py-2 text-sm w-full flex items-center justify-center gap-2"
                    >
                        {exportingSnapshot ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Database className="w-4 h-4" />
                        )}
                        {exportingSnapshot ? "Creating..." : "Create Snapshot"}
                    </button>
                    {snapshotResult && (
                        <div className="mt-4 p-3 rounded bg-green-500/10 border border-green-500/20">
                            <p className="text-green-400 text-xs font-medium mb-1">Snapshot Created</p>
                            <p className="text-foreground-muted text-xs font-mono">{snapshotResult.snapshot_name}</p>
                        </div>
                    )}
                </div>

                {/* API Endpoint */}
                <div className="glass-panel chrome-border rounded-lg p-6 hover:bg-white/[0.02] transition-colors">
                    <div className="flex items-center mb-4">
                        <div className="w-10 h-10 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-center justify-center mr-3">
                            <Code className="w-5 h-5 text-orange-400" />
                        </div>
                        <div>
                            <h3 className="text-base font-semibold text-foreground">API Endpoint</h3>
                            <p className="text-xs text-foreground-muted">REST API access</p>
                        </div>
                    </div>
                    <p className="text-sm text-foreground-muted mb-4">
                        Use the MaxQ API directly from your backend or CLI.
                    </p>
                    <div className="bg-black/30 rounded p-3 font-mono text-xs text-foreground-muted overflow-x-auto">
                        <code>POST /playground/search</code>
                    </div>
                </div>
            </div>

            {/* cURL Command */}
            <div className="glass-panel chrome-border rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-base font-semibold text-foreground">Quick Start: cURL</h3>
                    <button
                        onClick={() => copyToClipboard(curlCommand, "curl")}
                        className="text-xs text-foreground-muted hover:text-foreground flex items-center gap-1"
                    >
                        {copied === "curl" ? (
                            <>
                                <Check className="w-3 h-3 text-green-400" />
                                Copied!
                            </>
                        ) : (
                            <>
                                <Copy className="w-3 h-3" />
                                Copy
                            </>
                        )}
                    </button>
                </div>
                <pre className="bg-black/30 rounded p-4 font-mono text-xs text-foreground-muted overflow-x-auto whitespace-pre-wrap">
                    {curlCommand}
                </pre>
            </div>

            {/* Indexed Models Info */}
            <div className="mt-8">
                <h3 className="text-sm font-medium text-foreground-muted uppercase tracking-wide mb-4">
                    Available Collections
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {indexedModels.map((model) => (
                        <div
                            key={model.model_name}
                            className="glass-panel chrome-border rounded-lg p-4"
                        >
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm font-medium text-foreground">{model.model_name}</p>
                                    <p className="text-xs text-foreground-muted font-mono mt-1">{model.collection_name}</p>
                                </div>
                                <span className="text-xs bg-neon-cyan/10 text-neon-cyan px-2 py-1 rounded border border-neon-cyan/20">
                                    {model.point_count.toLocaleString()} pts
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
