"use client";

import { useState, useEffect } from "react";
import { getApiUrl } from "@/lib/api";

interface CollectionStats {
    name: string;
    points_count: number;
    vectors_count: number;
    segments_count: number;
    status: string;
    error?: string;
}

interface CleanupAction {
    action: string;
    target: string;
    reason: string;
    estimated_points: number;
}

interface CleanupReport {
    total_collections: number;
    total_points: number;
    collections: CollectionStats[];
    empty_collections: string[];
    stale_collections: string[];
    suggested_actions: CleanupAction[];
    reclaimable_points: number;
    llm_summary?: string;
}

export default function CleanupPage() {
    const [report, setReport] = useState<CleanupReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [executing, setExecuting] = useState(false);
    const [results, setResults] = useState<any[]>([]);
    const [error, setError] = useState("");

    async function analyze() {
        setLoading(true);
        setError("");
        setResults([]);
        try {
            const res = await fetch(`${getApiUrl()}/cleanup/analyze`);
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setReport(data);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Failed to analyze");
        }
        setLoading(false);
    }

    async function executeActions(dryRun: boolean) {
        if (!report?.suggested_actions.length) return;
        setExecuting(true);
        try {
            const res = await fetch(`${getApiUrl()}/cleanup/execute`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    actions: report.suggested_actions,
                    dry_run: dryRun,
                }),
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setResults(data.results || []);
            if (!dryRun) {
                // Re-analyze after execution
                await analyze();
            }
        } catch (e) {
            setError(e instanceof Error ? e.message : "Execution failed");
        }
        setExecuting(false);
    }

    useEffect(() => {
        analyze();
    }, []);

    return (
        <div className="p-6 max-w-6xl">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-foreground">Cluster Cleanup</h1>
                    <p className="text-foreground-muted text-sm mt-1">
                        Analyze and clean up your Qdrant cluster
                    </p>
                </div>
                <button
                    onClick={analyze}
                    disabled={loading}
                    className="bg-qdrant-red text-white px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 disabled:opacity-50"
                >
                    {loading ? "Analyzing..." : "Re-analyze"}
                </button>
            </div>

            {error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 text-red-400 text-sm">
                    {error}
                </div>
            )}

            {report && (
                <>
                    {/* Summary cards */}
                    <div className="grid grid-cols-4 gap-4 mb-6">
                        <div className="bg-sidebar-bg border border-border-light rounded-lg p-4">
                            <div className="text-2xl font-bold text-foreground">{report.total_collections}</div>
                            <div className="text-sm text-foreground-muted">Collections</div>
                        </div>
                        <div className="bg-sidebar-bg border border-border-light rounded-lg p-4">
                            <div className="text-2xl font-bold text-foreground">{report.total_points.toLocaleString()}</div>
                            <div className="text-sm text-foreground-muted">Total Points</div>
                        </div>
                        <div className="bg-sidebar-bg border border-border-light rounded-lg p-4">
                            <div className="text-2xl font-bold text-yellow-400">{report.empty_collections.length + report.stale_collections.length}</div>
                            <div className="text-sm text-foreground-muted">Issues Found</div>
                        </div>
                        <div className="bg-sidebar-bg border border-border-light rounded-lg p-4">
                            <div className="text-2xl font-bold text-red-400">{report.reclaimable_points.toLocaleString()}</div>
                            <div className="text-sm text-foreground-muted">Reclaimable Points</div>
                        </div>
                    </div>

                    {/* Collections table */}
                    <div className="bg-sidebar-bg border border-border-light rounded-lg overflow-hidden mb-6">
                        <div className="px-4 py-3 border-b border-border-light">
                            <h2 className="text-sm font-semibold text-foreground">Collections</h2>
                        </div>
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-border-light text-foreground-muted">
                                    <th className="text-left px-4 py-2 font-medium">Name</th>
                                    <th className="text-right px-4 py-2 font-medium">Points</th>
                                    <th className="text-right px-4 py-2 font-medium">Vectors</th>
                                    <th className="text-right px-4 py-2 font-medium">Segments</th>
                                    <th className="text-left px-4 py-2 font-medium">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {report.collections.map((col) => (
                                    <tr key={col.name} className="border-b border-border-light last:border-0">
                                        <td className="px-4 py-2 text-foreground font-mono text-xs">{col.name}</td>
                                        <td className="px-4 py-2 text-right text-foreground">{col.points_count.toLocaleString()}</td>
                                        <td className="px-4 py-2 text-right text-foreground">{col.vectors_count.toLocaleString()}</td>
                                        <td className="px-4 py-2 text-right text-foreground">{col.segments_count}</td>
                                        <td className="px-4 py-2">
                                            {col.error ? (
                                                <span className="text-red-400">{col.error}</span>
                                            ) : (
                                                <span className={col.status === "green" ? "text-green-400" : "text-yellow-400"}>
                                                    {col.status}
                                                </span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Suggested actions */}
                    {report.suggested_actions.length > 0 && (
                        <div className="bg-sidebar-bg border border-border-light rounded-lg overflow-hidden mb-6">
                            <div className="px-4 py-3 border-b border-border-light flex items-center justify-between">
                                <h2 className="text-sm font-semibold text-foreground">
                                    Suggested Actions ({report.suggested_actions.length})
                                </h2>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => executeActions(true)}
                                        disabled={executing}
                                        className="text-xs bg-sidebar-bg border border-border-light text-foreground px-3 py-1.5 rounded-md hover:bg-sidebar-hover-bg"
                                    >
                                        Dry Run
                                    </button>
                                    <button
                                        onClick={() => {
                                            if (confirm("Execute cleanup actions? This cannot be undone.")) {
                                                executeActions(false);
                                            }
                                        }}
                                        disabled={executing}
                                        className="text-xs bg-red-600 text-white px-3 py-1.5 rounded-md hover:bg-red-700"
                                    >
                                        {executing ? "Executing..." : "Execute"}
                                    </button>
                                </div>
                            </div>
                            <div className="divide-y divide-border-light">
                                {report.suggested_actions.map((action, i) => (
                                    <div key={i} className="px-4 py-3">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs font-mono bg-red-500/10 text-red-400 px-2 py-0.5 rounded">
                                                {action.action}
                                            </span>
                                            <span className="text-sm text-foreground font-mono">{action.target}</span>
                                        </div>
                                        <p className="text-xs text-foreground-muted mt-1">{action.reason}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Execution results */}
                    {results.length > 0 && (
                        <div className="bg-sidebar-bg border border-border-light rounded-lg overflow-hidden mb-6">
                            <div className="px-4 py-3 border-b border-border-light">
                                <h2 className="text-sm font-semibold text-foreground">Execution Results</h2>
                            </div>
                            <div className="divide-y divide-border-light">
                                {results.map((r, i) => (
                                    <div key={i} className="px-4 py-3 flex items-center gap-3">
                                        <span className={`text-xs font-medium ${r.status === "completed" ? "text-green-400" : r.status === "skipped" ? "text-yellow-400" : "text-red-400"}`}>
                                            {r.status}
                                        </span>
                                        <span className="text-sm text-foreground">{r.message}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* LLM Summary */}
                    {report.llm_summary && (
                        <div className="bg-sidebar-bg border border-qdrant-red/30 rounded-lg p-4">
                            <h2 className="text-sm font-semibold text-foreground mb-2">AI Analysis</h2>
                            <p className="text-sm text-foreground-muted whitespace-pre-wrap">{report.llm_summary}</p>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
