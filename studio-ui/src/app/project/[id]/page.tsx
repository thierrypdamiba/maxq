"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { fetchProject } from "@/lib/api";
import { Activity, Database, Zap, BarChart3 } from "lucide-react";

export default function ProjectOverview() {
    const params = useParams();
    const [project, setProject] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (params.id) {
            loadProject(params.id as string);
        }
    }, [params.id]);

    async function loadProject(id: string) {
        try {
            const data = await fetchProject(id);
            setProject(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }

    if (loading) return <div className="p-10 text-foreground-muted text-sm animate-pulse">Loading telemetry...</div>;
    if (!project) return <div className="p-10 text-foreground-muted text-sm">Project not found</div>;

    return (
        <div className="p-10 max-w-7xl mx-auto">
            <div className="flex justify-between items-center mb-10">
                <div>
                    <h1 className="text-2xl font-bold text-foreground tracking-tight">{project.name}</h1>
                    <p className="text-foreground-muted mt-1 text-sm">Mission Control</p>
                </div>
                <div className="flex space-x-3">
                    <button className="px-4 py-2 bg-white/[0.03] border border-white/[0.06] text-foreground rounded-md text-sm font-medium hover:bg-white/[0.06] transition-colors">
                        Edit Config
                    </button>
                    <button className="btn-miami px-4 py-2 text-sm">
                        New Run
                    </button>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
                {[
                    { label: "Total Queries", value: "1.2M", change: "+12%", icon: Activity },
                    { label: "Avg Latency", value: "145ms", change: "-5%", icon: Zap },
                    { label: "Index Size", value: "4.5GB", change: "+2%", icon: Database },
                    { label: "Quality (nDCG)", value: "0.89", change: "+0.02", icon: BarChart3 },
                ].map((stat) => (
                    <div key={stat.label} className="glass-panel chrome-border rounded-lg p-6 relative overflow-hidden group">
                        <div className="flex justify-between items-start mb-4">
                            <div className="text-foreground-muted group-hover:text-neon-cyan transition-colors">
                                <stat.icon className="w-5 h-5" />
                            </div>
                            <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${stat.change.startsWith("+") ? "bg-success/10 text-success border-success/10" : "bg-success/10 text-success border-success/10"
                                }`}>
                                {stat.change}
                            </span>
                        </div>
                        <h3 className="text-2xl font-bold text-foreground mb-1 tracking-tight">{stat.value}</h3>
                        <p className="text-xs text-foreground-muted font-medium uppercase tracking-wide opacity-80">{stat.label}</p>
                    </div>
                ))}
            </div>

            {/* Activity Feed */}
            <div className="glass-panel chrome-border rounded-lg p-6">
                <h2 className="text-base font-semibold text-foreground mb-6 flex items-center">
                    <Activity className="w-4 h-4 mr-2 text-foreground-muted" />
                    Recent Activity
                </h2>
                <div className="space-y-6">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="flex items-start pb-6 border-b border-white/[0.06] last:border-0 last:pb-0 group">
                            <div className="h-8 w-8 rounded-full bg-white/[0.03] flex items-center justify-center mt-0.5 border border-white/[0.06] group-hover:border-neon-cyan/30 transition-colors">
                                <Activity className="w-4 h-4 text-foreground-muted group-hover:text-neon-cyan transition-colors" />
                            </div>
                            <div className="ml-4">
                                <p className="text-sm font-medium text-foreground">
                                    New index build completed
                                </p>
                                <p className="text-xs text-foreground-muted mt-1">
                                    Ingested 50k documents from <span className="text-foreground">legal-corpus-v2</span>
                                </p>
                                <p className="text-[10px] text-foreground-muted/60 mt-2 uppercase tracking-wide">2 hours ago</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
