"use client";

import { Zap, ExternalLink, Rocket, MoreVertical, Server, HardDrive, Cpu, Network } from "lucide-react";
import Link from "next/link";

export default function ClustersPage() {
  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Page Title */}
      <h1 className="text-3xl font-bold text-foreground mb-6">Clusters</h1>

      {/* Info Banner */}
      <div className="bg-qdrant-blue/20 border border-qdrant-blue/30 rounded-lg p-4 mb-6 flex items-start justify-between">
        <div className="flex items-start gap-3">
          <Zap className="w-5 h-5 text-qdrant-blue mt-0.5" />
          <div>
            <div className="font-semibold text-foreground mb-1">Live now: Qdrant Cloud Inference</div>
            <div className="text-sm text-foreground-muted">
              Directly create vector embeddings (dense or sparse) from text or images within Qdrant Cloud.{" "}
              <Link href="#" className="text-qdrant-blue hover:underline">
                Learn more
              </Link>
            </div>
          </div>
        </div>
        <button className="text-foreground-muted hover:text-foreground">
          ×
        </button>
      </div>

      {/* Clusters Table */}
      <div className="bg-surface rounded-lg border border-border-light overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-surface-highlight border-b border-border-light">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  CLUSTER
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  CONFIGURATION
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  PROVIDER
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  STATUS
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  ACTIONS
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-light">
              <tr className="hover:bg-surface-highlight/50 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex flex-col gap-2">
                    <span className="badge badge-purple">FREE TIER</span>
                    <div className="flex items-center gap-2">
                      <span className="text-foreground font-medium">testing</span>
                      <ExternalLink className="w-4 h-4 text-foreground-muted" />
                    </div>
                    <Link href="#" className="text-sm text-qdrant-blue hover:underline flex items-center gap-1">
                      <Rocket className="w-3 h-3" />
                      Upgrade to a Paid Cluster
                    </Link>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex flex-col gap-2 text-sm text-foreground-muted">
                    <div className="flex items-center gap-2">
                      <Server className="w-4 h-4" />
                      <span>1 NODE</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <HardDrive className="w-4 h-4" />
                      <span>Disk: 4GiB</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4" />
                      <span>RAM: 1GiB</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Network className="w-4 h-4" />
                      <span>vCPUs: 0.5</span>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 bg-white rounded flex items-center justify-center text-xs font-bold text-black">
                      G
                    </div>
                    <span className="text-sm text-foreground-muted">us-east4</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="badge badge-success">HEALTHY</span>
                </td>
                <td className="px-6 py-4">
                  <button className="text-foreground-muted hover:text-foreground">
                    <MoreVertical className="w-5 h-5" />
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
