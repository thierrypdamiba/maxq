"use client";

import { Cloud, Server, Zap } from "lucide-react";
import Link from "next/link";

export default function HybridCloudPage() {
  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Main Heading */}
      <h1 className="text-3xl font-bold text-foreground mb-6">Hybrid Cloud</h1>

      {/* Description */}
      <p className="text-lg text-foreground mb-10 max-w-3xl">
        Seamlessly deploy and manage your vector databases across diverse environments, ensuring performance, security, and cost efficiency for AI-driven applications.
      </p>

      {/* Central Diagram Placeholder */}
      <div className="flex justify-center items-center my-16">
        <div className="relative w-96 h-96">
          {/* Outer circle */}
          <div className="absolute inset-0 border-2 border-border-light rounded-full"></div>
          {/* Middle circle */}
          <div className="absolute inset-8 border-2 border-border-light rounded-full"></div>
          {/* Inner circle with Kubernetes logo placeholder */}
          <div className="absolute inset-16 border-2 border-qdrant-blue rounded-full flex items-center justify-center bg-surface">
            <div className="text-4xl font-bold text-qdrant-blue">K8s</div>
          </div>
          
          {/* Qdrant logo in center */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="w-12 h-12 bg-qdrant-red rounded-sm flex items-center justify-center">
              <span className="text-white text-lg font-bold">Q</span>
            </div>
          </div>

          {/* ON-PREMISES label */}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-12 text-center">
            <Server className="w-6 h-6 text-foreground-muted mx-auto mb-2" />
            <span className="text-sm font-semibold text-foreground">ON-PREMISES</span>
          </div>

          {/* CLOUD BASED label */}
          <div className="absolute right-0 top-1/2 transform translate-x-12 -translate-y-1/2 text-center">
            <Cloud className="w-6 h-6 text-qdrant-purple mx-auto mb-2" />
            <span className="text-sm font-semibold text-foreground">CLOUD BASED</span>
          </div>

          {/* EDGE BASED label */}
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-12 text-center">
            <Zap className="w-6 h-6 text-qdrant-green mx-auto mb-2" />
            <span className="text-sm font-semibold text-foreground">EDGE BASED</span>
          </div>
        </div>
      </div>

      {/* Additional Description */}
      <div className="max-w-3xl space-y-4 mb-10">
        <p className="text-foreground-muted">
          Qdrant Hybrid Cloud integrates Kubernetes clusters from any environment - cloud, on-premises, or edge - into a unified, enterprise-grade managed service.
        </p>
        <p className="text-foreground-muted">
          Qdrant Hybrid Cloud ensures data privacy, deployment flexibility, low latency, and delivers cost savings, elevating standards for vector search and AI applications.
        </p>
      </div>

      {/* Documentation Link */}
      <div className="mb-8">
        <p className="text-foreground-muted">
          For more information see{" "}
          <Link href="https://qdrant.tech/documentation/hybrid-cloud/" className="text-qdrant-blue hover:underline">
            https://qdrant.tech/documentation/hybrid-cloud/
          </Link>
        </p>
      </div>

      {/* Get Started Button */}
      <button className="btn-primary">
        Get Started
      </button>
    </div>
  );
}
