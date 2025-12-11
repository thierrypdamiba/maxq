"use client";

import { Rocket, ExternalLink } from "lucide-react";
import Link from "next/link";

export default function GetStartedPage() {
  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Main Heading */}
      <h1 className="text-3xl font-bold text-foreground mb-8">Explore Qdrant Cloud</h1>

      {/* Cloud Quickstart Card */}
      <div className="bg-surface rounded-lg p-6 mb-10">
        <h2 className="text-xl font-bold text-foreground mb-2">Cloud Quickstart</h2>
        <p className="text-foreground-muted mb-4">
          Learn how to get started with Qdrant Cloud in a few steps:
        </p>
        <button className="btn-primary flex items-center gap-2">
          View Quickstart
        </button>
      </div>

      {/* Code Snippet */}
      <div className="bg-surface rounded-lg p-4 mb-10 font-mono text-sm border border-border-light">
        <div className="text-foreground-muted mb-2">
          <span className="text-foreground">curl</span>{" "}
          <span className="text-qdrant-blue">-X GET</span>{" "}
          <span className="text-qdrant-blue">https://xyz-example.eu-central.aws.cloud.qdrant.io:6333</span>{" "}
          <span className="text-foreground">\</span>
        </div>
        <div className="text-foreground-muted">
          <span className="text-foreground">--header</span>{" "}
          <span className="text-foreground-muted">api-key:</span>{" "}
          <span className="text-warning">&lt;your-api-key&gt;</span>
        </div>
        <div className="text-foreground-secondary mt-4 mb-2">
          # Alternatively, you can use the 'Authorization' header with the Bearer prefix
        </div>
        <div className="text-foreground-muted">
          <span className="text-foreground">curl</span>{" "}
          <span className="text-qdrant-blue">-X GET</span>{" "}
          <span className="text-qdrant-blue">https://xyz-example.eu-central.aws.cloud.qdrant.io:6333</span>{" "}
          <span className="text-foreground">\</span>
        </div>
        <div className="text-foreground-muted">
          <span className="text-foreground">--header</span>{" "}
          <span className="text-foreground-muted">Authorization:</span>{" "}
          <span className="text-foreground">Bearer</span>{" "}
          <span className="text-warning">&lt;your-api-key&gt;</span>
        </div>
      </div>

      {/* Explore Your Data Section */}
      <h2 className="text-xl font-bold text-foreground mb-6">Explore Your Data or Start with Samples</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Card 1: Connect to Your Cluster */}
        <div className="bg-surface rounded-lg p-6">
          <div className="w-10 h-10 bg-surface-highlight rounded-lg flex items-center justify-center mb-4">
            <div className="w-6 h-6 border-2 border-qdrant-blue rounded"></div>
          </div>
          <h3 className="text-base font-semibold text-foreground mb-2">Connect to Your Cluster</h3>
          <p className="text-sm text-foreground-muted mb-4">
            Here are the different options you have to connect to your Qdrant cluster
          </p>
          <Link href="#" className="text-sm text-qdrant-purple hover:underline flex items-center gap-1">
            Connect Cluster <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {/* Card 2: Qdrant API */}
        <div className="bg-surface rounded-lg p-6">
          <div className="w-10 h-10 bg-surface-highlight rounded-lg flex items-center justify-center mb-4">
            <span className="text-qdrant-blue text-lg">&lt; &gt;</span>
          </div>
          <h3 className="text-base font-semibold text-foreground mb-2">Qdrant API</h3>
          <p className="text-sm text-foreground-muted mb-4">
            Use Qdrant API to manage, query, and scale vector data effortlessly
          </p>
          <Link href="#" className="text-sm text-qdrant-purple hover:underline flex items-center gap-1">
            Learn More <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {/* Card 3: Qdrant Cluster UI */}
        <div className="bg-surface rounded-lg p-6">
          <div className="w-10 h-10 bg-surface-highlight rounded-lg flex items-center justify-center mb-4">
            <div className="w-6 h-6 border-2 border-qdrant-blue rounded"></div>
          </div>
          <h3 className="text-base font-semibold text-foreground mb-2">Qdrant Cluster UI</h3>
          <p className="text-sm text-foreground-muted mb-4">
            View your collection data, load sample data, and access tutorials.
          </p>
          <Link href="#" className="text-sm text-qdrant-purple hover:underline flex items-center gap-1">
            Qdrant Cluster UI <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {/* Card 4: Sample Data */}
        <div className="bg-surface rounded-lg p-6">
          <div className="w-10 h-10 bg-surface-highlight rounded-lg flex items-center justify-center mb-4">
            <div className="w-5 h-5 border-2 border-qdrant-blue rounded-t"></div>
          </div>
          <h3 className="text-base font-semibold text-foreground mb-2">Sample Data</h3>
          <p className="text-sm text-foreground-muted mb-4">
            Explore Qdrant with our sample data sets.
          </p>
          <Link href="#" className="text-sm text-qdrant-purple hover:underline flex items-center gap-1">
            Load Sample Data <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {/* Card 5: Migrate to Qdrant Cloud */}
        <div className="bg-surface rounded-lg p-6">
          <div className="w-10 h-10 bg-surface-highlight rounded-lg flex items-center justify-center mb-4">
            <div className="w-5 h-5 border-2 border-qdrant-blue rounded-b"></div>
          </div>
          <h3 className="text-base font-semibold text-foreground mb-2">Migrate to Qdrant Cloud</h3>
          <p className="text-sm text-foreground-muted mb-4">
            Migrate your data to from other vector databases or other Qdrant instances to Qdrant Cloud with ease.
          </p>
          <Link href="#" className="text-sm text-qdrant-purple hover:underline flex items-center gap-1">
            Migrate Data <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {/* Card 6: Qdrant Cloud Inference */}
        <div className="bg-surface rounded-lg p-6">
          <div className="w-10 h-10 bg-surface-highlight rounded-lg flex items-center justify-center mb-4">
            <div className="w-5 h-5 border-2 border-qdrant-blue rounded"></div>
          </div>
          <h3 className="text-base font-semibold text-foreground mb-2">Qdrant Cloud Inference</h3>
          <p className="text-sm text-foreground-muted mb-4">
            Learn how to transform your data into vectors directly in Qdrant Cloud.
          </p>
          <Link href="#" className="text-sm text-qdrant-purple hover:underline flex items-center gap-1">
            Use Inference <ExternalLink className="w-3 h-3" />
          </Link>
        </div>
      </div>
    </div>
  );
}
