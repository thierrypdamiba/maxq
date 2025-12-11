"use client";

import { Archive, ChevronDown } from "lucide-react";

export default function BackupsPage() {
  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Page Title */}
      <h1 className="text-3xl font-bold text-foreground mb-8">Backups</h1>

      {/* Schedules Section */}
      <div className="mb-10">
        <div className="border-b border-border-light pb-4 mb-6">
          <h2 className="text-lg font-semibold text-foreground">Schedules</h2>
        </div>
        <p className="text-foreground-muted">
          There are no Backup Schedules and you have no clusters eligible for Backups.
        </p>
      </div>

      {/* Available Backups Section */}
      <div>
        <div className="border-b border-border-light pb-4 mb-6 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">Available Backups</h2>
          <button className="flex items-center gap-2 px-4 py-2 bg-surface border border-border-light rounded-md text-foreground hover:bg-surface-highlight transition-colors">
            Backup Now
            <ChevronDown className="w-4 h-4" />
          </button>
        </div>
        <p className="text-foreground-muted">
          There are no Backups available.
        </p>
      </div>
    </div>
  );
}
