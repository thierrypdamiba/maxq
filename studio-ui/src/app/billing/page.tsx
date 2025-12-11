"use client";

import { CreditCard, AlertTriangle } from "lucide-react";

export default function BillingPage() {
  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Page Title */}
      <h1 className="text-3xl font-bold text-foreground mb-8">Billing Details</h1>

      {/* Payment Information Section */}
      <div className="mb-10">
        <h2 className="text-lg font-semibold text-foreground mb-2">Payment Information</h2>
        <p className="text-sm text-foreground-muted mb-6">Payment Connection Setup.</p>

        {/* Payment Provider Logos */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-surface border border-border-light rounded-lg p-6 flex items-center justify-center">
            <div className="text-center">
              <div className="text-xs font-semibold text-foreground mb-2">VISA</div>
              <div className="text-xs font-semibold text-foreground mb-2">Mastercard</div>
              <div className="text-xs font-semibold text-foreground">AMERICAN EXPRESS</div>
            </div>
          </div>
          <div className="bg-surface border border-border-light rounded-lg p-6 flex items-center justify-center">
            <div className="text-xs font-semibold text-foreground">aws marketplace</div>
          </div>
          <div className="bg-surface border border-border-light rounded-lg p-6 flex items-center justify-center">
            <div className="text-xs font-semibold text-foreground">Google Cloud</div>
          </div>
          <div className="bg-surface border border-border-light rounded-lg p-6 flex items-center justify-center">
            <div className="text-xs font-semibold text-foreground">Microsoft Azure Marketplace</div>
          </div>
        </div>

        {/* Status Message */}
        <div className="mb-4">
          <p className="text-danger flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Payment connection not set up yet.
          </p>
        </div>

        {/* Disclaimer */}
        <p className="text-sm text-foreground-muted">
          All sensitive data is stored on the provider side only.
        </p>
      </div>

      {/* Billing Cycles Section */}
      <div>
        <h2 className="text-lg font-semibold text-foreground mb-6">Billing Cycles</h2>
        <p className="text-foreground-muted">
          You have no Billing Cycles.
        </p>
      </div>
    </div>
  );
}
