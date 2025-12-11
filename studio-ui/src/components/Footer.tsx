"use client";

export function Footer() {
    return (
        <div className="border-t border-border-light bg-background px-8 py-4">
            <div className="flex items-center gap-6 text-sm text-foreground-secondary">
                <a href="/status" className="hover:text-foreground transition-colors">
                    Status
                </a>
                <a href="/terms" className="hover:text-foreground transition-colors">
                    Terms
                </a>
                <a href="/privacy" className="hover:text-foreground transition-colors">
                    Privacy
                </a>
                <a href="/sla" className="hover:text-foreground transition-colors">
                    SLA
                </a>
                <a href="/security" className="hover:text-foreground transition-colors">
                    Security & Compliance
                </a>
            </div>
        </div>
    );
}
