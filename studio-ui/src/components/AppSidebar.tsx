"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    Rocket,
    LayoutGrid,
    Cloud,
    Archive,
    Users,
    CreditCard,
    Settings,
    HelpCircle
} from "lucide-react";
import { cn } from "@/lib/utils";

export function AppSidebar() {
    const pathname = usePathname();

    // Qdrant-style navigation structure
    const dashboardItems = [
        { name: "Get Started", href: "/get-started", icon: Rocket },
        { name: "Clusters", href: "/clusters", icon: LayoutGrid },
        { name: "Hybrid Cloud", href: "/hybrid-cloud", icon: Cloud },
        { name: "Backups", href: "/backups", icon: Archive },
    ];

    const accountItems = [
        { name: "Access Management", href: "/access-management", icon: Users },
        { name: "Billing", href: "/billing", icon: CreditCard },
        { name: "Settings", href: "/settings", icon: Settings },
    ];

    return (
        <div className="flex h-screen w-64 flex-col bg-sidebar-bg text-foreground border-r border-border-light">
            {/* Qdrant Logo and Account Name */}
            <div className="flex h-16 items-center px-6 border-b border-border-light">
                <div className="w-6 h-6 bg-qdrant-red rounded-sm flex items-center justify-center mr-3 flex-shrink-0">
                    <span className="text-white text-xs font-bold">Q</span>
                </div>
                <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
                    <span className="text-sm font-medium text-foreground">Thierry Damiba's Personal Account</span>
                    <svg className="w-4 h-4 text-foreground-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto py-4">
                {/* DASHBOARD Section */}
                <div className="px-6 mb-6">
                    <div className="text-xs font-semibold text-foreground-muted uppercase tracking-wider mb-3">
                        DASHBOARD
                    </div>
                    <nav className="space-y-1">
                        {dashboardItems.map((item) => {
                            const isActive = pathname === item.href || (item.href === "/get-started" && pathname === "/");
                            return (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={cn(
                                        "group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-all duration-200",
                                        isActive
                                            ? "nav-item-active"
                                            : "text-foreground-muted hover:text-foreground hover:bg-sidebar-hover-bg"
                                    )}
                                >
                                    <item.icon className={cn("mr-3 h-4 w-4 flex-shrink-0 transition-colors",
                                        isActive ? "text-white" : "text-foreground-muted group-hover:text-foreground")} />
                                    {item.name}
                                </Link>
                            );
                        })}
                    </nav>
                </div>

                {/* Divider */}
                <div className="border-t border-border-light my-4"></div>

                {/* ACCOUNT Section */}
                <div className="px-6">
                    <div className="text-xs font-semibold text-foreground-muted uppercase tracking-wider mb-3">
                        ACCOUNT
                    </div>
                    <nav className="space-y-1">
                        {accountItems.map((item) => {
                            const isActive = pathname === item.href;
                            return (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={cn(
                                        "group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-all duration-200",
                                        isActive
                                            ? "nav-item-active"
                                            : "text-foreground-muted hover:text-foreground hover:bg-sidebar-hover-bg"
                                    )}
                                >
                                    <item.icon className={cn("mr-3 h-4 w-4 flex-shrink-0 transition-colors",
                                        isActive ? "text-white" : "text-foreground-muted group-hover:text-foreground")} />
                                    {item.name}
                                </Link>
                            );
                        })}
                    </nav>
                </div>
            </div>

            {/* Get Support Link */}
            <div className="border-t border-border-light p-4">
                <Link
                    href="/support"
                    className="flex items-center text-sm text-foreground-muted hover:text-foreground transition-colors"
                >
                    <HelpCircle className="w-4 h-4 mr-3" />
                    Get Support
                </Link>
            </div>
        </div>
    );
}
