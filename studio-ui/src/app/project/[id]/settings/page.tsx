"use client";

import { useTheme } from "@/components/ThemeProvider";
import { Moon, Sun } from "lucide-react";

export default function SettingsPage() {
    const { theme, setTheme } = useTheme();

    return (
        <div className="p-10 max-w-4xl mx-auto">
            <div className="mb-10">
                <h1 className="text-2xl font-bold text-foreground tracking-tight">Project Settings</h1>
                <p className="text-foreground-muted text-sm mt-1">Manage appearance, API keys, and team access.</p>
            </div>

            <div className="space-y-8">
                {/* Appearance Section */}
                <div className="glass-panel chrome-border rounded-lg p-6">
                    <h2 className="text-base font-semibold text-foreground mb-4">Appearance</h2>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-foreground">Interface Theme</p>
                            <p className="text-xs text-foreground-muted mt-1">Select your preferred workspace appearance.</p>
                        </div>
                        <div className="flex items-center toggle-container p-1 rounded-lg">
                            <button
                                onClick={() => setTheme("dark")}
                                className={`flex items-center px-3 py-1.5 rounded-md text-xs font-medium transition-all ${theme === "dark"
                                    ? "bg-white/10 text-foreground shadow-sm"
                                    : "text-foreground-muted hover:text-foreground"
                                    }`}
                            >
                                <Moon className="w-3 h-3 mr-2" />
                                Dark (Default)
                            </button>
                            <button
                                onClick={() => setTheme("ice")}
                                className={`flex items-center px-3 py-1.5 rounded-md text-xs font-medium transition-all ${theme === "ice"
                                    ? "bg-white text-black shadow-sm"
                                    : "text-foreground-muted hover:text-foreground"
                                    }`}
                            >
                                <Sun className="w-3 h-3 mr-2" />
                                Ice Blue
                            </button>
                        </div>
                    </div>
                </div>

                {/* API Keys Section (Placeholder) */}
                <div className="glass-panel chrome-border rounded-lg p-6 opacity-50 pointer-events-none">
                    <h2 className="text-base font-semibold text-foreground mb-4">API Keys</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-xs font-medium text-foreground-muted mb-2 uppercase tracking-wide">Qdrant API Key</label>
                            <input type="password" value="sk-qdrant-xxxxxxxx" disabled className="w-full bg-black/20 border border-white/10 rounded-md px-3 py-2 text-foreground-muted text-sm" />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-foreground-muted mb-2 uppercase tracking-wide">OpenAI API Key</label>
                            <input type="password" value="sk-openai-xxxxxxxx" disabled className="w-full bg-black/20 border border-white/10 rounded-md px-3 py-2 text-foreground-muted text-sm" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
