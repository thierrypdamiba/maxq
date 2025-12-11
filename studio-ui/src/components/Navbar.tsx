"use client";

import { ChevronDown, User } from "lucide-react";
import { useState } from "react";

export function Navbar() {
    const [showAccountMenu, setShowAccountMenu] = useState(false);
    const [showUserMenu, setShowUserMenu] = useState(false);

    return (
        <div className="h-16 border-b border-border-light bg-background px-8 flex items-center justify-end sticky top-0 z-10">
            {/* Right: Create Button and User Profile */}
            <div className="flex items-center gap-4">
                <button className="btn-primary flex items-center gap-2">
                    <span>Create</span>
                    <span className="text-lg leading-none">+</span>
                </button>
                <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
                    <div className="w-8 h-8 rounded-full bg-surface-highlight border border-border-light flex items-center justify-center overflow-hidden">
                        <User className="w-4 h-4 text-foreground-muted" />
                    </div>
                    <span className="text-sm font-medium text-foreground">Thierry Damiba</span>
                    <ChevronDown className="w-4 h-4 text-foreground-muted" />
                </div>
            </div>
        </div>
    );
}
