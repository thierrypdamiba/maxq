"use client";

import { createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "ice";

interface ThemeContextType {
    theme: Theme;
    setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const [theme, setTheme] = useState<Theme>("dark");

    useEffect(() => {
        const savedTheme = localStorage.getItem("theme") as Theme;
        if (savedTheme) {
            setTheme(savedTheme);
            document.documentElement.setAttribute("data-theme", savedTheme);
        }
    }, []);

    const handleSetTheme = (newTheme: Theme) => {
        setTheme(newTheme);
        localStorage.setItem("theme", newTheme);
        if (newTheme === "dark") {
            document.documentElement.removeAttribute("data-theme");
        } else {
            document.documentElement.setAttribute("data-theme", newTheme);
        }
    };

    return (
        <ThemeContext.Provider value={{ theme, setTheme: handleSetTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (context === undefined) {
        throw new Error("useTheme must be used within a ThemeProvider");
    }
    return context;
}
