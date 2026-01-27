"use client";

import { useState, useRef, useEffect } from "react";
import { getApiUrl } from "@/lib/api";

interface Message {
    role: "user" | "assistant";
    content: string;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [scope, setScope] = useState<"cluster" | "collection" | "point">("cluster");
    const [collectionName, setCollectionName] = useState("");
    const [pointId, setPointId] = useState("");
    const [collections, setCollections] = useState<string[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Load collections on mount
    useEffect(() => {
        async function loadCollections() {
            try {
                const res = await fetch(`${getApiUrl()}/health`);
                if (!res.ok) return;
                // Fetch collections via cleanup analyze endpoint
                const analyzeRes = await fetch(`${getApiUrl()}/cleanup/analyze`);
                if (analyzeRes.ok) {
                    const data = await analyzeRes.json();
                    setCollections(data.collections?.map((c: any) => c.name) || []);
                }
            } catch {
                // Server not running
            }
        }
        loadCollections();
    }, []);

    async function sendMessage() {
        if (!input.trim() || isStreaming) return;

        const userMessage: Message = { role: "user", content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setIsStreaming(true);

        const assistantMessage: Message = { role: "assistant", content: "" };
        setMessages(prev => [...prev, assistantMessage]);

        try {
            const res = await fetch(`${getApiUrl()}/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: input,
                    scope,
                    collection_name: scope !== "cluster" ? collectionName : undefined,
                    point_id: scope === "point" ? pointId : undefined,
                    history: messages.map(m => ({ role: m.role, content: m.content })),
                }),
            });

            const reader = res.body?.getReader();
            const decoder = new TextDecoder();
            let fullText = "";

            if (reader) {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split("\n");

                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            const data = line.slice(6);
                            if (data === "[DONE]") continue;
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.token) {
                                    fullText += parsed.token;
                                    setMessages(prev => {
                                        const updated = [...prev];
                                        updated[updated.length - 1] = {
                                            role: "assistant",
                                            content: fullText,
                                        };
                                        return updated;
                                    });
                                }
                                if (parsed.error) {
                                    fullText += `\n\nError: ${parsed.error}`;
                                    setMessages(prev => {
                                        const updated = [...prev];
                                        updated[updated.length - 1] = {
                                            role: "assistant",
                                            content: fullText,
                                        };
                                        return updated;
                                    });
                                }
                            } catch {
                                // skip malformed JSON
                            }
                        }
                    }
                }
            }
        } catch (e) {
            setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                    role: "assistant",
                    content: `Connection error: ${e instanceof Error ? e.message : "Unknown error"}. Is the MaxQ server running?`,
                };
                return updated;
            });
        }

        setIsStreaming(false);
    }

    return (
        <div className="flex flex-col h-full">
            <div className="border-b border-border-light p-6">
                <h1 className="text-2xl font-bold text-foreground">Chat</h1>
                <p className="text-foreground-muted text-sm mt-1">
                    Chat with your Qdrant clusters, collections, and points
                </p>

                {/* Scope selector */}
                <div className="flex items-center gap-4 mt-4">
                    <div className="flex gap-1 bg-sidebar-bg rounded-lg p-1">
                        {(["cluster", "collection", "point"] as const).map((s) => (
                            <button
                                key={s}
                                onClick={() => setScope(s)}
                                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                                    scope === s
                                        ? "bg-qdrant-red text-white"
                                        : "text-foreground-muted hover:text-foreground"
                                }`}
                            >
                                {s.charAt(0).toUpperCase() + s.slice(1)}
                            </button>
                        ))}
                    </div>

                    {scope !== "cluster" && (
                        <select
                            value={collectionName}
                            onChange={(e) => setCollectionName(e.target.value)}
                            className="bg-sidebar-bg border border-border-light rounded-md px-3 py-1.5 text-sm text-foreground"
                        >
                            <option value="">Select collection...</option>
                            {collections.map((c) => (
                                <option key={c} value={c}>{c}</option>
                            ))}
                        </select>
                    )}

                    {scope === "point" && (
                        <input
                            type="text"
                            placeholder="Point ID"
                            value={pointId}
                            onChange={(e) => setPointId(e.target.value)}
                            className="bg-sidebar-bg border border-border-light rounded-md px-3 py-1.5 text-sm text-foreground w-48"
                        />
                    )}
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-foreground-muted py-20">
                        <p className="text-lg font-medium">Start a conversation</p>
                        <p className="text-sm mt-2">
                            {scope === "cluster" && "Ask about your collections, cluster health, and stats"}
                            {scope === "collection" && "Search and chat with your collection data"}
                            {scope === "point" && "Examine and understand specific points"}
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div
                        key={i}
                        className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                        <div
                            className={`max-w-[80%] rounded-lg px-4 py-3 text-sm whitespace-pre-wrap ${
                                msg.role === "user"
                                    ? "bg-qdrant-red text-white"
                                    : "bg-sidebar-bg text-foreground border border-border-light"
                            }`}
                        >
                            {msg.content || (isStreaming && i === messages.length - 1 ? "..." : "")}
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-border-light p-4">
                <div className="flex gap-3">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
                        placeholder={
                            scope === "cluster"
                                ? "Ask about your cluster..."
                                : scope === "collection"
                                ? "Search and ask about your data..."
                                : "Ask about this point..."
                        }
                        disabled={isStreaming}
                        className="flex-1 bg-sidebar-bg border border-border-light rounded-lg px-4 py-2.5 text-sm text-foreground placeholder:text-foreground-muted focus:outline-none focus:border-qdrant-red"
                    />
                    <button
                        onClick={sendMessage}
                        disabled={isStreaming || !input.trim()}
                        className="bg-qdrant-red text-white px-6 py-2.5 rounded-lg text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
                    >
                        {isStreaming ? "..." : "Send"}
                    </button>
                </div>
            </div>
        </div>
    );
}
