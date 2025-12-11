"use client";

import { 
  Key, 
  Shield, 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  Eye, 
  EyeOff,
  RefreshCw,
  Save,
  Loader2
} from "lucide-react";
import { useState, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7777";

interface APIKeysStatus {
  qdrant_url: string | null;
  qdrant_configured: boolean;
  openai_configured: boolean;
  linkup_configured: boolean;
}

interface TestResults {
  qdrant: { status: string; message: string | null };
  openai: { status: string; message: string | null };
  linkup: { status: string; message: string | null };
}

export default function SettingsPage() {
  // Form state
  const [qdrantUrl, setQdrantUrl] = useState("");
  const [qdrantApiKey, setQdrantApiKey] = useState("");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [linkupApiKey, setLinkupApiKey] = useState("");
  
  // UI state
  const [showQdrantKey, setShowQdrantKey] = useState(false);
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showLinkupKey, setShowLinkupKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [status, setStatus] = useState<APIKeysStatus | null>(null);
  const [testResults, setTestResults] = useState<TestResults | null>(null);
  const [saveMessage, setSaveMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Load current status on mount
  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/settings/api-keys`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        if (data.qdrant_url) {
          setQdrantUrl(data.qdrant_url);
        }
      }
    } catch (error) {
      console.error("Failed to load settings:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMessage(null);
    
    try {
      const payload: Record<string, string> = {};
      
      // Only include fields that have values
      if (qdrantUrl) payload.qdrant_url = qdrantUrl;
      if (qdrantApiKey) payload.qdrant_api_key = qdrantApiKey;
      if (openaiApiKey) payload.openai_api_key = openaiApiKey;
      if (linkupApiKey) payload.linkup_api_key = linkupApiKey;
      
      const res = await fetch(`${API_BASE}/settings/api-keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      
      if (res.ok) {
        const data = await res.json();
        setSaveMessage({ type: "success", text: "Settings saved successfully!" });
        // Refresh status
        await loadStatus();
        // Clear the key fields (they're saved but we don't show them)
        setQdrantApiKey("");
        setOpenaiApiKey("");
        setLinkupApiKey("");
      } else {
        const error = await res.json();
        setSaveMessage({ type: "error", text: error.detail || "Failed to save settings" });
      }
    } catch (error) {
      setSaveMessage({ type: "error", text: "Failed to connect to server" });
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResults(null);
    
    try {
      const res = await fetch(`${API_BASE}/settings/api-keys/test`, {
        method: "POST",
      });
      
      if (res.ok) {
        const data = await res.json();
        setTestResults(data);
      }
    } catch (error) {
      console.error("Failed to test connections:", error);
    } finally {
      setTesting(false);
    }
  };

  const getStatusIcon = (configured: boolean) => {
    if (configured) {
      return <CheckCircle2 className="w-5 h-5 text-green-500" />;
    }
    return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const getTestStatusBadge = (status: string) => {
    switch (status) {
      case "connected":
      case "configured":
        return <span className="badge badge-success">Connected</span>;
      case "error":
        return <span className="badge badge-danger">Error</span>;
      case "warning":
        return <span className="badge badge-warning">Warning</span>;
      default:
        return <span className="badge badge-muted">Not Configured</span>;
    }
  };

  if (loading) {
    return (
      <div className="p-10 max-w-4xl mx-auto">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-qdrant-blue" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-10 max-w-4xl mx-auto">
      {/* Page Title */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Settings</h1>
        <p className="text-foreground-muted">
          Configure your API keys and credentials for MaxQ services.
        </p>
      </div>

      {/* Current Status Card */}
      <div className="bg-surface rounded-lg border border-border-light p-6 mb-8">
        <div className="flex items-center gap-3 mb-4">
          <Shield className="w-5 h-5 text-foreground" />
          <h2 className="text-lg font-semibold text-foreground">Connection Status</h2>
          <button
            onClick={handleTest}
            disabled={testing}
            className="ml-auto flex items-center gap-2 text-sm text-qdrant-blue hover:underline disabled:opacity-50"
          >
            {testing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Test Connections
          </button>
        </div>

        <div className="grid gap-4">
          {/* Qdrant Status */}
          <div className="flex items-center justify-between p-4 bg-surface-highlight rounded-lg">
            <div className="flex items-center gap-3">
              {getStatusIcon(status?.qdrant_configured || false)}
              <div>
                <p className="font-medium text-foreground">Qdrant Cloud</p>
                <p className="text-sm text-foreground-muted">
                  {status?.qdrant_url || "Not configured"}
                </p>
              </div>
            </div>
            {testResults && getTestStatusBadge(testResults.qdrant.status)}
          </div>

          {/* OpenAI Status */}
          <div className="flex items-center justify-between p-4 bg-surface-highlight rounded-lg">
            <div className="flex items-center gap-3">
              {getStatusIcon(status?.openai_configured || false)}
              <div>
                <p className="font-medium text-foreground">OpenAI</p>
                <p className="text-sm text-foreground-muted">
                  {status?.openai_configured ? "API key configured" : "Not configured"}
                </p>
              </div>
            </div>
            {testResults && getTestStatusBadge(testResults.openai.status)}
          </div>

          {/* Linkup Status */}
          <div className="flex items-center justify-between p-4 bg-surface-highlight rounded-lg">
            <div className="flex items-center gap-3">
              {getStatusIcon(status?.linkup_configured || false)}
              <div>
                <p className="font-medium text-foreground">Linkup</p>
                <p className="text-sm text-foreground-muted">
                  {status?.linkup_configured ? "API key configured" : "Not configured"}
                </p>
              </div>
            </div>
            {testResults && getTestStatusBadge(testResults.linkup.status)}
          </div>
        </div>

        {/* Test Results Messages */}
        {testResults && (
          <div className="mt-4 space-y-2">
            {testResults.qdrant.message && (
              <p className={`text-sm ${testResults.qdrant.status === "error" ? "text-red-500" : "text-green-500"}`}>
                Qdrant: {testResults.qdrant.message}
              </p>
            )}
            {testResults.openai.message && (
              <p className={`text-sm ${testResults.openai.status === "error" ? "text-red-500" : "text-green-500"}`}>
                OpenAI: {testResults.openai.message}
              </p>
            )}
            {testResults.linkup.message && (
              <p className={`text-sm ${testResults.linkup.status === "error" ? "text-red-500" : "text-green-500"}`}>
                Linkup: {testResults.linkup.message}
              </p>
            )}
          </div>
        )}
      </div>

      {/* API Keys Form */}
      <div className="bg-surface rounded-lg border border-border-light p-6 mb-8">
        <div className="flex items-center gap-3 mb-6">
          <Key className="w-5 h-5 text-foreground" />
          <h2 className="text-lg font-semibold text-foreground">API Credentials</h2>
        </div>

        <div className="space-y-6">
          {/* Qdrant URL */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Qdrant Cloud URL
            </label>
            <input
              type="text"
              value={qdrantUrl}
              onChange={(e) => setQdrantUrl(e.target.value)}
              placeholder="https://your-cluster.cloud.qdrant.io"
              className="w-full px-4 py-2 bg-surface border border-border-light rounded-lg text-foreground placeholder:text-foreground-muted focus:outline-none focus:ring-2 focus:ring-qdrant-blue focus:border-transparent"
            />
            <p className="text-xs text-foreground-muted mt-1">
              Get this from your Qdrant Cloud dashboard
            </p>
          </div>

          {/* Qdrant API Key */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Qdrant API Key
            </label>
            <div className="relative">
              <input
                type={showQdrantKey ? "text" : "password"}
                value={qdrantApiKey}
                onChange={(e) => setQdrantApiKey(e.target.value)}
                placeholder={status?.qdrant_configured ? "********** (configured)" : "Enter your Qdrant API key"}
                className="w-full px-4 py-2 pr-10 bg-surface border border-border-light rounded-lg text-foreground placeholder:text-foreground-muted focus:outline-none focus:ring-2 focus:ring-qdrant-blue focus:border-transparent"
              />
              <button
                type="button"
                onClick={() => setShowQdrantKey(!showQdrantKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground-muted hover:text-foreground"
              >
                {showQdrantKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* OpenAI API Key */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              OpenAI API Key
              <span className="text-foreground-muted font-normal ml-2">(optional)</span>
            </label>
            <div className="relative">
              <input
                type={showOpenaiKey ? "text" : "password"}
                value={openaiApiKey}
                onChange={(e) => setOpenaiApiKey(e.target.value)}
                placeholder={status?.openai_configured ? "********** (configured)" : "sk-..."}
                className="w-full px-4 py-2 pr-10 bg-surface border border-border-light rounded-lg text-foreground placeholder:text-foreground-muted focus:outline-none focus:ring-2 focus:ring-qdrant-blue focus:border-transparent"
              />
              <button
                type="button"
                onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground-muted hover:text-foreground"
              >
                {showOpenaiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-xs text-foreground-muted mt-1">
              Required for RAG answer generation features
            </p>
          </div>

          {/* Linkup API Key */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Linkup API Key
              <span className="text-foreground-muted font-normal ml-2">(optional)</span>
            </label>
            <div className="relative">
              <input
                type={showLinkupKey ? "text" : "password"}
                value={linkupApiKey}
                onChange={(e) => setLinkupApiKey(e.target.value)}
                placeholder={status?.linkup_configured ? "********** (configured)" : "Enter your Linkup API key"}
                className="w-full px-4 py-2 pr-10 bg-surface border border-border-light rounded-lg text-foreground placeholder:text-foreground-muted focus:outline-none focus:ring-2 focus:ring-qdrant-blue focus:border-transparent"
              />
              <button
                type="button"
                onClick={() => setShowLinkupKey(!showLinkupKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground-muted hover:text-foreground"
              >
                {showLinkupKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-xs text-foreground-muted mt-1">
              Required for web search augmented retrieval
            </p>
          </div>
        </div>

        {/* Save Message */}
        {saveMessage && (
          <div className={`mt-4 p-3 rounded-lg ${
            saveMessage.type === "success" 
              ? "bg-green-500/10 text-green-500 border border-green-500/20" 
              : "bg-red-500/10 text-red-500 border border-red-500/20"
          }`}>
            <div className="flex items-center gap-2">
              {saveMessage.type === "success" ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <XCircle className="w-4 h-4" />
              )}
              {saveMessage.text}
            </div>
          </div>
        )}

        {/* Save Button */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-6 py-2 bg-qdrant-blue text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {saving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Save Credentials
          </button>
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <h3 className="font-medium text-foreground mb-1">Security Note</h3>
            <p className="text-sm text-foreground-muted">
              Your API keys are stored locally in <code className="bg-surface px-1 py-0.5 rounded">~/.maxq/.env</code> and are never sent to external servers except their respective services (Qdrant, OpenAI).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
