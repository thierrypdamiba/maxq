// ============================================
// MaxQ API Client
// ============================================

// Default to FastAPI backend on port 8888 for local development
const DEFAULT_API_URL = "http://localhost:8888";

// ============================================
// Configuration
// ============================================

export function getApiUrl(): string {
    if (typeof window === "undefined") return DEFAULT_API_URL;
    const stored = localStorage.getItem("maxq_api_url");
    return stored || DEFAULT_API_URL;
}

export function setApiUrl(url: string): void {
    localStorage.setItem("maxq_api_url", url);
}

// ============================================
// Core Fetch Wrapper
// ============================================

async function apiFetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const baseUrl = getApiUrl();
    // Use /api prefix for Next.js API routes
    const apiEndpoint = baseUrl ? `${baseUrl}${endpoint}` : `/api${endpoint}`;
    return fetch(apiEndpoint, options);
}

// ============================================
// Health Check
// ============================================

export async function testConnection(): Promise<{ ok: boolean; error?: string }> {
    try {
        const res = await apiFetch("/health");
        if (res.ok) {
            return { ok: true };
        }
        return { ok: false, error: `Server returned ${res.status}` };
    } catch (e) {
        return { ok: false, error: "Connection failed. Is the server running?" };
    }
}

// ============================================
// Project API
// ============================================

export async function fetchProjects() {
    const res = await apiFetch("/projects");
    if (!res.ok) throw new Error("Failed to fetch projects");
    return res.json();
}

export async function createProject(name: string, description?: string, task_type: string = "general") {
    const res = await apiFetch(`/projects?name=${encodeURIComponent(name)}&description=${encodeURIComponent(description || "")}&task_type=${encodeURIComponent(task_type)}`, {
        method: "POST",
    });
    if (!res.ok) throw new Error("Failed to create project");
    return res.json();
}

export async function fetchProject(id: string) {
    const res = await apiFetch(`/projects/${id}`);
    if (!res.ok) throw new Error("Failed to fetch project");
    return res.json();
}

// ============================================
// Indexing API (Simple)
// ============================================

export interface IndexResult {
    success: boolean;
    collection_name: string;
    points_indexed: number;
    message: string;
}

export async function indexDataset(
    projectId: string,
    datasetName: string,
    embeddingModel: string = "sentence-transformers/all-MiniLM-L6-v2",
    limit: number = 500
): Promise<IndexResult> {
    const res = await apiFetch("/index/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            project_id: projectId,
            dataset_name: datasetName,
            embedding_model: embeddingModel,
            limit: limit
        })
    });

    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Indexing failed");
    }

    return res.json();
}

export interface IndexedModel {
    model_name: string;
    collection_name: string;
    indexed_at: string | null;
    point_count: number;
}

export async function fetchIndexStatus(projectId: string): Promise<{
    project_id: string;
    indexed_models: IndexedModel[];
    has_data: boolean;
}> {
    const res = await apiFetch(`/index/status/${projectId}`);
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to fetch index status");
    }
    return res.json();
}

// Alias for backward compatibility
export async function fetchIndexedModels(projectId: string): Promise<{
    project_id: string;
    indexed_models: IndexedModel[];
    default_model: string | null;
}> {
    const status = await fetchIndexStatus(projectId);
    return {
        project_id: status.project_id,
        indexed_models: status.indexed_models,
        default_model: status.indexed_models[0]?.model_name || null
    };
}

export async function fetchEmbeddingModels(): Promise<Array<{model: string; dim: number; description: string}>> {
    const res = await apiFetch("/index/models");
    if (!res.ok) throw new Error("Failed to fetch embedding models");
    return res.json();
}

// ============================================
// Search API (Simple)
// ============================================

export interface SearchResult {
    id: string;
    score: number;
    text: string;
    metadata: Record<string, any>;
}

export interface SearchResponse {
    results: SearchResult[];
    model_used: string;
    collection_name: string;
    strategy: string;
}

export async function searchProject(
    projectId: string,
    query: string,
    strategy: "hybrid" | "dense" | "sparse" = "hybrid",
    modelName?: string
): Promise<SearchResponse> {
    const res = await apiFetch("/search/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            project_id: projectId,
            query: query,
            strategy: strategy,
            model_name: modelName
        })
    });

    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Search failed");
    }

    return res.json();
}

// ============================================
// Tuning & Evals API
// ============================================

export async function fetchExperiments(projectId: string) {
    const res = await apiFetch(`/tuning/?project_id=${projectId}`);
    if (!res.ok) throw new Error("Failed to fetch experiments");
    return res.json();
}

export async function createExperiment(projectId: string, name: string, embeddingModel: string, searchStrategy: string = "hybrid") {
    const res = await apiFetch(`/tuning/?project_id=${projectId}&name=${encodeURIComponent(name)}&embedding_model=${encodeURIComponent(embeddingModel)}&search_strategy=${encodeURIComponent(searchStrategy)}`, {
        method: "POST",
    });
    if (!res.ok) throw new Error("Failed to create experiment");
    return res.json();
}

export async function runEval(experimentId: string) {
    const res = await apiFetch(`/evals/run?experiment_id=${experimentId}`, {
        method: "POST",
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to run evaluation");
    }
    return res.json();
}

// ============================================
// Export API
// ============================================

export function getReactComponentUrl(projectId: string, apiUrl?: string): string {
    const baseUrl = getApiUrl();
    const targetUrl = apiUrl || baseUrl;
    return `${baseUrl}/projects/${projectId}/export-react?api_url=${encodeURIComponent(targetUrl)}`;
}

export async function exportProjectConfig(projectId: string) {
    const res = await apiFetch(`/projects/${projectId}/export-config`);
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to export config");
    }
    return res.json();
}

export async function createSnapshot(projectId: string, collectionName?: string) {
    const res = await apiFetch(`/projects/${projectId}/export-snapshot${collectionName ? `?collection_name=${encodeURIComponent(collectionName)}` : ""}`, {
        method: "POST",
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to create snapshot");
    }
    return res.json();
}

export function getPdfReportUrl(evalId: string): string {
    return `${getApiUrl()}/evals/${evalId}/report.pdf`;
}

// ============================================
// Settings API
// ============================================

export interface APIKeysStatus {
    qdrant_url: string | null;
    qdrant_configured: boolean;
    openai_configured: boolean;
}

export interface APIKeys {
    qdrant_url?: string;
    qdrant_api_key?: string;
    openai_api_key?: string;
}

export interface TestResult {
    status: "not_configured" | "connected" | "configured" | "warning" | "error";
    message: string | null;
}

export async function fetchAPIKeysStatus(): Promise<APIKeysStatus> {
    const res = await apiFetch("/settings/api-keys");
    if (!res.ok) throw new Error("Failed to fetch API keys status");
    return res.json();
}

export async function saveAPIKeys(keys: APIKeys): Promise<{ status: string; message: string; qdrant_configured: boolean; openai_configured: boolean }> {
    const res = await apiFetch("/settings/api-keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(keys)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to save API keys");
    }
    return res.json();
}

export async function testAPIKeys(): Promise<{ qdrant: TestResult; openai: TestResult }> {
    const res = await apiFetch("/settings/api-keys/test", {
        method: "POST"
    });
    if (!res.ok) throw new Error("Failed to test API keys");
    return res.json();
}
