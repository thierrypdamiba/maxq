import { NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";

type TestStatus = "not_configured" | "connected" | "configured" | "warning" | "error";

interface TestResult {
    status: TestStatus;
    message: string;
}

export async function POST() {
    const qdrantUrl = process.env.QDRANT_URL;
    const qdrantApiKey = process.env.QDRANT_API_KEY;
    const openaiApiKey = process.env.OPENAI_API_KEY;

    // Test Qdrant
    let qdrantResult: TestResult = {
        status: "not_configured",
        message: "Qdrant URL and API key not configured",
    };

    if (qdrantUrl && qdrantApiKey) {
        try {
            const client = new QdrantClient({
                url: qdrantUrl,
                apiKey: qdrantApiKey,
            });
            const collections = await client.getCollections();
            qdrantResult = {
                status: "connected",
                message: `Connected. ${collections.collections.length} collections found.`,
            };
        } catch (e: unknown) {
            const error = e as Error;
            qdrantResult = {
                status: "error",
                message: error.message?.slice(0, 100) || "Connection failed",
            };
        }
    }

    // Test OpenAI
    let openaiResult: TestResult = {
        status: "not_configured",
        message: "OpenAI API key not configured (optional)",
    };

    if (openaiApiKey) {
        try {
            const response = await fetch("https://api.openai.com/v1/models", {
                headers: {
                    Authorization: `Bearer ${openaiApiKey}`,
                },
            });
            if (response.ok) {
                openaiResult = {
                    status: "connected",
                    message: "OpenAI API key is valid",
                };
            } else {
                openaiResult = {
                    status: "error",
                    message: `API returned ${response.status}`,
                };
            }
        } catch (e: unknown) {
            const error = e as Error;
            openaiResult = {
                status: "error",
                message: error.message?.slice(0, 100) || "Connection failed",
            };
        }
    }

    return NextResponse.json({
        qdrant: qdrantResult,
        openai: openaiResult,
    });
}
