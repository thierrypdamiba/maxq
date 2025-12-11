import { NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";

export async function GET() {
    const qdrantUrl = process.env.QDRANT_URL;
    const qdrantApiKey = process.env.QDRANT_API_KEY;

    let qdrantStatus = "not_configured";

    if (qdrantUrl && qdrantApiKey) {
        try {
            const client = new QdrantClient({
                url: qdrantUrl,
                apiKey: qdrantApiKey,
            });
            await client.getCollections();
            qdrantStatus = "healthy";
        } catch (e: any) {
            qdrantStatus = `unhealthy: ${e.message?.slice(0, 50)}`;
        }
    }

    return NextResponse.json({
        status: qdrantStatus === "healthy" ? "ok" : "degraded",
        version: "0.0.1",
        timestamp: new Date().toISOString(),
        services: {
            qdrant: {
                status: qdrantStatus,
                configured: !!(qdrantUrl && qdrantApiKey),
            },
        },
    });
}
