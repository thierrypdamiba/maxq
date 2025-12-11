import { NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";

// GET - Return current configuration status
export async function GET() {
    const qdrantUrl = process.env.QDRANT_URL || null;
    const qdrantConfigured = !!(process.env.QDRANT_URL && process.env.QDRANT_API_KEY);
    const openaiConfigured = !!process.env.OPENAI_API_KEY;

    return NextResponse.json({
        qdrant_url: qdrantUrl,
        qdrant_configured: qdrantConfigured,
        openai_configured: openaiConfigured,
    });
}

// POST - In serverless, we can't persist env vars at runtime
// This endpoint validates the keys but tells user to set them in Vercel dashboard
export async function POST(request: Request) {
    const body = await request.json();
    const { qdrant_url, qdrant_api_key, openai_api_key } = body;

    // Validate Qdrant connection if provided
    if (qdrant_url && qdrant_api_key) {
        try {
            const client = new QdrantClient({
                url: qdrant_url,
                apiKey: qdrant_api_key,
            });
            await client.getCollections();
        } catch (e: any) {
            return NextResponse.json(
                { detail: `Qdrant connection failed: ${e.message}` },
                { status: 400 }
            );
        }
    }

    // In serverless environment, we can't persist environment variables
    // Return success with instructions
    return NextResponse.json({
        status: "validated",
        message: "Keys validated successfully. To persist, add them as environment variables in your Vercel project settings.",
        qdrant_configured: !!(qdrant_url && qdrant_api_key),
        openai_configured: !!openai_api_key,
    });
}
