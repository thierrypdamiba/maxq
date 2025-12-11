import { NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";

const PROJECTS_COLLECTION = "maxq_projects";

function getClient() {
    const url = process.env.QDRANT_URL;
    const apiKey = process.env.QDRANT_API_KEY;

    if (!url || !apiKey) {
        throw new Error("Qdrant not configured. Set QDRANT_URL and QDRANT_API_KEY environment variables.");
    }

    return new QdrantClient({ url, apiKey });
}

// GET - Get a single project
export async function GET(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const { id } = await params;
        const client = getClient();

        const result = await client.retrieve(PROJECTS_COLLECTION, {
            ids: [id],
            with_payload: true,
        });

        if (result.length === 0) {
            return NextResponse.json(
                { detail: "Project not found" },
                { status: 404 }
            );
        }

        const point = result[0];

        // Update last_accessed
        await client.setPayload(PROJECTS_COLLECTION, {
            points: [id],
            payload: {
                last_accessed: new Date().toISOString(),
            },
        });

        return NextResponse.json({
            id: point.id,
            ...point.payload,
        });
    } catch (e: any) {
        return NextResponse.json(
            { detail: e.message },
            { status: 500 }
        );
    }
}

// DELETE - Delete a project
export async function DELETE(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const { id } = await params;
        const client = getClient();

        await client.delete(PROJECTS_COLLECTION, {
            points: [id],
        });

        return NextResponse.json({ status: "deleted" });
    } catch (e: any) {
        return NextResponse.json(
            { detail: e.message },
            { status: 500 }
        );
    }
}
