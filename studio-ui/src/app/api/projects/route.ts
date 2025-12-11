import { NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";
import { v4 as uuidv4 } from "uuid";

const PROJECTS_COLLECTION = "maxq_projects";

function getClient() {
    const url = process.env.QDRANT_URL;
    const apiKey = process.env.QDRANT_API_KEY;

    if (!url || !apiKey) {
        throw new Error("Qdrant not configured. Set QDRANT_URL and QDRANT_API_KEY environment variables.");
    }

    return new QdrantClient({ url, apiKey });
}

async function ensureProjectsCollection(client: QdrantClient) {
    try {
        await client.getCollection(PROJECTS_COLLECTION);
    } catch {
        // Collection doesn't exist, create it
        await client.createCollection(PROJECTS_COLLECTION, {
            vectors: {
                size: 4,  // Minimal vector size for metadata-only storage
                distance: "Cosine",
            },
        });
    }
}

// GET - List all projects
export async function GET() {
    try {
        const client = getClient();
        await ensureProjectsCollection(client);

        const result = await client.scroll(PROJECTS_COLLECTION, {
            limit: 100,
            with_payload: true,
        });

        const projects = result.points.map((point) => ({
            id: point.id,
            ...point.payload,
        }));

        return NextResponse.json(projects);
    } catch (e: any) {
        return NextResponse.json(
            { detail: e.message },
            { status: 500 }
        );
    }
}

// POST - Create a new project
export async function POST(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const name = searchParams.get("name");
        const description = searchParams.get("description") || "";
        const taskType = searchParams.get("task_type") || "general";

        if (!name) {
            return NextResponse.json(
                { detail: "Project name is required" },
                { status: 400 }
            );
        }

        const client = getClient();
        await ensureProjectsCollection(client);

        const projectId = uuidv4();
        const now = new Date().toISOString();

        const project = {
            name,
            description,
            task_type: taskType,
            created_at: now,
            last_accessed: now,
        };

        await client.upsert(PROJECTS_COLLECTION, {
            points: [
                {
                    id: projectId,
                    vector: [0, 0, 0, 0],  // Dummy vector for metadata storage
                    payload: project,
                },
            ],
        });

        return NextResponse.json({ id: projectId, ...project });
    } catch (e: any) {
        return NextResponse.json(
            { detail: e.message },
            { status: 500 }
        );
    }
}
