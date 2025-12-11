"""
Projects API router.

Uses SQLite database for persistence.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
import uuid

from ..models import Project, ProjectStatus
from ..database import ProjectStore

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=List[Project])
async def list_projects():
    """List all projects, sorted by last accessed."""
    return ProjectStore.list_all()


@router.post("/", response_model=Project)
async def create_project(name: str, description: str = None, task_type: str = "general"):
    """Create a new project."""
    project = Project(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        created_at=datetime.now(),
        last_accessed=datetime.now(),
        task_type=task_type
    )
    return ProjectStore.create(project)


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get a project by ID."""
    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update last accessed
    ProjectStore.touch(project_id)
    project.last_accessed = datetime.now()
    return project


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all associated data."""
    if not ProjectStore.delete(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "project_id": project_id}


@router.get("/{project_id}/indexed-models")
async def get_indexed_models(project_id: str):
    """Get all embedding models that have been used to index data for this project."""
    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project_id": project_id,
        "indexed_models": [
            {
                "model_name": m.model_name,
                "collection_name": m.collection_name,
                "indexed_at": m.indexed_at.isoformat() if m.indexed_at else None,
                "point_count": m.point_count
            }
            for m in project.indexed_models
        ],
        "default_model": project.embedding_model
    }


@router.get("/{project_id}/export-react")
async def export_react_component(project_id: str, api_url: str = "http://localhost:8000"):
    """
    Generate a React search component for this project.

    Returns downloadable JSX file.
    """
    from fastapi.responses import Response
    from maxq.component_generator import generate_react_component

    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    component_code = generate_react_component(
        project_name=project.name,
        project_id=project_id,
        api_url=api_url
    )

    return Response(
        content=component_code,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="MaxQSearch-{project.name.replace(" ", "_")}.jsx"'
        }
    )


@router.post("/{project_id}/export-snapshot")
async def export_snapshot(project_id: str, collection_name: str = None):
    """
    Create a Qdrant snapshot of the project's collection.

    Returns snapshot info and download URL.
    """
    from maxq.server.dependencies import get_engine

    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.indexed_models:
        raise HTTPException(
            status_code=400,
            detail="No data indexed for this project. Index data first."
        )

    # Use specified collection or first indexed model's collection
    if collection_name:
        coll = collection_name
    else:
        coll = project.indexed_models[0].collection_name

    try:
        engine = get_engine()

        # Check if collection exists
        if not engine.client.collection_exists(coll):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{coll}' not found"
            )

        # Create snapshot
        snapshot_info = engine.client.create_snapshot(collection_name=coll)

        return {
            "snapshot_name": snapshot_info.name,
            "collection_name": coll,
            "project_id": project_id,
            "download_url": f"/collections/{coll}/snapshots/{snapshot_info.name}",
            "message": "Snapshot created successfully. Use the download_url with your Qdrant instance to retrieve it."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create snapshot: {str(e)}"
        )


@router.get("/{project_id}/export-config")
async def export_project_config(project_id: str):
    """
    Export project configuration as JSON.

    Useful for replicating the project setup.
    """
    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "task_type": project.task_type,
            "created_at": project.created_at.isoformat() if project.created_at else None,
        },
        "indexed_models": [
            {
                "model_name": m.model_name,
                "collection_name": m.collection_name,
                "point_count": m.point_count
            }
            for m in project.indexed_models
        ],
        "default_embedding_model": project.embedding_model,
        "exported_at": datetime.now().isoformat(),
        "maxq_version": "0.1.0"
    }
