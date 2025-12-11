"""
SQLite database module for MaxQ project persistence.

Provides proper database storage replacing JSON files.
"""

import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from maxq.config import MAXQ_APP_DIR
from .models import Project, ProjectStatus, IndexedModelInfo, Experiment, EvalResult

logger = logging.getLogger(__name__)

DATABASE_PATH = MAXQ_APP_DIR / "maxq.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    MAXQ_APP_DIR.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()

        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                task_type TEXT,
                embedding_model TEXT DEFAULT 'BAAI/bge-base-en-v1.5',
                last_accessed TEXT
            )
        """)

        # Indexed models table (related to projects)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                point_count INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, model_name)
            )
        """)

        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                chunk_size INTEGER DEFAULT 512,
                search_strategy TEXT DEFAULT 'hybrid',
                progress_current INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0,
                progress_message TEXT,
                started_at TEXT,
                updated_at TEXT,
                metrics TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        # Eval results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                metrics TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        """)

        # Indexing jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexing_jobs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                dataset_name TEXT,
                embedding_model TEXT,
                chunk_size INTEGER,
                chunk_overlap INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                progress_current INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0,
                progress_message TEXT,
                points_created INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexed_models_project ON indexed_models(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_project ON indexing_jobs(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON indexing_jobs(status)")

        logger.info(f"Database initialized at {DATABASE_PATH}")


def migrate_from_json():
    """Migrate existing JSON data to SQLite."""
    json_file = MAXQ_APP_DIR / "projects.json"

    if not json_file.exists():
        logger.info("No JSON data to migrate")
        return

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        if not data:
            return

        with get_db() as conn:
            cursor = conn.cursor()

            for item in data:
                # Check if already migrated
                cursor.execute("SELECT id FROM projects WHERE id = ?", (item['id'],))
                if cursor.fetchone():
                    continue

                # Insert project
                cursor.execute("""
                    INSERT INTO projects (id, name, description, created_at, status, task_type, embedding_model, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item['id'],
                    item['name'],
                    item.get('description'),
                    item['created_at'],
                    item.get('status', 'active'),
                    item.get('task_type'),
                    item.get('embedding_model', 'BAAI/bge-base-en-v1.5'),
                    item.get('last_accessed')
                ))

                # Insert indexed models
                for model in item.get('indexed_models', []):
                    cursor.execute("""
                        INSERT OR IGNORE INTO indexed_models (project_id, model_name, collection_name, indexed_at, point_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        item['id'],
                        model['model_name'],
                        model['collection_name'],
                        model['indexed_at'],
                        model.get('point_count', 0)
                    ))

        # Rename old file as backup
        backup_file = json_file.with_suffix('.json.migrated')
        json_file.rename(backup_file)
        logger.info(f"Migrated {len(data)} projects from JSON to SQLite")

    except Exception as e:
        logger.error(f"Migration failed: {e}")


class ProjectStore:
    """Project storage operations."""

    @staticmethod
    def list_all() -> List[Project]:
        """List all projects, sorted by last accessed."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM projects
                ORDER BY COALESCE(last_accessed, created_at) DESC
            """)
            rows = cursor.fetchall()

            projects = []
            for row in rows:
                project = ProjectStore._row_to_project(row, cursor)
                projects.append(project)

            return projects

    @staticmethod
    def get(project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return ProjectStore._row_to_project(row, cursor)

    @staticmethod
    def create(project: Project) -> Project:
        """Create a new project."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (id, name, description, created_at, status, task_type, embedding_model, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project.id,
                project.name,
                project.description,
                project.created_at.isoformat(),
                project.status.value,
                project.task_type,
                project.embedding_model,
                project.last_accessed.isoformat() if project.last_accessed else None
            ))

            return project

    @staticmethod
    def update(project: Project) -> Project:
        """Update an existing project."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects SET
                    name = ?,
                    description = ?,
                    status = ?,
                    task_type = ?,
                    embedding_model = ?,
                    last_accessed = ?
                WHERE id = ?
            """, (
                project.name,
                project.description,
                project.status.value,
                project.task_type,
                project.embedding_model,
                project.last_accessed.isoformat() if project.last_accessed else None,
                project.id
            ))

            return project

    @staticmethod
    def delete(project_id: str) -> bool:
        """Delete a project and all related data."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cursor.rowcount > 0

    @staticmethod
    def touch(project_id: str):
        """Update last_accessed timestamp."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects SET last_accessed = ? WHERE id = ?
            """, (datetime.now().isoformat(), project_id))

    @staticmethod
    def add_indexed_model(project_id: str, model_info: IndexedModelInfo):
        """Add or update an indexed model for a project."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO indexed_models (project_id, model_name, collection_name, indexed_at, point_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, model_name) DO UPDATE SET
                    collection_name = excluded.collection_name,
                    indexed_at = excluded.indexed_at,
                    point_count = excluded.point_count
            """, (
                project_id,
                model_info.model_name,
                model_info.collection_name,
                model_info.indexed_at.isoformat(),
                model_info.point_count
            ))

    @staticmethod
    def _row_to_project(row: sqlite3.Row, cursor: sqlite3.Cursor) -> Project:
        """Convert a database row to a Project object."""
        # Get indexed models
        cursor.execute("""
            SELECT model_name, collection_name, indexed_at, point_count
            FROM indexed_models WHERE project_id = ?
        """, (row['id'],))

        indexed_models = [
            IndexedModelInfo(
                model_name=m['model_name'],
                collection_name=m['collection_name'],
                indexed_at=datetime.fromisoformat(m['indexed_at']),
                point_count=m['point_count']
            )
            for m in cursor.fetchall()
        ]

        return Project(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            created_at=datetime.fromisoformat(row['created_at']),
            status=ProjectStatus(row['status']),
            task_type=row['task_type'],
            embedding_model=row['embedding_model'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            indexed_models=indexed_models
        )


class ExperimentStore:
    """Experiment storage operations."""

    @staticmethod
    def list_by_project(project_id: str) -> List[Experiment]:
        """List all experiments for a project."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))

            return [ExperimentStore._row_to_experiment(row) for row in cursor.fetchall()]

    @staticmethod
    def get(experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            return ExperimentStore._row_to_experiment(row) if row else None

    @staticmethod
    def create(experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    id, project_id, name, status, created_at, embedding_model,
                    chunk_size, search_strategy, progress_current, progress_total,
                    progress_message, started_at, updated_at, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                experiment.project_id,
                experiment.name,
                experiment.status,
                experiment.created_at.isoformat(),
                experiment.embedding_model,
                experiment.chunk_size,
                experiment.search_strategy,
                experiment.progress_current,
                experiment.progress_total,
                experiment.progress_message,
                experiment.started_at.isoformat() if experiment.started_at else None,
                experiment.updated_at.isoformat() if experiment.updated_at else None,
                json.dumps(experiment.metrics) if experiment.metrics else None
            ))
            return experiment

    @staticmethod
    def update(experiment: Experiment) -> Experiment:
        """Update an experiment."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments SET
                    status = ?,
                    progress_current = ?,
                    progress_total = ?,
                    progress_message = ?,
                    started_at = ?,
                    updated_at = ?,
                    metrics = ?
                WHERE id = ?
            """, (
                experiment.status,
                experiment.progress_current,
                experiment.progress_total,
                experiment.progress_message,
                experiment.started_at.isoformat() if experiment.started_at else None,
                experiment.updated_at.isoformat() if experiment.updated_at else None,
                json.dumps(experiment.metrics) if experiment.metrics else None,
                experiment.id
            ))
            return experiment

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        """Convert a database row to an Experiment object."""
        return Experiment(
            id=row['id'],
            project_id=row['project_id'],
            name=row['name'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']),
            embedding_model=row['embedding_model'],
            chunk_size=row['chunk_size'],
            search_strategy=row['search_strategy'],
            progress_current=row['progress_current'],
            progress_total=row['progress_total'],
            progress_message=row['progress_message'],
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
            metrics=json.loads(row['metrics']) if row['metrics'] else None
        )


# Initialize database on module load
init_db()
migrate_from_json()
