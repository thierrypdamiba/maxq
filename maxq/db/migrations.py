"""Database migrations and schema setup."""

import sqlite3
from pathlib import Path

from maxq.core.config import settings


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    # Ensure app directory exists for database
    db_path = Path(settings.db_path)
    if not db_path.is_absolute():
        db_path = settings.app_dir / settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def run_migrations() -> None:
    """Run all database migrations."""
    conn = get_connection()
    cursor = conn.cursor()

    # ========================================
    # New tables from maxq3 (runs + jobs)
    # ========================================

    # Create runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            collection TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Create jobs table (generic job queue)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            payload_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (run_id)
        )
    """)

    # Create indexes for job queue queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_run_id ON jobs (run_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status)
    """)

    # ========================================
    # Existing tables from maxq (projects etc)
    # ========================================

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

    # Indexing jobs table (legacy, keeping for backward compat)
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_jobs_project ON indexing_jobs(project_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_jobs_status ON indexing_jobs(status)")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    run_migrations()
    print("Migrations complete")
