"""MaxQ CLI onboarding - GUI/CLI wizard with non-interactive support."""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt


console = Console()


def run_init(
    studio: bool = False,
    cli: bool = False,
    qdrant_url: str | None = None,
    api_key: str | None = None,
    collection: str | None = None,
    docker: bool = False,
) -> bool:
    """
    Initialize MaxQ.

    Non-interactive flags for AI/automation:
        --studio: Launch GUI directly
        --cli: Run CLI wizard
        --qdrant-url: Qdrant URL (skips connection prompts)
        --collection: Collection name (skips collection prompts)
        --docker: Start Docker Qdrant automatically

    Returns True on success.
    """
    # Non-interactive: --studio launches GUI immediately
    if studio:
        return _launch_studio()

    # Non-interactive: all flags provided
    if qdrant_url and collection:
        if api_key:
            _save_env(QDRANT_URL=qdrant_url, QDRANT_API_KEY=api_key)
        else:
            _save_env(QDRANT_URL=qdrant_url)
        _create_config(collection)
        console.print(f"[green]✓[/green] Configured: {collection} @ {qdrant_url}")
        console.print("\nRun: [cyan]maxq test[/cyan]")
        return True

    # Non-interactive: --docker + --collection
    if docker and collection:
        url = _setup_docker_auto()
        if url:
            _create_config(collection)
            console.print(f"[green]✓[/green] Configured: {collection} @ {url}")
            console.print("\nRun: [cyan]maxq test[/cyan]")
            return True
        return False

    # Non-interactive: --cli skips the GUI question
    if cli:
        return _run_cli_wizard(qdrant_url, collection, docker)

    # Interactive: Ask GUI or CLI
    console.print()
    console.print("  1) Studio (GUI)")
    console.print("  2) CLI")
    console.print()

    choice = Prompt.ask("", default="1", choices=["1", "2"], show_choices=False)

    if choice == "1":
        return _launch_studio()
    else:
        return _run_cli_wizard(qdrant_url, collection, docker)


def _launch_studio() -> bool:
    """Launch MaxQ Studio."""
    console.print("[dim]Launching Studio...[/dim]")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "maxq.cli", "studio"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print("[green]✓[/green] Studio launching at http://localhost:3333")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to launch: {e}")
        console.print("Run manually: [cyan]maxq studio[/cyan]")
        return False


def _run_cli_wizard(
    qdrant_url: str | None = None,
    collection: str | None = None,
    docker: bool = False,
) -> bool:
    """Run the CLI wizard."""
    console.print()

    # Step 1: Connect to Qdrant
    if not qdrant_url:
        qdrant_url = os.getenv("QDRANT_URL")

    if not qdrant_url:
        if docker:
            qdrant_url = _setup_docker_auto()
        else:
            qdrant_url = _connect_qdrant()
        if not qdrant_url:
            return False
    else:
        if _test_connection(qdrant_url, os.getenv("QDRANT_API_KEY")):
            console.print(f"[green]✓[/green] Connected to Qdrant")
        else:
            console.print(f"[yellow]![/yellow] Could not verify connection to {qdrant_url}")

    # Step 2: Get or select collection
    if not collection:
        collection = _get_collection(qdrant_url)
        if not collection:
            return False

    # Step 3: Create config
    _create_config(collection)

    # Step 4: Next steps
    console.print()
    console.print("Run your first eval:")
    console.print("  [cyan]maxq test[/cyan]")
    console.print()
    return True


def _connect_qdrant() -> str | None:
    """Connect to Qdrant. Returns URL if successful."""
    console.print("Where is your Qdrant?")
    console.print()
    console.print("  1) Cloud")
    console.print("  2) Docker")
    console.print("  3) Not sure")
    console.print()

    choice = Prompt.ask("", default="1", choices=["1", "2", "3"], show_choices=False)

    if choice == "1":
        return _setup_cloud()
    elif choice == "2":
        return _setup_docker()
    else:
        _show_help()
        return None


def _setup_cloud() -> str | None:
    """Cloud setup. Returns URL if successful."""
    console.print()
    console.print("[dim]Get credentials from cloud.qdrant.io[/dim]")
    console.print()

    url = Prompt.ask("Qdrant URL")
    if not url:
        return None

    api_key = Prompt.ask("API Key", password=True)

    if _test_connection(url, api_key):
        _save_env(QDRANT_URL=url, QDRANT_API_KEY=api_key)
        console.print("[green]✓[/green] Connected")
        return url
    else:
        console.print("[red]✗[/red] Could not connect")
        return None


def _setup_docker() -> str | None:
    """Docker setup (interactive). Returns URL if successful."""
    console.print()

    start = Prompt.ask("Start Qdrant container?", choices=["y", "n"], default="y")

    if start == "y":
        return _setup_docker_auto()
    else:
        console.print()
        console.print("Start Qdrant manually:")
        console.print("  docker run -p 6333:6333 qdrant/qdrant")
        console.print()
        console.print("Then: maxq init --cli")
        return None


def _setup_docker_auto() -> str | None:
    """Start Docker Qdrant automatically (non-interactive)."""
    try:
        result = subprocess.run(["docker", "start", "maxq-qdrant"], capture_output=True)
        if result.returncode != 0:
            subprocess.run([
                "docker", "run", "-d",
                "--name", "maxq-qdrant",
                "-p", "6333:6333",
                "qdrant/qdrant"
            ], capture_output=True, check=True)

        url = "http://localhost:6333"
        _save_env(QDRANT_URL=url)
        console.print("[green]✓[/green] Qdrant running")
        return url

    except FileNotFoundError:
        console.print("[red]✗[/red] Docker not found")
        console.print("[dim]Install from docker.com[/dim]")
        return None
    except subprocess.CalledProcessError:
        console.print("[red]✗[/red] Failed to start")
        return None


def _get_collection(qdrant_url: str) -> str | None:
    """Get or create collection. Returns collection name."""
    console.print()
    console.print("  1) Import data")
    console.print("  2) Use existing collection")
    console.print()

    choice = Prompt.ask("", default="1", choices=["1", "2"], show_choices=False)

    if choice == "1":
        return _import_data()
    else:
        collection = Prompt.ask("Collection name")
        if collection:
            console.print(f"[green]✓[/green] Using {collection}")
            return collection
        return None


def _import_data() -> str | None:
    """Import data from file. Returns collection name."""
    console.print()
    file_path = Prompt.ask("File path")

    if not file_path or not Path(file_path).exists():
        console.print("[red]✗[/red] File not found")
        return None

    name = Path(file_path).stem
    collection = Prompt.ask("Collection name", default=name)

    console.print("[dim]Indexing...[/dim]")

    try:
        result = subprocess.run(
            ["maxq", "import", file_path, "--collection", collection],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            console.print(f"[green]✓[/green] Indexed to {collection}")
            return collection
        else:
            console.print("[red]✗[/red] Import failed")
            console.print(result.stderr[:200] if result.stderr else "")
            return None
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        return None


def _create_config(collection: str) -> None:
    """Create maxq.yaml config."""
    config_path = Path.cwd() / "maxq.yaml"

    if config_path.exists():
        console.print("[dim]maxq.yaml already exists[/dim]")
        return

    config = f'''# MaxQ evaluation config
collection: {collection}

tests:
  - query: "example search query"
    assertions:
      - type: not-empty
      - type: latency
        max_ms: 200

  # Add your test queries:
  # - query: "another query"
  #   ground_truth:
  #     relevant_ids: ["doc1", "doc2"]
  #   assertions:
  #     - type: recall
  #       min: 0.8
'''

    with open(config_path, "w") as f:
        f.write(config)

    console.print("[green]✓[/green] Created maxq.yaml")


def _test_connection(url: str, api_key: str = None) -> bool:
    """Test Qdrant connection."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=url, api_key=api_key if api_key else None, timeout=5)
        client.get_collections()
        return True
    except:
        return False


def _save_env(**kwargs) -> None:
    """Save to .env file."""
    env_path = Path.cwd() / ".env"

    existing = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    existing[key] = value

    existing.update({k: v for k, v in kwargs.items() if v})

    with open(env_path, "w") as f:
        for key, value in existing.items():
            f.write(f"{key}={value}\n")


def _show_help() -> None:
    """Help for users who aren't sure."""
    console.print()
    console.print("Qdrant is a vector database. You need one to use MaxQ.")
    console.print()
    console.print("[bold]Cloud[/bold] - Easiest. Free tier at cloud.qdrant.io")
    console.print("[bold]Docker[/bold] - Local: docker run -p 6333:6333 qdrant/qdrant")
    console.print()
    console.print("Then: maxq init")


# Backwards compat
run_smart_onboarding = run_init
run_onboarding = run_init
