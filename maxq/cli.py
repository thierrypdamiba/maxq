import os
import sys
from pathlib import Path
from datetime import datetime
import json
import typer
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.live import Live
from rich.text import Text

# Import logic
from maxq.search_engine import MaxQEngine, CollectionStrategy, SearchRequest
from maxq.autoconfig import (
    PRESETS,
    CLOUD_DENSE_MODELS,
    get_preset,
    list_presets,
    print_config_summary,
)

# Backwards compatibility
SearchEngine = MaxQEngine
from maxq.data_sources import (
    DataSourceManager,
    DataSourceConfig,
    DataSourceType,
    MultiDatasetCollection,
    ConflictDetector,
)

# Load environment variables
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    # Fallback: Try looking in parent directory explicitly
    parent_env = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        env_path = parent_env

console = Console()

if env_path:
    console.print(f"[dim]Loaded configuration from: {env_path}[/dim]")
else:
    console.print(f"[dim]No .env file found (checked current and parent directories)[/dim]")


# ============================================
# Interactive Mode Detection & Config
# ============================================


def is_interactive() -> bool:
    """Check if we're in an interactive terminal context."""
    # Not interactive if stdin/stdout aren't TTYs
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    # Not interactive in CI environments
    if os.getenv("CI", "").lower() in ("true", "1", "yes"):
        return False
    # Not interactive with dumb terminal
    if os.getenv("TERM") == "dumb":
        return False
    return True


def get_default_mode() -> str:
    """Get default mode from env or config file.

    Returns: 'studio', 'cli', or 'prompt'
    """
    # Check env var first
    env_default = os.getenv("MAXQ_DEFAULT", "").lower()
    if env_default in ("studio", "cli", "prompt"):
        return env_default

    # Check config file
    config_path = Path.home() / ".maxq" / "config.toml"
    if config_path.exists():
        try:
            import tomllib

            with open(config_path, "rb") as f:
                config = tomllib.load(f)
                mode = config.get("default_mode", "prompt").lower()
                if mode in ("studio", "cli", "prompt"):
                    return mode
        except Exception:
            pass

    return "prompt"


def show_interactive_menu() -> str:
    """Show the interactive startup menu and return choice."""
    console.print()
    console.print("[bold]MaxQ[/bold]")
    console.print()
    console.print("  [cyan]1)[/cyan] Open Studio [dim](recommended)[/dim]")
    console.print("  [cyan]2)[/cyan] Use CLI")
    console.print("  [cyan]3)[/cyan] Help [dim](see all commands)[/dim]")
    console.print()

    return Prompt.ask("Select", default="1", choices=["1", "2", "3"])


def run_setup_wizard() -> bool:
    """Run first-time setup wizard. Returns True if setup completed."""
    console.print()
    console.print("[bold]Welcome to MaxQ[/bold]")
    console.print("[dim]Let's get you set up.[/dim]")
    console.print()

    # Step 1: How do you want to use MaxQ?
    console.print("[bold]How do you want to use MaxQ?[/bold]")
    console.print()
    console.print("  [cyan]1)[/cyan] Studio [dim](web UI - recommended for beginners)[/dim]")
    console.print("  [cyan]2)[/cyan] CLI [dim](command line - for power users)[/dim]")
    console.print()
    interface = Prompt.ask("Select", default="1", choices=["1", "2"])

    # Step 2: Where do you want to run Qdrant?
    console.print()
    console.print("[bold]Where do you want to run Qdrant?[/bold]")
    console.print("[dim]You can change this later.[/dim]")
    console.print()
    console.print(
        "  [cyan]1)[/cyan] Cloud [dim](Qdrant Cloud - recommended, free tier available)[/dim]"
    )
    console.print("  [cyan]2)[/cyan] Docker [dim](local container)[/dim]")
    console.print("  [cyan]3)[/cyan] Local [dim](run Qdrant directly)[/dim]")
    console.print()
    backend = Prompt.ask("Select", default="1", choices=["1", "2", "3"])

    # Step 3: Set up based on choice
    if backend == "1":
        setup_cloud()
    elif backend == "2":
        setup_docker()
    else:
        setup_local()

    # Save preferences
    save_config(interface="studio" if interface == "1" else "cli")

    # Return interface choice
    return interface == "1"  # True = studio, False = cli


def setup_cloud():
    """Set up Qdrant Cloud credentials."""
    console.print()
    console.print("[bold]Qdrant Cloud Setup[/bold]")
    console.print()
    console.print("[dim]1. Sign up at https://cloud.qdrant.io (free tier available)[/dim]")
    console.print("[dim]2. Create a cluster and get your URL + API key[/dim]")
    console.print()

    url = Prompt.ask("Cluster URL", default=os.getenv("QDRANT_URL", ""))
    api_key = Prompt.ask("API Key", password=True, default="")

    if url and api_key:
        # Test connection
        console.print("[dim]Testing connection...[/dim]")
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=url, api_key=api_key, timeout=10)
            client.get_collections()
            console.print("[green]✓ Connected to Qdrant Cloud[/green]")

            # Save credentials
            save_credentials(url, api_key)
        except Exception as e:
            console.print(f"[red]✗ Connection failed: {e}[/red]")
            console.print("[dim]You can fix this later by editing ~/.maxq/.env[/dim]")
    else:
        console.print("[yellow]Skipped - you can add credentials later[/yellow]")


def setup_docker():
    """Set up local Docker-based Qdrant."""
    console.print()
    console.print("[bold]Docker Setup[/bold]")
    console.print()

    # Check if Docker is available
    import subprocess

    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]✓ Docker found: {result.stdout.strip()}[/green]")
        else:
            console.print("[red]✗ Docker not found[/red]")
            console.print("[dim]Install Docker: https://docs.docker.com/get-docker/[/dim]")
            return
    except FileNotFoundError:
        console.print("[red]✗ Docker not found[/red]")
        console.print("[dim]Install Docker: https://docs.docker.com/get-docker/[/dim]")
        return

    # Offer to start Qdrant
    start = Prompt.ask("Start Qdrant container now?", choices=["y", "n"], default="y")
    if start == "y":
        console.print("[dim]Starting Qdrant...[/dim]")
        try:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "qdrant",
                    "-p",
                    "6333:6333",
                    "-p",
                    "6334:6334",
                    "-v",
                    "qdrant_storage:/qdrant/storage",
                    "qdrant/qdrant",
                ],
                check=True,
                capture_output=True,
            )
            console.print("[green]✓ Qdrant running at http://localhost:6333[/green]")
            save_credentials("http://localhost:6333", "")
        except subprocess.CalledProcessError as e:
            if "already in use" in str(e.stderr):
                console.print("[yellow]Container 'qdrant' already exists[/yellow]")
                subprocess.run(["docker", "start", "qdrant"], capture_output=True)
                console.print("[green]✓ Started existing container[/green]")
                save_credentials("http://localhost:6333", "")
            else:
                console.print(f"[red]✗ Failed to start: {e.stderr}[/red]")


def setup_local():
    """Set up local Qdrant (assumes user will run it themselves)."""
    console.print()
    console.print("[bold]Local Setup[/bold]")
    console.print()
    console.print("[dim]MaxQ will connect to Qdrant at http://localhost:6333[/dim]")
    console.print()
    console.print("To run Qdrant locally:")
    console.print("  [cyan]# Download and run[/cyan]")
    console.print(
        "  curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz"
    )
    console.print("  ./qdrant")
    console.print()

    save_credentials("http://localhost:6333", "")
    console.print("[green]✓ Configured for local Qdrant[/green]")


def save_credentials(url: str, api_key: str):
    """Save Qdrant credentials to ~/.maxq/.env"""
    config_dir = Path.home() / ".maxq"
    config_dir.mkdir(exist_ok=True)
    env_file = config_dir / ".env"

    lines = []
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if not line.startswith("QDRANT_URL=") and not line.startswith("QDRANT_API_KEY="):
                    lines.append(line)

    lines.append(f"QDRANT_URL={url}\n")
    if api_key:
        lines.append(f"QDRANT_API_KEY={api_key}\n")

    with open(env_file, "w") as f:
        f.writelines(lines)

    console.print(f"[dim]Saved to {env_file}[/dim]")


def save_config(interface: str):
    """Save user preferences to config file."""
    config_dir = Path.home() / ".maxq"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.toml"

    with open(config_file, "w") as f:
        f.write(f'default_mode = "{interface}"\n')

    console.print(f"[dim]Preferences saved to {config_file}[/dim]")


def needs_setup() -> bool:
    """Check if first-time setup is needed."""
    config_file = Path.home() / ".maxq" / "config.toml"
    env_file = Path.home() / ".maxq" / ".env"
    # Need setup if no config file exists
    return not config_file.exists()


def show_cli_quickstart() -> None:
    """Show CLI quickstart and common commands."""
    console.print()
    console.print("[bold]MaxQ CLI[/bold]")
    console.print()
    console.print("[dim]Getting started:[/dim]")
    console.print("  maxq demo              [dim]# Load sample data[/dim]")
    console.print('  maxq search "query"    [dim]# Search indexed data[/dim]')
    console.print()
    console.print("[dim]Common commands:[/dim]")
    console.print("  maxq doctor            [dim]# Health check[/dim]")
    console.print("  maxq import            [dim]# Import a dataset[/dim]")
    console.print("  maxq studio            [dim]# Launch web UI[/dim]")
    console.print("  maxq run               [dim]# Run index pipeline[/dim]")
    console.print("  maxq worker            [dim]# Start background worker[/dim]")
    console.print()
    console.print("[dim]For all commands:[/dim]")
    console.print("  maxq --help")
    console.print()


app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    studio_flag: bool = typer.Option(False, "--studio", help="Launch Studio immediately"),
    cli_flag: bool = typer.Option(False, "--cli", help="Show CLI help immediately"),
):
    """
    MaxQ - Vector search tuning and evaluation platform.

    Run without arguments for interactive mode, or use a subcommand.
    """
    # If a subcommand is being invoked, let it run
    if ctx.invoked_subcommand is not None:
        return

    # Handle explicit flags
    if studio_flag:
        _launch_studio()
        raise typer.Exit()

    if cli_flag:
        show_cli_quickstart()
        raise typer.Exit()

    # No subcommand - show help
    console.print(ctx.get_help())
    raise typer.Exit(0)


def _launch_studio():
    """Helper to launch studio without circular import."""
    import subprocess
    import webbrowser
    import time

    console.print("[bold]Starting MaxQ Studio...[/bold]")

    # Start API server
    api_process = subprocess.Popen(
        ["uvicorn", "maxq.server.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Start frontend
    frontend_dir = Path(__file__).parent / "studio-ui"
    if frontend_dir.exists():
        subprocess.Popen(
            ["pnpm", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    time.sleep(2)

    if is_interactive():
        webbrowser.open("http://localhost:3000")

    console.print("[green]Studio running at http://localhost:3000[/green]")
    console.print("[dim]API at http://localhost:8000/docs[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        api_process.wait()
    except KeyboardInterrupt:
        api_process.terminate()


import readchar
from readchar import key


def input_with_placeholder(prefix, placeholder):
    """
    Simulates an input with a placeholder that disappears when typing starts.
    """
    # Print the prompt prefix
    console.print(prefix, end="")

    # Print placeholder
    console.print(f"[dim]{placeholder}[/dim]", end="")

    # Move cursor back to start of placeholder
    back_len = len(placeholder)
    sys.stdout.write("\b" * back_len)
    sys.stdout.flush()

    buffer = []
    typing_started = False

    while True:
        k = readchar.readkey()

        if k == key.ENTER:
            sys.stdout.write("\n")
            return "".join(buffer)

        elif k == key.CTRL_C:
            raise KeyboardInterrupt()

        elif k == key.BACKSPACE or k == "\x7f":
            if buffer:
                buffer.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()

            if not buffer and typing_started:
                typing_started = False
                console.print(f"[dim]{placeholder}[/dim]", end="")
                sys.stdout.write("\b" * back_len)
                sys.stdout.flush()

        elif len(k) == 1 and not isinstance(k, bytes):
            if ord(k) >= 32 or k == "\t":
                if not typing_started:
                    sys.stdout.write(" " * back_len)
                    sys.stdout.write("\b" * back_len)
                    typing_started = True

                buffer.append(k)
                console.print(k, end="")
                sys.stdout.flush()


def interactive_select(options, default_index=0):
    """
    Interactive selection menu using arrow keys.
    Returns the selected option string.
    """
    selected_index = default_index
    line_color = "cyan"

    def generate_menu():
        text = Text()
        for i, option in enumerate(options):
            bullet = "●" if i == selected_index else "○"
            # Highlight selected
            if i == selected_index:
                style = "bold white"
            else:
                style = "dim white"

            # Draw the connected line prefix
            text.append("│", style=line_color)
            text.append(f"  {bullet} {option}\n", style=style)

        return text

    # Hide cursor
    console.show_cursor(False)
    try:
        with Live(generate_menu(), console=console, auto_refresh=False, transient=True) as live:
            while True:
                live.update(generate_menu())
                live.refresh()

                # Read keypress using readchar (more robust)
                k = readchar.readkey()

                if k == key.ENTER:
                    break
                elif k == key.CTRL_C:
                    raise KeyboardInterrupt()
                elif k == key.UP:
                    selected_index = (selected_index - 1) % len(options)
                elif k == key.DOWN:
                    selected_index = (selected_index + 1) % len(options)
    finally:
        console.show_cursor(True)

    # Re-print the final selection statically so it stays in the log
    for i, option in enumerate(options):
        bullet = "●" if i == selected_index else "○"
        if i == selected_index:
            console.print(f"[{line_color}]│[/]  [bold white]{bullet} {option}[/bold white]")
        else:
            console.print(f"[{line_color}]│[/]  [dim white]{bullet} {option}[/dim white]")

    return options[selected_index]


def normalize_dataset_name(dataset_input: str) -> str:
    """
    Normalize dataset input to extract the dataset ID.
    Accepts both:
    - Short form: "fka/awesome-chatgpt-prompts"
    - Full URL: "https://huggingface.co/datasets/fka/awesome-chatgpt-prompts"
    """
    if "huggingface.co/datasets/" in dataset_input:
        # Extract from URL
        import re

        match = re.search(
            r"huggingface\.co/datasets/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)", dataset_input
        )
        if match:
            return match.group(1)

    # Return as-is (assume it's already in correct format)
    return dataset_input.strip()


def print_header():
    console.print()
    console.print(
        Panel(
            "[bold white]MAXQ[/bold white]   VECTOR SEARCH STUDIO",
            style="bold white on orange3",
            expand=False,
        )
    )


def select_data_source(console, line_color, text_color):
    """Interactive wizard for data selection."""
    import os
    from .utils import normalize_dataset_name  # Assuming this exists or is imported

    # Check for env var
    env_dataset = os.getenv("MAXQ_DATASET")

    dataset_name = None
    folder_path = None

    if env_dataset:
        console.print(
            f"[{line_color}]◆[/]  [green]✓ Detected dataset in environment: {env_dataset}[/green]"
        )
        dataset_name = env_dataset
    else:
        console.print(f"[{line_color}]◆[/]  [{text_color}]Where is your data located?[/]")

        data_source_options = ["Online (Hugging Face, API, URL)", "Local (Files on your computer)"]
        data_source = interactive_select(data_source_options, default_index=0)

        if "Online" in data_source:
            console.print(f"[{line_color}]│[/]")
            console.print(
                f"[{line_color}]◆[/]  [{text_color}]How do you want to find your dataset?[/]"
            )

            find_options = [
                "I have a link (Hugging Face dataset ID)",
                "Find it for me (Natural language search)",
            ]
            find_choice = interactive_select(find_options, default_index=0)

            if "Find it for me" in find_choice:
                console.print(f"[{line_color}]│[/]")
                search_query = Prompt.ask(
                    f"[{line_color}]│[/]  [dim]Describe the dataset you need[/dim]"
                )

                with console.status("[bold green]Searching for datasets...[/bold green]"):
                    try:
                        from linkup import LinkupClient
                        from maxq.config import LINKUP_API_KEY

                        client = LinkupClient(api_key=LINKUP_API_KEY)

                        response = client.search(
                            query=f"Hugging Face dataset for: {search_query}",
                            depth="standard",
                            output_type="searchResults",
                        )

                        results = response.results if hasattr(response, "results") else []
                        candidates = []
                        for r in results:
                            if "huggingface.co/datasets/" in r.url:
                                name = r.url.split("datasets/")[-1].strip("/")
                                if name not in candidates:
                                    candidates.append(name)

                        if candidates:
                            console.print(
                                f"[{line_color}]│[/]  [green]Found {len(candidates)} candidates:[/green]"
                            )
                            candidates = candidates[:5]
                            candidates.append("None of these (Enter manually)")

                            selection = interactive_select(candidates, default_index=0)

                            if "None of these" in selection:
                                dataset_name = Prompt.ask(
                                    f"[{line_color}]│[/]  [dim]Dataset ID or URL[/dim]"
                                )
                            else:
                                dataset_name = selection
                        else:
                            console.print(
                                f"[{line_color}]│[/]  [yellow]No direct matches found.[/yellow]"
                            )
                            dataset_name = Prompt.ask(
                                f"[{line_color}]│[/]  [dim]Dataset ID or URL[/dim]"
                            )

                    except Exception as e:
                        console.print(f"[{line_color}]│[/]  [red]Search failed: {e}[/red]")
                        dataset_name = Prompt.ask(
                            f"[{line_color}]│[/]  [dim]Dataset ID or URL[/dim]"
                        )

            else:
                console.print(f"[{line_color}]│[/]")
                dataset_input = Prompt.ask(
                    f"[{line_color}]│[/]  [dim]Dataset ID or URL[/dim]",
                    default="fka/awesome-chatgpt-prompts",
                )
                dataset_name = normalize_dataset_name(dataset_input)

        else:
            # Local
            console.print(f"[{line_color}]│[/]")
            folder_path = Prompt.ask(
                f"[{line_color}]│[/]  [dim]Path to folder[/dim]", default="./src"
            )

    return dataset_name, folder_path


def project_setup_wizard():
    """
    Interactive wizard to configure the project.
    Returns a dictionary with configuration.
    """
    # Visual styling
    line_color = "cyan"
    label_style = "bold #4a1c1c on #d8d3c5"
    text_color = "white"

    console.print()
    console.print(f"[{line_color}]┌[/]  [{label_style}] MaxQ Create [/]")

    # ==============================================
    # STEP 0: PROJECT NAME
    # ==============================================
    console.print(f"[{line_color}]│[/]")
    console.print(f"[{line_color}]◆[/]  [{text_color}]What do you want to name your project?[/]")
    project_name = ""
    while not project_name.strip():
        project_name = input_with_placeholder(f"[{line_color}]│[/]  ", "my-maxq-app")

    # ==============================================
    # STEP 1: QDRANT CONFIGURATION
    # ==============================================
    console.print(f"[{line_color}]│[/]")

    import os

    env_q_url = os.getenv("QDRANT_URL")
    env_q_key = os.getenv("QDRANT_API_KEY")

    if env_q_url and env_q_key:
        console.print(
            f"[{line_color}]◆[/]  [green]✓ Detected Qdrant Cloud credentials in environment[/green]"
        )
        qdrant_deploy = "Cloud"
        q_url = env_q_url
        q_key = env_q_key
        use_qci = False

        # Save to ~/.maxq/.env for Studio access
        try:
            from pathlib import Path

            maxq_dir = Path.home() / ".maxq"
            maxq_dir.mkdir(exist_ok=True)
            maxq_env = maxq_dir / ".env"
            env_content = f"QDRANT_URL={q_url}\nQDRANT_API_KEY={q_key}\n"
            if os.getenv("OPENAI_API_KEY"):
                env_content += f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}\n"
            maxq_env.write_text(env_content)
        except:
            pass
    else:
        console.print(f"[{line_color}]◆[/]  [{text_color}]How would you like to use Qdrant?[/]")
        qdrant_options = ["Cloud (recommended)", "Local", "Docker"]
        qdrant_deploy = interactive_select(qdrant_options, default_index=0).split(" (")[0]

        q_url = ":memory:"
        q_key = None
        use_qci = False

        if "Docker" in qdrant_deploy:
            console.print(f"[{line_color}]│[/]")
            q_url = Prompt.ask(
                f"[{line_color}]│[/]  [dim]Qdrant URL[/dim]", default="http://localhost:6333"
            )

        elif "Cloud" in qdrant_deploy:
            console.print(f"[{line_color}]│[/]")
            console.print(
                f"[{line_color}]◆[/]  [{text_color}]Enter your Qdrant Cloud credentials?[/]"
            )
            if "Enter Credentials" in interactive_select(
                ["Skip for now (default)", "Enter Credentials"], default_index=0
            ):
                console.print(f"[{line_color}]│[/]")
                q_url = Prompt.ask(f"[{line_color}]│[/]  [dim]Qdrant Cloud URL[/dim]")
                q_key = Prompt.ask(f"[{line_color}]│[/]  [dim]Qdrant API Key[/dim]", password=True)

    # ==============================================
    # STEP 2: DATA
    # ==============================================
    console.print(f"[{line_color}]│[/]")
    dataset_name, folder_path = select_data_source(console, line_color, text_color)

    # Determine target collection name
    target_collection_name = project_name
    if dataset_name:
        # Use dataset name (e.g. "fka/awesome-chatgpt-prompts" -> "awesome-chatgpt-prompts")
        # We'll use the last part of the path to be clean, but unique enough
        if "/" in dataset_name:
            target_collection_name = dataset_name.split("/")[-1]
        else:
            target_collection_name = dataset_name

        # Sanitize slightly (replace spaces with dashes, though HF names usually clean)
        target_collection_name = target_collection_name.replace(" ", "-")

    # ==============================================
    # STEP 1.5: CHECK EXISTING COLLECTION
    # ==============================================
    skip_data_step = False
    collection_exists = False

    if q_url and q_url != ":memory:":
        try:
            from qdrant_client import QdrantClient

            temp_client = QdrantClient(url=q_url, api_key=q_key)
            if temp_client.collection_exists(target_collection_name):
                collection_exists = True
                console.print(f"[{line_color}]│[/]")
                console.print(
                    f"[{line_color}]◆[/]  [yellow]Found existing collection: '{target_collection_name}'[/yellow]"
                )
                if (
                    Prompt.ask(
                        f"[{line_color}]│[/]  [cyan]Use existing collection (Skip data setup)?[/cyan]",
                        choices=["y", "n"],
                        default="y",
                    )
                    == "y"
                ):
                    skip_data_step = True
        except:
            pass

    # ==============================================
    # STEP 3: PERFORMANCE PROFILE
    # ==============================================
    console.print(f"[{line_color}]│[/]")
    console.print(f"[{line_color}]◆[/]  [{text_color}]Select a Performance Profile:[/]")

    profiles = [
        "Fast (Speed focused, smaller models)",
        "Balanced (Recommended, best trade-off)",
        "Accurate (Quality focused, larger models)",
        "Auto-Select (AI analyzes your data)",
        "Custom Configuration (Advanced menu)",
    ]
    profile_choice = interactive_select(profiles, default_index=1)

    # Default settings
    location = "src/"
    provider = "OpenAI"
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Qdrant Cloud Inference supported
    search_strategy = "Hybrid (Dense + Sparse)"
    quantization = "Int8"
    max_documents = 1000
    ide_choice = "Skip for now"

    # Handle Auto-Select using presets
    selected_preset = None
    if "Auto-Select" in profile_choice:
        console.print(f"[{line_color}]│[/]")
        console.print(f"[{line_color}]│[/]  [bold white]SELECT PRESET[/bold white]")
        console.print(f"[{line_color}]│[/]")
        console.print(f"[{line_color}]◆[/]  [{text_color}]What's your priority?[/]")
        priority_options = [
            "Fast (fastest response)",
            "Balanced (best trade-off)",
            "Accurate (highest quality)",
        ]
        priority_choice = interactive_select(priority_options, default_index=1)

        preset_name = (
            "fast"
            if "Fast" in priority_choice
            else ("accurate" if "Accurate" in priority_choice else "balanced")
        )
        selected_preset = PRESETS[preset_name]
        embedding_model = selected_preset.dense_model
        quantization = "Int8"

        console.print(f"[{line_color}]│[/]")
        console.print(f"[{line_color}]│[/]  [green]✓ Selected: [bold]{preset_name}[/bold][/green]")
        console.print(f"[{line_color}]│[/]  [dim]Model: {embedding_model}[/dim]")

    if "Custom" in profile_choice:
        console.print(f"[{line_color}]│[/]")
        console.print(f"[{line_color}]│[/]  [bold white]ADVANCED CONFIGURATION MENU[/bold white]")

        advanced_settings = {
            "location": "src/",
            "provider": "OpenAI",
            "api_key": api_key,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "search_strategy": "Hybrid (Dense + Sparse)",
            "quantization": "Int8",
            "max_documents": 100,
            "ide_choice": "Skip for now",
        }

        while True:
            console.print(f"[{line_color}]│[/]")
            console.print(f"[{line_color}]◆[/]  [{text_color}]Configuration Menu:[/]")
            menu_options = [
                f"1. Source File Destination ({advanced_settings['location']})",
                f"2. Language Model ({advanced_settings['provider']})",
                f"3. API Key ({'✓ Set' if advanced_settings['api_key'] else 'Not Set'})",
                f"4. Embedding Model ({advanced_settings['embedding_model'].split('/')[-1]})",
                f"5. Search Strategy ({advanced_settings['search_strategy']})",
                f"6. Quantization ({advanced_settings['quantization']})",
                f"7. Max Documents ({advanced_settings['max_documents']})",
                f"8. IDE Integration ({advanced_settings['ide_choice']})",
                "✓ Done",
            ]
            choice = interactive_select(menu_options, default_index=len(menu_options) - 1)

            if "Done" in choice:
                break
            elif "1." in choice:
                advanced_settings["location"] = Prompt.ask(
                    f"[{line_color}]│[/]  [dim]Destination[/dim]",
                    default=advanced_settings["location"],
                )
            elif "2." in choice:
                advanced_settings["provider"] = interactive_select(
                    ["OpenAI", "Anthropic", "Groq"], default_index=0
                )
            elif "3." in choice:
                advanced_settings["api_key"] = Prompt.ask(
                    f"[{line_color}]│[/]  [dim]API Key[/dim]", password=True
                )
            elif "4." in choice:
                advanced_settings["embedding_model"] = interactive_select(
                    [
                        "sentence-transformers/all-MiniLM-L6-v2 (Fast)",
                        "mixedbread-ai/mxbai-embed-large-v1 (Accurate)",
                    ],
                    default_index=0,
                ).split(" (")[0]
            elif "5." in choice:
                advanced_settings["search_strategy"] = interactive_select(
                    ["Hybrid (Dense + Sparse)", "Dense only", "Sparse only"], default_index=0
                )
            elif "6." in choice:
                advanced_settings["quantization"] = interactive_select(
                    ["Int8", "Scalar", "None"], default_index=0
                ).split(" (")[0]
            elif "7." in choice:
                advanced_settings["max_documents"] = IntPrompt.ask(
                    f"[{line_color}]│[/]  [dim]Max docs[/dim]",
                    default=advanced_settings["max_documents"],
                )
            elif "8." in choice:
                advanced_settings["ide_choice"] = interactive_select(
                    ["Skip for now", "Cursor", "VSCode"], default_index=0
                )

        location = advanced_settings["location"]
        provider = advanced_settings["provider"]
        api_key = advanced_settings["api_key"]
        max_documents = advanced_settings["max_documents"]
        ide_choice = advanced_settings["ide_choice"]
        embedding_model = advanced_settings["embedding_model"]
        search_strategy = advanced_settings["search_strategy"]
        quantization = advanced_settings["quantization"]

    elif selected_preset:
        # Use selected preset
        embedding_model = selected_preset.dense_model
        quantization = "Int8"
        max_documents = 1000
        search_strategy = "Hybrid (Dense + Sparse)"

    else:
        if "Fast" in profile_choice:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            quantization = "Scalar"
            max_documents = 100
        elif "Balanced" in profile_choice:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            quantization = "Int8"
            max_documents = 1000
        elif "Accurate" in profile_choice:
            embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
            quantization = "None"
            max_documents = 1000

        if not api_key:
            console.print(f"[{line_color}]│[/]")
            console.print(
                f"[{line_color}]◆[/]  [{text_color}]Enter your [bold]OpenAI[/bold] API Key?[/]"
            )
            if "Enter API Key" in interactive_select(
                ["Skip for now (default)", "Enter API Key"], default_index=0
            ):
                console.print(f"[{line_color}]│[/]")
                api_key = Prompt.ask(f"[{line_color}]│[/]  ", password=True, show_default=False)

    # ==============================================
    # STEP 4: VALIDATE CONFIGURATION
    # ==============================================
    if collection_exists and q_url != ":memory:":
        try:
            col_info = temp_client.get_collection(target_collection_name)
            params = col_info.config.params

            # Get remote size
            remote_size = None
            if isinstance(params.vectors, dict) and "dense" in params.vectors:
                remote_size = params.vectors["dense"].size
            elif hasattr(params.vectors, "size"):
                remote_size = params.vectors.size

            # Get local size
            local_size = 768
            if "small" in embedding_model:
                local_size = 384
            elif "large" in embedding_model:
                local_size = 1024

            if remote_size and remote_size != local_size:
                console.print(f"[{line_color}]│[/]")
                console.print(f"[{line_color}]◆[/]  [yellow]⚠️  Configuration Mismatch[/yellow]")
                console.print(
                    f"[{line_color}]│[/]  • You selected: {embedding_model} ({local_size} dims)"
                )
                console.print(f"[{line_color}]│[/]  • Existing collection: {remote_size} dims")

                console.print(f"[{line_color}]│[/]")
                console.print(f"[{line_color}]◆[/]  [{text_color}]How do you want to proceed?[/]")

                options = [
                    f"Use Collection Config (Switch to {remote_size} dims)",
                    "Overwrite Collection (Use my selection)",
                ]
                choice = interactive_select(options, default_index=0)

                if "Use Collection Config" in choice:
                    # Auto-switch model
                    if remote_size == 384:
                        embedding_model = "BAAI/bge-small-en-v1.5"
                    elif remote_size == 768:
                        embedding_model = "BAAI/bge-base-en-v1.5"
                    elif remote_size == 1024:
                        embedding_model = "BAAI/bge-large-en-v1.5"
                    console.print(
                        f"[{line_color}]│[/]  [green]✓ Switched to {embedding_model}[/green]"
                    )

                elif "Overwrite" in choice:
                    # If we skipped data, we MUST ask for it now
                    if skip_data_step:
                        console.print(
                            f"[{line_color}]│[/]  [yellow]Overwrite requires data ingestion.[/yellow]"
                        )
                        console.print(f"[{line_color}]│[/]")
                        # We already have selected data, just unskip
                        skip_data_step = False
        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")

    console.print(f"[{line_color}]└[/]")

    return {
        "name": project_name,
        "collection_name": target_collection_name,
        "location": location,
        "qdrant_deploy": qdrant_deploy,
        "q_url": q_url,
        "q_key": q_key,
        "use_qci": use_qci,
        "provider": provider,
        "api_key": api_key,
        "max_documents": max_documents,
        "ide_choice": ide_choice,
        "dataset_name": dataset_name,
        "folder_path": folder_path,
        "embedding_model": embedding_model,
        "search_strategy": search_strategy,
        "quantization": quantization,
        "skip_data_step": skip_data_step,
    }


# init command is defined below with full options


@app.command()
def configs():
    """List available cloud inference presets."""
    print_header()
    console.print("\n[bold]Available Presets (Cloud Inference)[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Preset", style="cyan")
    table.add_column("Dense Model")
    table.add_column("HNSW")
    table.add_column("Shards")
    table.add_column("Use Case", style="dim")

    for name, cfg in PRESETS.items():
        table.add_row(
            name,
            cfg.dense_model.split("/")[-1],
            f"m={cfg.hnsw.m}",
            str(cfg.shard_number),
            cfg.use_case[:30] + "..." if len(cfg.use_case) > 30 else cfg.use_case,
        )

    console.print(table)
    console.print("\n[dim]Use 'maxq config <name>' to see full details.[/dim]")


@app.command()
def config(name: str = typer.Argument(None, help="Preset name to inspect")):
    """View details of a specific preset."""
    if not name:
        configs()
        return

    if name not in PRESETS:
        console.print(f"[red]Unknown preset: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(PRESETS.keys())}[/dim]")
        return

    cfg = PRESETS[name]
    print_header()
    console.print(f"\n[bold cyan]Preset: {name}[/bold cyan]\n")
    console.print(print_config_summary(cfg))


@app.command()
def auto(
    priority: str = typer.Option(
        "balanced", "--priority", "-p", help="Priority: fast, accurate, balanced"
    ),
):
    """
    Select a preset based on your priority.
    Uses Qdrant Cloud Inference for all embeddings.
    """
    print_header()

    # Map priority to actual preset names
    # fast -> base (MiniLM, fastest)
    # balanced -> mxbai_bm25 (good balance of speed/accuracy)
    # accurate -> max_accuracy (highest quality)
    preset_map = {
        "fast": "base",
        "speed": "base",
        "accurate": "max_accuracy",
        "accuracy": "max_accuracy",
        "balanced": "mxbai_bm25",
    }
    preset_name = preset_map.get(priority.lower(), "mxbai_bm25")

    cfg = PRESETS[preset_name]

    console.print(
        Panel(
            f"[bold green]{preset_name}[/bold green]\n\n"
            f"Dense: {cfg.dense_model}\n"
            f"Sparse: {cfg.sparse_model}\n\n"
            f"[dim]{cfg.description}[/dim]",
            title="[bold]Selected Preset[/bold]",
            expand=False,
        )
    )

    console.print(f"\n[dim]Available presets: {', '.join(PRESETS.keys())}[/dim]")
    console.print("[dim]Use 'maxq start' to begin indexing.[/dim]")


@app.command()
def search_data(
    query: str = typer.Argument(..., help="Natural language query for dataset search"),
    results: int = typer.Option(5, "--results", "-n", help="Number of results to show"),
):
    """
    Search for datasets using natural language (powered by Linkup API).

    Example: maxq search-data "dataset for legal document evaluation"
    """
    print_header()
    console.print(f'\n[dim]Searching for: "{query}"[/dim]\n')

    try:
        from .config import LINKUP_API_KEY

        linkup_key = LINKUP_API_KEY
    except ImportError:
        linkup_key = os.getenv("LINKUP_API_KEY")

    data_manager = DataSourceManager(linkup_api_key=linkup_key)

    with console.status("[bold green]Searching datasets...[/bold green]"):
        candidates = data_manager.search_datasets_nl(query, max_results=results)

    if not candidates:
        console.print("[yellow]No datasets found.[/yellow]")
        return

    if "error" in candidates[0]:
        console.print(f"[red]{candidates[0]['error']}[/red]")
        return

    console.print("[bold]Found Datasets:[/bold]\n")

    for i, dataset in enumerate(candidates, 1):
        source_type = dataset.get("source_type", "unknown")
        type_color = "green" if source_type == DataSourceType.HUGGINGFACE else "blue"

        console.print(
            Panel(
                f"[bold cyan]{dataset.get('name', 'Unknown')}[/bold cyan]\n\n"
                f"{dataset.get('description', 'No description')}\n\n"
                f"[dim]URL: {dataset.get('url', 'N/A')}[/dim]\n"
                f"[{type_color}]Type: {source_type}[/{type_color}]",
                title=f"[{i}]",
                expand=False,
            )
        )

    console.print("\n[dim]Use 'maxq import <url>' to import a dataset[/dim]")


@app.command(name="import")
def import_data(
    source: str = typer.Argument(
        ..., help="Data source (URL, path, dataset name, or 's3://bucket/path')"
    ),
    collection: str = typer.Option(None, "--collection", "-c", help="Target collection name"),
    limit: int = typer.Option(1000, "--limit", "-l", help="Maximum documents to import"),
    embedding_column: str = typer.Option(None, "--column", help="Column to embed"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip conflict checks"),
):
    """
    Import data from various sources into a collection.

    Supported sources:
    - HuggingFace: "fka/awesome-chatgpt-prompts"
    - URL: "https://example.com/data.json"
    - S3: "s3://bucket/path/to/files"
    - Local: "./data/files" or "/path/to/file.txt"
    - Qdrant snapshot: "snapshot://path/to/snapshot.snapshot"

    Example: maxq import fka/awesome-chatgpt-prompts --collection my-data
    """
    print_header()

    # Detect source type
    source_type = _detect_source_type(source)

    console.print(f"\n[dim]Source type: {source_type.value}[/dim]")
    console.print(f"[dim]Source: {source}[/dim]")

    if not collection:
        # Auto-generate collection name
        if "/" in source:
            collection = source.split("/")[-1].replace(".", "_").replace("-", "_")
        else:
            collection = source.replace(".", "_").replace("-", "_")[:30]
        console.print(f"[dim]Collection: {collection}[/dim]")

    # Get credentials
    q_url = os.getenv("QDRANT_URL", ":memory:")
    q_key = os.getenv("QDRANT_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    engine = MaxQEngine(qdrant_url=q_url, qdrant_api_key=q_key, openai_api_key=openai_key)

    # Create source config
    source_config = DataSourceConfig(
        source_type=source_type,
        source_path=source,
        name=source.split("/")[-1] if "/" in source else source,
        embedding_column=embedding_column,
        limit=limit,
    )

    # Check for conflicts
    multi_collection = MultiDatasetCollection(collection_name=collection, engine=engine)

    if not force and engine.collection_exists(collection):
        console.print(f"\n[yellow]Collection '{collection}' already exists.[/yellow]")

        # Check compatibility
        existing_datasets = multi_collection.list_datasets()
        if existing_datasets:
            console.print(f"[dim]Contains {len(existing_datasets)} dataset(s):[/dim]")
            for ds in existing_datasets[:3]:
                console.print(f"  • {ds.get('name', 'unknown')} ({ds.get('count', 0)} docs)")

        # Get collection config for conflict check
        config_check = {
            "dense_model": "BAAI/bge-base-en-v1.5",  # Default
            "dense_dimensions": 768,
        }
        conflict_report = multi_collection.check_compatibility(config_check)

        if conflict_report.warnings:
            for warning in conflict_report.warnings:
                console.print(f"[yellow]Warning: {warning}[/yellow]")

        if not conflict_report.can_proceed:
            console.print("[red]Cannot proceed due to conflicts:[/red]")
            for conflict in conflict_report.conflicts:
                console.print(f"  [red]• {conflict['message']}[/red]")
            return

        # Ask to add or overwrite
        console.print()
        add_choice = interactive_select(
            ["Add to existing collection", "Overwrite collection", "Cancel"], default_index=0
        )

        if "Cancel" in add_choice:
            return
        elif "Overwrite" in add_choice:
            force = True

    # Auto-configure based on data
    console.print("\n[bold dim]CONFIGURING[/bold dim]")

    with console.status("[bold green]Analyzing data and selecting config...[/bold green]"):
        # Sample data for config recommendation
        samples = []
        if source_type == DataSourceType.HUGGINGFACE:
            try:
                from datasets import load_dataset

                ds = load_dataset(source, split="train", streaming=True, trust_remote_code=True)
                for i, row in enumerate(ds):
                    if i >= 5:
                        break
                    samples.append({k: str(v)[:500] for k, v in row.items()})
            except Exception as e:
                console.print(f"[yellow]Could not sample data: {e}[/yellow]")

        if samples:
            recommendation = engine.recommend_config(
                samples, priority="balanced", use_llm=bool(openai_key)
            )
            config = engine.load_config(recommendation.recommended_config)
            console.print(
                f"[green]✓ Auto-selected config: {recommendation.recommended_config}[/green]"
            )
            console.print(f"[dim]{recommendation.reasoning}[/dim]")
        else:
            config = engine.load_config("paraphrase_bm25")
            console.print("[dim]Using default config: paraphrase_bm25[/dim]")

    # Import
    console.print("\n[bold dim]IMPORTING[/bold dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(f"[cyan]Importing from {source_type.value}...[/cyan]", total=limit)

        def update_progress(n):
            progress.update(task, advance=n)

        count, error = multi_collection.add_dataset(
            source_config=source_config, config=config, force=force, callback=update_progress
        )

    if error:
        console.print(f"\n[red]Import failed: {error}[/red]")
        return

    console.print(f"\n[bold green]✓ Imported {count} documents to '{collection}'[/bold green]")

    # Show collection summary
    datasets = multi_collection.list_datasets()
    if len(datasets) > 1:
        console.print(f"\n[dim]Collection now contains {len(datasets)} datasets:[/dim]")
        for ds in datasets:
            console.print(f"  • {ds.get('name', 'unknown')}: {ds.get('count', 0)} docs")


@app.command()
def datasets(collection: str = typer.Argument(..., help="Collection name")):
    """
    List all datasets in a collection.
    """
    print_header()

    q_url = os.getenv("QDRANT_URL", ":memory:")
    q_key = os.getenv("QDRANT_API_KEY")

    engine = MaxQEngine(qdrant_url=q_url, qdrant_api_key=q_key)

    if not engine.collection_exists(collection):
        console.print(f"[red]Collection '{collection}' not found[/red]")
        return

    data_manager = DataSourceManager(qdrant_client=engine.client)
    datasets_list = data_manager.list_datasets_in_collection(collection)

    if not datasets_list:
        console.print(f"[yellow]No datasets found in '{collection}'[/yellow]")
        return

    console.print(f"\n[bold]Datasets in '{collection}':[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Name")
    table.add_column("Version", style="green")
    table.add_column("Source")
    table.add_column("Model")
    table.add_column("Documents", justify="right")

    for ds in datasets_list:
        table.add_row(
            ds.get("dataset_id", "default")[:12],
            ds.get("name", "unknown"),
            ds.get("version", "1.0.0"),
            ds.get("source", "unknown")[:25] + "..."
            if len(ds.get("source", "")) > 25
            else ds.get("source", "unknown"),
            ds.get("embedding_model", "unknown").split("/")[-1][:15],
            str(ds.get("count", 0)),
        )

    console.print(table)
    console.print(
        "\n[dim]Use 'maxq versions <collection> <dataset_id>' to see version history[/dim]"
    )

    # Check eval compatibility
    conflict_detector = ConflictDetector(engine.client)
    eval_report = conflict_detector.check_eval_compatibility(collection)

    if eval_report.warnings:
        console.print("\n[yellow]Evaluation Warnings:[/yellow]")
        for warning in eval_report.warnings:
            console.print(f"  [yellow]• {warning}[/yellow]")


@app.command()
def check_conflicts(collection: str = typer.Argument(..., help="Collection name")):
    """
    Check a collection for compatibility issues (useful before eval).
    """
    print_header()

    q_url = os.getenv("QDRANT_URL", ":memory:")
    q_key = os.getenv("QDRANT_API_KEY")

    engine = MaxQEngine(qdrant_url=q_url, qdrant_api_key=q_key)

    if not engine.collection_exists(collection):
        console.print(f"[red]Collection '{collection}' not found[/red]")
        return

    console.print(f"\n[bold]Checking collection '{collection}' for issues...[/bold]\n")

    conflict_detector = ConflictDetector(engine.client)
    report = conflict_detector.check_eval_compatibility(collection)

    if not report.has_conflicts and not report.warnings:
        console.print("[green]✓ No issues detected. Collection is ready for evaluation.[/green]")
        return

    if report.conflicts:
        console.print("[red]Conflicts Detected:[/red]")
        for conflict in report.conflicts:
            console.print(
                f"  [red]• {conflict.get('type', 'unknown')}: {conflict.get('message', '')}[/red]"
            )

    if report.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in report.warnings:
            console.print(f"  [yellow]• {warning}[/yellow]")

    if report.can_proceed:
        console.print(
            "\n[dim]You can still run evaluations, but results may need interpretation.[/dim]"
        )
    else:
        console.print("\n[red]Cannot run reliable evaluations on this collection.[/red]")


def _detect_source_type(source: str) -> DataSourceType:
    """Detect the type of data source from the source string."""
    source_lower = source.lower()

    if source_lower.startswith("s3://"):
        return DataSourceType.S3_BUCKET
    elif source_lower.startswith("snapshot://"):
        return DataSourceType.QDRANT_SNAPSHOT
    elif source_lower.startswith(("http://", "https://")):
        if "huggingface.co/datasets/" in source_lower:
            # Extract dataset name from URL
            return DataSourceType.HUGGINGFACE
        return DataSourceType.URL
    elif os.path.exists(source):
        if os.path.isfile(source):
            return DataSourceType.LOCAL_FILE
        return DataSourceType.LOCAL_FOLDER
    elif "/" in source and not source.startswith("/"):
        # Looks like a HuggingFace dataset ID (org/name)
        return DataSourceType.HUGGINGFACE
    else:
        # Default to local path
        return DataSourceType.LOCAL_FOLDER


@app.command()
def versions(
    collection: str = typer.Argument(..., help="Collection name"),
    dataset_id: str = typer.Argument(..., help="Dataset ID to show versions for"),
):
    """
    Show version history for a dataset.
    """
    print_header()

    q_url = os.getenv("QDRANT_URL", ":memory:")
    q_key = os.getenv("QDRANT_API_KEY")

    engine = MaxQEngine(qdrant_url=q_url, qdrant_api_key=q_key)

    if not engine.collection_exists(collection):
        console.print(f"[red]Collection '{collection}' not found[/red]")
        return

    multi_collection = MultiDatasetCollection(collection_name=collection, engine=engine)
    version_list = multi_collection.get_dataset_versions(dataset_id)

    if not version_list:
        console.print(f"[yellow]No versions found for dataset '{dataset_id}'[/yellow]")
        return

    console.print(f"\n[bold]Version History for '{dataset_id}':[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Version", style="cyan")
    table.add_column("Added", style="dim")
    table.add_column("Documents", justify="right")
    table.add_column("Status")

    for v in version_list:
        status = "[green]current[/green]" if v.get("is_current", False) else "[dim]archived[/dim]"
        table.add_row(
            v.get("version", "?"),
            v.get("added_at", "unknown")[:19],
            str(v.get("document_count", 0)),
            status,
        )

    console.print(table)


@app.command()
def update_dataset(
    collection: str = typer.Argument(..., help="Collection name"),
    dataset_id: str = typer.Argument(..., help="Dataset ID to update"),
    source: str = typer.Argument(..., help="New data source"),
    change_type: str = typer.Option(
        "update", "--type", "-t", help="Change type: update, patch, reindex"
    ),
    description: str = typer.Option("", "--desc", "-d", help="Description of changes"),
    keep_old: bool = typer.Option(True, "--keep-old/--replace", help="Keep old version documents"),
    limit: int = typer.Option(1000, "--limit", "-l", help="Document limit"),
):
    """
    Update a dataset to a new version.

    Change types:
    - patch: Bug fixes, small corrections (1.0.0 -> 1.0.1)
    - update: New data added, minor changes (1.0.0 -> 1.1.0)
    - reindex: Major changes, re-embedding (1.0.0 -> 2.0.0)

    Example: maxq update-dataset my-collection abc123 fka/awesome-chatgpt-prompts --type update
    """
    print_header()

    q_url = os.getenv("QDRANT_URL", ":memory:")
    q_key = os.getenv("QDRANT_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    engine = MaxQEngine(qdrant_url=q_url, qdrant_api_key=q_key, openai_api_key=openai_key)

    if not engine.collection_exists(collection):
        console.print(f"[red]Collection '{collection}' not found[/red]")
        return

    multi_collection = MultiDatasetCollection(collection_name=collection, engine=engine)

    # Check dataset exists
    datasets = multi_collection.list_datasets()
    current_dataset = None
    for ds in datasets:
        if ds.get("dataset_id") == dataset_id:
            current_dataset = ds
            break

    if not current_dataset:
        console.print(f"[red]Dataset '{dataset_id}' not found[/red]")
        console.print(f"[dim]Available datasets: {[d.get('dataset_id') for d in datasets]}[/dim]")
        return

    current_version = current_dataset.get("version", "1.0.0")
    console.print(f"\n[dim]Current version: {current_version}[/dim]")
    console.print(f"[dim]Change type: {change_type}[/dim]")

    # Calculate new version for display
    parts = current_version.split(".")
    major, minor, patch_num = int(parts[0]), int(parts[1]), int(parts[2])
    if change_type == "reindex":
        new_version = f"{major + 1}.0.0"
    elif change_type == "update":
        new_version = f"{major}.{minor + 1}.0"
    else:
        new_version = f"{major}.{minor}.{patch_num + 1}"

    console.print(f"[cyan]New version will be: {new_version}[/cyan]\n")

    # Create source config
    source_type = _detect_source_type(source)
    source_config = DataSourceConfig(
        source_type=source_type, source_path=source, name=current_dataset.get("name"), limit=limit
    )

    # Load config (use default)
    config = engine.load_config("paraphrase_bm25")

    # Perform update
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(f"[cyan]Updating to v{new_version}...[/cyan]", total=limit)

        def update_progress(n):
            progress.update(task, advance=n)

        count, error = multi_collection.update_dataset(
            dataset_id=dataset_id,
            source_config=source_config,
            config=config,
            change_type=change_type,
            description=description,
            keep_old_versions=keep_old,
            callback=update_progress,
        )

    if error:
        console.print(f"\n[red]Update failed: {error}[/red]")
        return

    console.print(
        f"\n[bold green]✓ Updated dataset to v{new_version} ({count} documents)[/bold green]"
    )

    if keep_old:
        console.print(f"[dim]Old version v{current_version} preserved[/dim]")
    else:
        console.print(f"[dim]Old version v{current_version} replaced[/dim]")


@app.command()
def rollback(
    collection: str = typer.Argument(..., help="Collection name"),
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    version: str = typer.Argument(..., help="Version to rollback to"),
):
    """
    Rollback a dataset to a previous version.

    Example: maxq rollback my-collection abc123 1.0.0
    """
    print_header()

    q_url = os.getenv("QDRANT_URL", ":memory:")
    q_key = os.getenv("QDRANT_API_KEY")

    engine = MaxQEngine(qdrant_url=q_url, qdrant_api_key=q_key)

    if not engine.collection_exists(collection):
        console.print(f"[red]Collection '{collection}' not found[/red]")
        return

    multi_collection = MultiDatasetCollection(collection_name=collection, engine=engine)

    # Show current versions
    version_list = multi_collection.get_dataset_versions(dataset_id)
    if not version_list:
        console.print(f"[red]Dataset '{dataset_id}' not found[/red]")
        return

    current = next((v for v in version_list if v.get("is_current")), None)
    if current:
        console.print(f"\n[dim]Current version: {current.get('version')}[/dim]")

    console.print(f"[cyan]Rolling back to: {version}[/cyan]\n")

    # Confirm
    confirm = interactive_select([f"Yes, rollback to v{version}", "Cancel"], default_index=1)

    if "Cancel" in confirm:
        console.print("[dim]Rollback cancelled[/dim]")
        return

    # Perform rollback
    with console.status(f"[bold green]Rolling back to v{version}...[/bold green]"):
        success, error = multi_collection.rollback_dataset(dataset_id, version)

    if not success:
        console.print(f"\n[red]Rollback failed: {error}[/red]")
        return

    console.print(f"\n[bold green]✓ Successfully rolled back to v{version}[/bold green]")


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@app.command()
def studio():
    """Launch the MaxQ Studio (Next.js + FastAPI)"""
    import subprocess
    import sys
    import time
    import webbrowser

    server_path = os.path.join(os.path.dirname(__file__), "server", "main.py")
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "studio-ui"))

    # Check if ports are available
    if is_port_in_use(8888):
        console.print("[red]Error: Port 8888 is already in use.[/red]")
        console.print("[dim]Stop the existing process or use a different port.[/dim]")
        raise typer.Exit(1)

    if is_port_in_use(3333):
        console.print("[red]Error: Port 3333 is already in use.[/red]")
        console.print("[dim]Stop the existing process or use a different port.[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]Launching MaxQ Studio...[/green]")

    processes = []
    try:
        # 1. Start FastAPI Backend
        console.print(f"[dim]Starting Backend on http://localhost:8888...[/dim]")
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "maxq.server.main:app", "--reload", "--port", "8888"],
            cwd=os.path.join(os.path.dirname(__file__), ".."),  # Run from maxq-project root
        )
        processes.append(backend_process)

        # 2. Start Next.js Frontend
        console.print(f"[dim]Starting Frontend on http://localhost:3333...[/dim]")

        # Check if node_modules exists, if not install
        node_modules_path = os.path.join(ui_path, "node_modules")
        if not os.path.exists(node_modules_path):
            console.print("[dim]Installing frontend dependencies (first run)...[/dim]")
            install_result = subprocess.run(
                ["pnpm", "install"], cwd=ui_path, capture_output=True, text=True
            )
            if install_result.returncode != 0:
                # Try npm if pnpm fails
                install_result = subprocess.run(
                    ["npm", "install"], cwd=ui_path, capture_output=True, text=True
                )
                if install_result.returncode != 0:
                    console.print(f"[red]Failed to install frontend dependencies[/red]")
                    console.print(f"[dim]{install_result.stderr}[/dim]")
                    raise typer.Exit(1)

        frontend_process = subprocess.Popen(["pnpm", "dev", "-p", "3333"], cwd=ui_path)
        processes.append(frontend_process)

        # 3. Open Browser
        time.sleep(3)  # Wait a bit for servers to start
        webbrowser.open("http://localhost:3333")

        console.print(f"[bold green]Studio is running![/bold green]")
        console.print(f"Backend: http://localhost:8888")
        console.print(f"Frontend: http://localhost:3333")
        console.print(f"[dim]Press Ctrl+C to stop[/dim]")

        # Wait for processes
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping Studio...[/yellow]")
        for p in processes:
            p.terminate()
    except Exception as e:
        console.print(f"[red]Failed to launch studio: {e}[/red]")
        for p in processes:
            p.terminate()


@app.command()
def demo(
    ci: bool = typer.Option(False, "--ci", help="Run in non-interactive CI mode"),
    limit: int = typer.Option(100, "--limit", "-l", help="Number of movies to index"),
):
    """
    Run the MaxQ demo with a movie search engine.

    Indexes 100 movie descriptions and lets you search semantically.
    """
    import time

    print_header()
    console.print("\n[bold cyan]MaxQ Demo - Movie Search Engine[/bold cyan]\n")

    # Movie dataset - embedded in code for zero dependencies
    MOVIES = [
        {
            "title": "The Matrix",
            "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers. Features mind-bending action and philosophy about simulation theory.",
        },
        {
            "title": "Inception",
            "description": "A thief who enters people's dreams to steal secrets is given a chance to have his criminal record erased if he can plant an idea in someone's mind.",
        },
        {
            "title": "Interstellar",
            "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival. Epic space adventure with emotional father-daughter story.",
        },
        {
            "title": "The Shawshank Redemption",
            "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. A tale of hope and friendship.",
        },
        {
            "title": "Pulp Fiction",
            "description": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption. Iconic dialogue and non-linear storytelling.",
        },
        {
            "title": "The Dark Knight",
            "description": "Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy. A dark superhero film about chaos and moral choices.",
        },
        {
            "title": "Forrest Gump",
            "description": "The story of a man with low IQ who achieves great things in life while his childhood love struggles with personal issues. Heartwarming and inspiring.",
        },
        {
            "title": "Fight Club",
            "description": "An insomniac office worker forms an underground fight club that evolves into something much more. A psychological thriller about identity and consumerism.",
        },
        {
            "title": "The Godfather",
            "description": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son. Epic saga about family, power, and the American dream.",
        },
        {
            "title": "The Lord of the Rings: The Fellowship of the Ring",
            "description": "A hobbit and his companions set out on a journey to destroy a powerful ring and save Middle-earth. Fantasy adventure with epic battles.",
        },
        {
            "title": "Star Wars: A New Hope",
            "description": "Luke Skywalker joins forces with a Jedi Knight and a pilot to rescue a princess and save the galaxy. Classic space opera with lightsabers and the Force.",
        },
        {
            "title": "Jurassic Park",
            "description": "Scientists clone dinosaurs to create a theme park, but things go wrong when the dinosaurs escape. Thrilling adventure with groundbreaking visual effects.",
        },
        {
            "title": "Titanic",
            "description": "A young couple from different social classes fall in love aboard the ill-fated ship. Epic romance and disaster film set in 1912.",
        },
        {
            "title": "Avatar",
            "description": "A paraplegic marine on an alien moon bonds with the native species and fights to protect their world. Visually stunning science fiction with environmental themes.",
        },
        {
            "title": "The Avengers",
            "description": "Earth's mightiest heroes must come together to stop an alien invasion led by Loki. Action-packed superhero team-up movie.",
        },
        {
            "title": "Gladiator",
            "description": "A betrayed Roman general seeks revenge against the emperor who murdered his family. Epic historical action with powerful performances.",
        },
        {
            "title": "The Silence of the Lambs",
            "description": "A young FBI trainee seeks the help of an imprisoned cannibalistic serial killer to catch another killer. Psychological thriller and horror.",
        },
        {
            "title": "Schindler's List",
            "description": "The true story of a businessman who saved over a thousand Jews during the Holocaust. Powerful and emotional historical drama.",
        },
        {
            "title": "Goodfellas",
            "description": "The story of Henry Hill and his life in the mob. Crime drama chronicling the rise and fall of a gangster.",
        },
        {
            "title": "The Departed",
            "description": "An undercover cop and a mole in the police try to identify each other while infiltrating an Irish gang. Crime thriller with twist endings.",
        },
        {
            "title": "Saving Private Ryan",
            "description": "Following D-Day, a group of soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed. Intense war film.",
        },
        {
            "title": "Se7en",
            "description": "Two detectives hunt a serial killer who uses the seven deadly sins as motifs. Dark and disturbing crime thriller.",
        },
        {
            "title": "The Usual Suspects",
            "description": "A sole survivor tells the twisted events leading up to a horrific gun battle on a boat. Mystery thriller with legendary twist.",
        },
        {
            "title": "Memento",
            "description": "A man with short-term memory loss attempts to track down his wife's murderer. Mind-bending thriller told in reverse chronological order.",
        },
        {
            "title": "Back to the Future",
            "description": "A teenager is accidentally sent back in time and must ensure his parents fall in love. Fun sci-fi adventure with time travel.",
        },
        {
            "title": "Raiders of the Lost Ark",
            "description": "Archaeologist Indiana Jones races against Nazis to find the Ark of the Covenant. Classic action adventure with whip-cracking hero.",
        },
        {
            "title": "Alien",
            "description": "The crew of a spaceship encounters a deadly alien creature. Sci-fi horror with claustrophobic tension and iconic monster.",
        },
        {
            "title": "The Terminator",
            "description": "A cyborg assassin from the future is sent back to kill a woman whose son will lead humanity's resistance. Sci-fi action classic.",
        },
        {
            "title": "E.T. the Extra-Terrestrial",
            "description": "A boy befriends an alien stranded on Earth and helps him return home. Heartwarming family sci-fi about friendship.",
        },
        {
            "title": "Jaws",
            "description": "A giant shark attacks beach-goers and three men hunt it down. The original summer blockbuster thriller.",
        },
        {
            "title": "Die Hard",
            "description": "An NYPD officer tries to save hostages from terrorists in a Los Angeles skyscraper. Action movie with witty one-liners.",
        },
        {
            "title": "The Lion King",
            "description": "A young lion prince must reclaim his throne from his evil uncle. Animated musical about family and responsibility.",
        },
        {
            "title": "Toy Story",
            "description": "A cowboy doll feels threatened when a new spaceman figure arrives. Groundbreaking animated film about toys that come to life.",
        },
        {
            "title": "Finding Nemo",
            "description": "A clownfish searches the ocean to find his captured son. Animated underwater adventure about parental love.",
        },
        {
            "title": "The Incredibles",
            "description": "A family of superheroes must come out of hiding to save the world. Animated action comedy about family dynamics.",
        },
        {
            "title": "WALL-E",
            "description": "A robot designed to clean up Earth falls in love and follows his crush into space. Animated romance with environmental message.",
        },
        {
            "title": "Up",
            "description": "An elderly man ties balloons to his house to fly to South America. Animated adventure about loss, friendship, and adventure.",
        },
        {
            "title": "Spirited Away",
            "description": "A girl enters a world of spirits and must work in a bathhouse to save her parents. Japanese animated fantasy masterpiece.",
        },
        {
            "title": "Princess Mononoke",
            "description": "A prince gets involved in a struggle between forest gods and humans who consume resources. Epic animated environmental fantasy.",
        },
        {
            "title": "Akira",
            "description": "A secret military project endangers Neo-Tokyo when it turns a biker gang member into a rampaging psychic. Cyberpunk anime classic.",
        },
        {
            "title": "Blade Runner",
            "description": "A cop hunts down genetically engineered humans in a dystopian Los Angeles. Neo-noir sci-fi about what it means to be human.",
        },
        {
            "title": "2001: A Space Odyssey",
            "description": "Humanity finds a mysterious artifact buried on the Moon and sets off to find its origins. Philosophical sci-fi epic.",
        },
        {
            "title": "The Exorcist",
            "description": "A mother seeks help when her daughter is possessed by a mysterious entity. Classic horror about demonic possession.",
        },
        {
            "title": "The Shining",
            "description": "A family heads to an isolated hotel where an evil presence drives the father into madness. Psychological horror classic.",
        },
        {
            "title": "Get Out",
            "description": "A young African-American man visits his white girlfriend's family with disturbing results. Horror thriller with social commentary.",
        },
        {
            "title": "A Quiet Place",
            "description": "A family must live in silence to avoid creatures that hunt by sound. Tense horror with unique premise.",
        },
        {
            "title": "Hereditary",
            "description": "After their grandmother dies, a family discovers disturbing secrets about their ancestry. Disturbing psychological horror.",
        },
        {
            "title": "The Conjuring",
            "description": "Paranormal investigators help a family terrorized by a dark presence in their farmhouse. Effective supernatural horror.",
        },
        {
            "title": "It",
            "description": "Kids in a small town face an evil clown that emerges every 27 years to prey on children. Horror based on Stephen King.",
        },
        {
            "title": "Psycho",
            "description": "A secretary steals money and ends up at a motel run by a disturbed man. Hitchcock's groundbreaking psychological horror.",
        },
        {
            "title": "The Sixth Sense",
            "description": "A boy who communicates with spirits seeks help from a child psychologist. Supernatural thriller with famous twist ending.",
        },
        {
            "title": "When Harry Met Sally",
            "description": "Two friends debate whether men and women can ever just be friends. Romantic comedy about love and friendship.",
        },
        {
            "title": "Pretty Woman",
            "description": "A businessman falls for a prostitute he hires to be his companion. Romantic comedy fairy tale.",
        },
        {
            "title": "Notting Hill",
            "description": "A bookshop owner falls in love with a famous actress. British romantic comedy about unlikely love.",
        },
        {
            "title": "The Princess Bride",
            "description": "A farmhand rescues a princess from an evil prince. Fantasy romance with adventure, humor, and true love.",
        },
        {
            "title": "Eternal Sunshine of the Spotless Mind",
            "description": "A couple undergoes a procedure to erase each other from their memories. Sci-fi romance about love and loss.",
        },
        {
            "title": "La La Land",
            "description": "A jazz musician and aspiring actress fall in love while pursuing their dreams in Los Angeles. Musical romance.",
        },
        {
            "title": "The Notebook",
            "description": "A poor man and a rich woman fall in love in the 1940s South. Romantic drama spanning decades.",
        },
        {
            "title": "Casablanca",
            "description": "A nightclub owner in Morocco must choose between love and virtue. Classic romantic drama set during WWII.",
        },
        {
            "title": "Annie Hall",
            "description": "A comedian reflects on his failed relationship with a woman. Influential romantic comedy with fourth-wall breaking.",
        },
        {
            "title": "Bridesmaids",
            "description": "A woman's life unravels as she helps her best friend plan her wedding. Raunchy female-led comedy.",
        },
        {
            "title": "Superbad",
            "description": "Two high school friends try to have one last party before graduation. Coming-of-age comedy about teenage friendship.",
        },
        {
            "title": "The Hangover",
            "description": "Three friends wake up in Las Vegas with no memory of the previous night and a missing groom. Wild comedy adventure.",
        },
        {
            "title": "Anchorman",
            "description": "A 1970s San Diego news anchor's career is threatened by an ambitious female reporter. Absurdist workplace comedy.",
        },
        {
            "title": "Step Brothers",
            "description": "Two middle-aged men become stepbrothers and must learn to live together. Silly comedy about arrested development.",
        },
        {
            "title": "Groundhog Day",
            "description": "A weatherman relives the same day over and over. Comedy with philosophical depth about self-improvement.",
        },
        {
            "title": "Ferris Bueller's Day Off",
            "description": "A high school student fakes being sick to have a day of adventure. 80s teen comedy about seizing the day.",
        },
        {
            "title": "Home Alone",
            "description": "A boy accidentally left behind during Christmas must defend his home from burglars. Family comedy classic.",
        },
        {
            "title": "Mrs. Doubtfire",
            "description": "A divorced father disguises himself as a female housekeeper to spend time with his children. Family comedy.",
        },
        {
            "title": "Shrek",
            "description": "An ogre and a donkey set out to rescue a princess. Animated comedy that parodies fairy tale conventions.",
        },
        {
            "title": "Zootopia",
            "description": "A bunny cop and a fox con artist team up to uncover a conspiracy. Animated comedy with social themes.",
        },
        {
            "title": "The Grand Budapest Hotel",
            "description": "A concierge and his lobby boy are framed for murder. Quirky comedy with distinctive visual style.",
        },
        {
            "title": "Moonlight",
            "description": "A young African-American man grapples with his identity and sexuality in Miami. Coming-of-age drama in three acts.",
        },
        {
            "title": "12 Years a Slave",
            "description": "A free black man is kidnapped and sold into slavery. Harrowing historical drama based on true story.",
        },
        {
            "title": "Parasite",
            "description": "A poor family schemes to become employed by a wealthy family. Korean thriller about class divide.",
        },
        {
            "title": "The Social Network",
            "description": "The story of the founding of Facebook and the lawsuits that followed. Drama about ambition and betrayal.",
        },
        {
            "title": "Whiplash",
            "description": "A young drummer is pushed to his limits by a demanding jazz instructor. Intense drama about pursuing excellence.",
        },
        {
            "title": "Manchester by the Sea",
            "description": "A man must care for his teenage nephew after his brother dies. Grief drama with devastating emotional impact.",
        },
        {
            "title": "Brokeback Mountain",
            "description": "Two cowboys develop a forbidden relationship over 20 years. Romantic drama about forbidden love.",
        },
        {
            "title": "Boyhood",
            "description": "The life of a boy from age 6 to 18. Drama filmed over 12 years with the same actors.",
        },
        {
            "title": "The Revenant",
            "description": "A frontiersman fights for survival after being mauled by a bear. Survival drama with breathtaking cinematography.",
        },
        {
            "title": "Mad Max: Fury Road",
            "description": "A woman rebels against a tyrant in a post-apocalyptic desert. Non-stop action with feminist themes.",
        },
        {
            "title": "John Wick",
            "description": "A retired hitman seeks vengeance for his murdered dog. Stylish action with elaborate gun choreography.",
        },
        {
            "title": "The Raid",
            "description": "A SWAT team becomes trapped in a building run by a crime lord. Indonesian martial arts action masterpiece.",
        },
        {
            "title": "Mission: Impossible - Fallout",
            "description": "Ethan Hunt and his team race against time after a mission goes wrong. High-octane spy action.",
        },
        {
            "title": "Edge of Tomorrow",
            "description": "A soldier fighting aliens relives the same day repeatedly. Sci-fi action with time loop premise.",
        },
        {
            "title": "District 9",
            "description": "Aliens land in South Africa and are forced to live in slums. Sci-fi action with apartheid allegory.",
        },
        {
            "title": "Ex Machina",
            "description": "A programmer is invited to test whether an AI has true consciousness. Sci-fi thriller about artificial intelligence.",
        },
        {
            "title": "Arrival",
            "description": "A linguist is recruited to communicate with aliens who have arrived on Earth. Thoughtful sci-fi about language and time.",
        },
        {
            "title": "Her",
            "description": "A man falls in love with an artificially intelligent operating system. Romantic sci-fi about connection and loneliness.",
        },
        {
            "title": "Gravity",
            "description": "Two astronauts must survive after debris destroys their space shuttle. Tense survival thriller in space.",
        },
        {
            "title": "The Martian",
            "description": "An astronaut stranded on Mars must find a way to survive. Sci-fi survival with humor and science.",
        },
        {
            "title": "Dunkirk",
            "description": "Allied soldiers are evacuated from the beaches of Dunkirk. WWII film told from land, sea, and air perspectives.",
        },
        {
            "title": "1917",
            "description": "Two soldiers must deliver a message through enemy territory. WWI film presented as one continuous shot.",
        },
        {
            "title": "Hacksaw Ridge",
            "description": "A WWII medic saves 75 men without carrying a weapon. War drama about faith and courage.",
        },
        {
            "title": "Black Panther",
            "description": "A new king must defend his technologically advanced African nation. Superhero film with Afrofuturist themes.",
        },
        {
            "title": "Spider-Man: Into the Spider-Verse",
            "description": "A teenager becomes Spider-Man and meets alternate versions of himself. Animated superhero with unique visual style.",
        },
        {
            "title": "Logan",
            "description": "An aging Wolverine cares for a sick Professor X. Gritty superhero drama about mortality and legacy.",
        },
        {
            "title": "Joker",
            "description": "A struggling comedian descends into madness and becomes a criminal. Dark character study of Batman villain.",
        },
    ]

    # Connect to Qdrant - try remote first, fall back to in-memory
    q_url = os.getenv("QDRANT_URL", "")
    use_memory = False

    from .search_engine import MaxQEngine, CollectionStrategy, SearchRequest

    if q_url:
        console.print(f"[dim]Connecting to {q_url}...[/dim]")
        try:
            engine = MaxQEngine(qdrant_url=q_url)
            console.print("[green]✓ Connected to Qdrant[/green]")
        except Exception as e:
            console.print(f"[yellow]Cannot connect to {q_url}[/yellow]")
            console.print("[dim]Using in-memory mode instead[/dim]")
            use_memory = True
    else:
        # No URL configured - use in-memory for instant demo
        use_memory = True

    if use_memory:
        console.print("[cyan]Running in-memory demo (data won't persist)[/cyan]")
        from qdrant_client import QdrantClient
        memory_client = QdrantClient(":memory:")
        engine = MaxQEngine(qdrant_client=memory_client)

    # Index movies
    movies_to_index = MOVIES[:limit]
    console.print(f"\n[dim]Indexing {len(movies_to_index)} movies with bge-small-en-v1.5...[/dim]")

    config = CollectionStrategy(
        collection_name="maxq-demo-movies",
        dense_model_name="BAAI/bge-small-en-v1.5",
        sparse_model_name="prithivida/Splade_PP_en_v1",
        estimated_doc_count=len(movies_to_index),
    )

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Indexing movies...[/cyan]", total=len(movies_to_index))

        # Prepare documents
        documents = [f"{m['title']}: {m['description']}" for m in movies_to_index]

        # Batch index
        engine.index_documents(
            config, documents, callback=lambda n: progress.update(task, advance=n)
        )

    elapsed = time.time() - start_time
    console.print(
        f"\n[bold green]Done! {len(movies_to_index)} movies indexed in {elapsed:.1f}s[/bold green]"
    )

    # Suggest queries
    console.print("\n[bold white]Try these searches:[/bold white]")
    console.print('  maxq search "space adventure with robots"')
    console.print('  maxq search "romantic comedy in new york"')
    console.print('  maxq search "scary movie with supernatural elements"')
    console.print('  maxq search "animated movie about family"')
    console.print('  maxq search "crime thriller with twist ending"')

    console.print(f"\n[dim]Open Studio UI: http://localhost:3000[/dim]")

    # Interactive mode (unless --ci)
    if not ci:
        console.print("\n[bold dim]INTERACTIVE SEARCH[/bold dim]")
        console.print("[dim]Type a query to search. Ctrl+C to exit.[/dim]\n")

        while True:
            try:
                query_text = Prompt.ask("[bold cyan]Search[/bold cyan]")

                if not query_text.strip():
                    continue

                request = SearchRequest(query=query_text, limit=5, strategy="hybrid")

                results = engine.query(config, request)

                if not results:
                    console.print("[yellow]No results found.[/yellow]\n")
                    continue

                console.print()
                for hit in results:
                    score = f"{hit.score:.3f}"
                    text = hit.payload.get("_text", "")
                    # Extract title
                    title = text.split(":")[0] if ":" in text else "Movie"
                    desc = (
                        text.split(":", 1)[1].strip()[:150] + "..."
                        if ":" in text
                        else text[:150] + "..."
                    )
                    console.print(f"[yellow]{score}[/yellow] [bold white]{title}[/bold white]")
                    console.print(f"  [dim]{desc}[/dim]")
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye![/dim]")
                break


@app.command()
def doctor():
    """
    Run system diagnostics and check MaxQ health.

    Checks Qdrant connection, API keys, cache directories, and versions.
    """
    import shutil
    import platform

    print_header()
    console.print("\n[bold white]MaxQ System Check[/bold white]")
    console.print("[dim]" + "─" * 40 + "[/dim]\n")

    all_ok = True

    # 1. Check Qdrant connection
    q_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    q_key = os.getenv("QDRANT_API_KEY")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=q_url, api_key=q_key, timeout=5)
        # Try to list collections as a health check
        client.get_collections()
        console.print(f"[green]✓[/green] Qdrant connection: {q_url} [green](healthy)[/green]")
    except Exception as e:
        console.print(f"[red]✗[/red] Qdrant connection: {q_url} [red](failed: {str(e)[:50]})[/red]")
        all_ok = False

    # 1b. Check Rust engine
    try:
        from .search_engine import get_engine_mode, is_engine_available

        engine_mode = get_engine_mode()
        engine_url = os.getenv("MAXQ_ENGINE_URL", os.getenv("MAXQ_SIDECAR_URL", "localhost:50051"))

        if engine_mode == "false":
            console.print(f"[dim]○[/dim] Rust engine: [dim]disabled[/dim] (MAXQ_USE_ENGINE=false)")
        else:
            engine_available = is_engine_available()
            if engine_available:
                console.print(
                    f"[green]✓[/green] Rust engine: {engine_url} [green](active, ~10x faster)[/green]"
                )
            else:
                status = (
                    "[yellow]not running[/yellow]"
                    if engine_mode == "auto"
                    else "[red]required but unavailable[/red]"
                )
                console.print(f"[yellow]⚠[/yellow] Rust engine: {engine_url} ({status})")
                if engine_mode == "true":
                    all_ok = False
    except Exception as e:
        console.print(f"[dim]○[/dim] Rust engine: [dim]check failed ({str(e)[:30]})[/dim]")

    # 2. Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        # Mask the key
        masked = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        console.print(f"[green]✓[/green] OpenAI API key: [green]configured[/green] ({masked})")
    else:
        console.print(f"[yellow]⚠[/yellow] OpenAI API key: [yellow]not set[/yellow] (optional)")

    linkup_key = os.getenv("LINKUP_API_KEY")
    if linkup_key:
        console.print(f"[green]✓[/green] Linkup API key: [green]configured[/green]")
    else:
        console.print(f"[dim]○[/dim] Linkup API key: [dim]not set[/dim] (optional)")

    maxq_api_key = os.getenv("MAXQ_API_KEY")
    if maxq_api_key:
        console.print(f"[green]✓[/green] MaxQ API key: [green]enabled[/green] (API auth active)")
    else:
        console.print(f"[dim]○[/dim] MaxQ API key: [dim]not set[/dim] (API auth disabled)")

    # 3. Check data directories
    from .config import MAXQ_APP_DIR

    if MAXQ_APP_DIR.exists():
        total_size = sum(f.stat().st_size for f in MAXQ_APP_DIR.rglob("*") if f.is_file())
        size_str = (
            f"{total_size / (1024 * 1024):.1f} MB"
            if total_size > 1024 * 1024
            else f"{total_size / 1024:.0f} KB"
        )
        console.print(f"[green]✓[/green] Data directory: {MAXQ_APP_DIR} ({size_str})")
    else:
        console.print(f"[dim]○[/dim] Data directory: {MAXQ_APP_DIR} [dim](not created yet)[/dim]")

    # 4. Python and MaxQ version
    python_version = platform.python_version()
    console.print(f"[green]✓[/green] Python version: {python_version}")

    try:
        import maxq

        maxq_version = getattr(maxq, "__version__", "0.1.0")
    except:
        maxq_version = "0.1.0"
    console.print(f"[green]✓[/green] MaxQ version: {maxq_version}")

    # 5. Check dependencies
    console.print()
    console.print("[dim]Dependencies:[/dim]")

    deps_to_check = [
        ("qdrant-client", "qdrant_client"),
        ("openai", "openai"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]

    for name, module in deps_to_check:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            console.print(f"  [green]✓[/green] {name}: {version}")
        except ImportError:
            console.print(f"  [red]✗[/red] {name}: [red]not installed[/red]")
            all_ok = False

    # Final status
    console.print()
    if all_ok:
        console.print("[bold green]All systems operational.[/bold green]")
    else:
        console.print("[bold yellow]Some issues detected. See above for details.[/bold yellow]")

    return 0 if all_ok else 1


@app.command()
def engine(
    install: bool = typer.Option(
        False, "--install", "-i", help="Download and install the engine binary"
    ),
    start: bool = typer.Option(False, "--start", "-s", help="Start the engine process"),
    status: bool = typer.Option(False, "--status", help="Show engine status"),
    qdrant_url: str = typer.Option("http://localhost:6334", "--qdrant-url", help="Qdrant gRPC URL"),
    port: int = typer.Option(50051, "--port", "-p", help="gRPC port for the engine"),
):
    """
    Manage the Rust engine for ~10x faster embeddings.

    Examples:
        maxq engine --status       # Show engine status
        maxq engine --install      # Download the engine binary
        maxq engine --start        # Start the engine process
    """
    from .search_engine import (
        is_engine_available,
        get_engine_mode,
        get_engine_status,
        ensure_engine_binary,
        start_engine_fn,
    )

    print_header()

    if status or (not install and not start):
        # Show status
        console.print("\n[bold white]Engine Status[/bold white]")
        console.print("[dim]" + "─" * 40 + "[/dim]\n")

        info = get_engine_status()
        mode = get_engine_mode()
        available = is_engine_available()

        console.print(f"Platform: {info['platform']}")
        console.print(f"Version: {info['version']}")
        console.print(f"Mode: {mode} (MAXQ_USE_ENGINE)")
        console.print(
            f"Supported: {'[green]Yes[/green]' if info['supported'] else '[red]No[/red]'}"
        )
        console.print(
            f"Installed: {'[green]Yes[/green]' if info['installed'] else '[yellow]No[/yellow]'}"
        )
        console.print(
            f"Available: {'[green]Yes (running)[/green]' if available else '[dim]No (not running)[/dim]'}"
        )
        console.print(f"Binary path: {info['binary_path'] or '[dim]N/A[/dim]'}")
        console.print(f"Cache dir: {info['cache_dir']}")

        if not info["installed"] and info["supported"]:
            console.print(
                "\n[yellow]Tip:[/yellow] Run 'maxq engine --install' to download the binary"
            )
        if info["installed"] and not available:
            console.print("\n[yellow]Tip:[/yellow] Run 'maxq engine --start' to start the engine")

    if install:
        console.print("\n[bold white]Installing Engine[/bold white]")
        console.print("[dim]" + "─" * 40 + "[/dim]\n")

        binary_path = ensure_engine_binary(force_download=True)
        if binary_path:
            console.print(f"[green]✓[/green] Installed: {binary_path}")
        else:
            console.print("[red]✗[/red] Installation failed")
            console.print(
                "[dim]You can build from source: cd maxq-rs && cargo build --release[/dim]"
            )

    if start:
        console.print("\n[bold white]Starting Engine[/bold white]")
        console.print("[dim]" + "─" * 40 + "[/dim]\n")

        process = start_engine_fn(qdrant_url=qdrant_url, grpc_port=port)
        if process:
            console.print(f"[green]✓[/green] Engine started (PID: {process.pid})")
            console.print(f"[dim]gRPC endpoint: localhost:{port}[/dim]")
            console.print("\n[yellow]Note:[/yellow] Press Ctrl+C to stop the engine")
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                console.print("\n[dim]Engine stopped[/dim]")
        else:
            console.print("[red]✗[/red] Failed to start engine")


# Backwards compatibility alias
@app.command(hidden=True)
def sidecar(
    install: bool = typer.Option(
        False, "--install", "-i", help="Download and install the engine binary"
    ),
    start: bool = typer.Option(False, "--start", "-s", help="Start the engine process"),
    status: bool = typer.Option(False, "--status", help="Show engine status"),
    qdrant_url: str = typer.Option("http://localhost:6334", "--qdrant-url", help="Qdrant gRPC URL"),
    port: int = typer.Option(50051, "--port", "-p", help="gRPC port for the engine"),
):
    """Alias for 'maxq engine' (deprecated, use 'maxq engine' instead)."""
    engine(install=install, start=start, status=status, qdrant_url=qdrant_url, port=port)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option(
        "maxq-demo-movies", "--collection", "-c", help="Collection to search"
    ),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results"),
    strategy: str = typer.Option(
        "hybrid", "--strategy", "-s", help="Search strategy: hybrid, dense, sparse"
    ),
):
    """
    Search a collection with a natural language query.

    Example: maxq search "space adventure with robots"
    """
    from .search_engine import MaxQEngine, CollectionStrategy, SearchRequest

    q_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    try:
        engine = MaxQEngine(qdrant_url=q_url)
    except Exception as e:
        console.print(f"[red]Failed to connect to Qdrant: {e}[/red]")
        return

    if not engine.collection_exists(collection):
        console.print(f"[red]Collection '{collection}' not found.[/red]")
        console.print("[dim]Run 'maxq demo' first to create a sample collection.[/dim]")
        return

    config = CollectionStrategy(
        collection_name=collection,
        dense_model_name="BAAI/bge-small-en-v1.5",
        sparse_model_name="prithivida/Splade_PP_en_v1",
    )

    request = SearchRequest(query=query, limit=limit, strategy=strategy)

    results = engine.query(config, request)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print()
    for hit in results:
        score = f"{hit.score:.3f}"
        text = hit.payload.get("_text", "")[:200]
        console.print(f"[yellow]{score}[/yellow] {text}...")
    console.print()


@app.command()
def start():
    """Start the MaxQ Workbench (Setup -> Configure -> Ingest)"""
    try:
        # --- PHASE 1: PROJECT SETUP & CONFIGURATION ---
        project_config = project_setup_wizard()

        print_header()

        # --- VALIDATION: Check for missing credentials ---
        warnings = []

        # Check if Qdrant Cloud selected but no credentials
        if project_config["qdrant_deploy"] == "Cloud" and not project_config["q_url"]:
            warnings.append(
                "⚠️  [yellow]Qdrant Cloud selected but no credentials provided.[/yellow]"
            )
            warnings.append("   [dim]→ Data will be stored in-memory only (not persisted)[/dim]")

        # Check if OpenAI features might be used but no key
        if not project_config["api_key"] and project_config["dataset_name"]:
            warnings.append("⚠️  [yellow]No API key provided.[/yellow]")
            warnings.append("   [dim]→ Automatic dataset analysis will be skipped[/dim]")
            warnings.append("   [dim]→ You'll need to manually specify which column to embed[/dim]")

        if warnings:
            console.print()
            for warning in warnings:
                console.print(warning)
            console.print()

            if (
                not Prompt.ask("[cyan]Continue anyway?[/cyan]", choices=["y", "n"], default="y")
                == "y"
            ):
                console.print("[yellow]Setup cancelled. Run 'maxq start' again to retry.[/yellow]")
                sys.exit(0)

        key_status = (
            "[green]Provided[/green]" if project_config["api_key"] else "[yellow]Skipped[/yellow]"
        )
        console.print(
            f"\n[dim]Initializing project: [bold cyan]{project_config['name']}[/bold cyan] using [bold cyan]{project_config['provider']}[/bold cyan] (Key: {key_status})...[/dim]"
        )
        console.print(
            f"[dim]Qdrant: [bold cyan]{project_config['qdrant_deploy']}[/bold cyan][/dim]"
        )

        # --- PHASE 1.5: GENERATE PROJECT FILES ---
        console.print(f"\n[bold dim]STEP 0: SCAFFOLDING[/bold dim]")
        project_dir = os.path.join(os.getcwd(), project_config["name"])

        uv_success = False
        try:
            import subprocess

            # Check if uv is available
            subprocess.run(["uv", "--version"], check=True, capture_output=True)

            console.print(f"   [dim]Initializing project with uv...[/dim]")
            # uv init creates the directory
            subprocess.run(
                ["uv", "init", "--no-workspace", project_config["name"]],
                check=True,
                capture_output=True,
            )
            uv_success = True
            console.print(
                f"   [green]✓ Created project with uv: [bold cyan]{project_config['name']}/[/bold cyan][/green]"
            )

            # Add dependencies (maxq is installed separately, not on PyPI)
            console.print(f"   [dim]Adding dependencies...[/dim]")
            subprocess.run(
                ["uv", "add", "qdrant-client", "openai", "python-dotenv"],
                cwd=project_dir,
                check=True,
                capture_output=True,
            )
            console.print(f"   [green]✓ Added dependencies to pyproject.toml[/green]")

            # Remove default hello.py if it exists
            hello_path = os.path.join(project_dir, "hello.py")
            if os.path.exists(hello_path):
                os.remove(hello_path)

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip() if e.stderr else str(e)
            console.print(f"   [yellow]⚠️ uv command failed: {error_msg}[/yellow]")
            console.print(f"   [dim]Falling back to standard setup.[/dim]")

            if not os.path.exists(project_dir):
                os.makedirs(project_dir, exist_ok=True)
                console.print(
                    f"   [green]✓ Created directory: [bold cyan]{project_config['name']}/[/bold cyan][/green]"
                )
            else:
                console.print(f"   [yellow]⚠️ Directory exists. Skipping creation.[/yellow]")

        except FileNotFoundError:
            console.print(
                f"   [yellow]⚠️ uv not found in PATH. Falling back to standard setup.[/yellow]"
            )
            if not os.path.exists(project_dir):
                os.makedirs(project_dir, exist_ok=True)
                console.print(
                    f"   [green]✓ Created directory: [bold cyan]{project_config['name']}/[/bold cyan][/green]"
                )
            else:
                console.print(f"   [yellow]⚠️ Directory exists. Skipping creation.[/yellow]")

        # Generate .env
        env_content = f"""QDRANT_URL={project_config["q_url"] or ":memory:"}
QDRANT_API_KEY={project_config["q_key"] or ""}
OPENAI_API_KEY={project_config["api_key"] or ""}
"""
        with open(os.path.join(project_dir, ".env"), "w") as f:
            f.write(env_content)
        console.print(f"   [green]✓ Generated [bold cyan].env[/bold cyan][/green]")

        # Generate requirements.txt ONLY if uv failed
        if not uv_success:
            req_content = """maxq
python-dotenv
qdrant-client
openai
"""
            with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
                f.write(req_content)
            console.print(f"   [green]✓ Generated [bold cyan]requirements.txt[/bold cyan][/green]")

        # Map search strategy
        strat_map = {
            "Hybrid (Dense + Sparse)": "hybrid",
            "Dense only": "dense",
            "Sparse only": "sparse",
        }
        strat_code = strat_map.get(project_config["search_strategy"], "hybrid")
        use_quant_bool = "Int8" in project_config["quantization"]

        # Generate main.py
        main_content = f"""import os
from dotenv import load_dotenv
from maxq.search_engine import MaxQEngine, CollectionStrategy, SearchRequest
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

# Initialize Rich Console for output
console = Console()

# Load environment variables
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
    console.print(f"[dim]Loaded configuration from: {{env_path}}[/dim]")
else:
    # Fallback: Try looking in parent directory explicitly
    parent_env = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        env_path = parent_env
        console.print(f"[dim]Loaded configuration from: {{env_path}} (parent directory)[/dim]")
    else:
        console.print(f"[dim]No .env file found (checked current and parent directories)[/dim]")

def main():
    # 1. Initialize Engine
    engine = MaxQEngine(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 2. Define Strategy
    config = CollectionStrategy(
        collection_name="{project_config["collection_name"]}",
        estimated_doc_count={project_config["max_documents"]},
        use_quantization={use_quant_bool},
        dense_model_name="{project_config["embedding_model"]}"
    )
    
    print(f"Connected to MaxQ Engine. Collection: {{config.collection_name}}")
    
    # 3. Interactive Search Loop
    print("\\n--- MaxQ Search (Type 'exit' to quit) ---")
    while True:
        try:
            query = input("\\nQuery: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            request = SearchRequest(
                query=query,
                limit=3,
                strategy="{strat_code}"
            )
            
            results = engine.query(config, request)
            
            if not results:
                print("No results found.")
                continue
                
            for hit in results:
                print(f"\\n[Score: {{hit.score:.2f}}]")
                print(f"{{hit.payload.get('_text', '')[:200]}}...")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {{e}}")

if __name__ == "__main__":
    main()
"""
        with open(os.path.join(project_dir, "main.py"), "w") as f:
            f.write(main_content)
        console.print(f"   [green]✓ Generated [bold cyan]main.py[/bold cyan][/green]")

        # Generate README.md
        readme_content = f"""# {project_config["name"]}

Powered by MaxQ Vector Search.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the search app:
   ```bash
   python main.py
   ```

## Configuration

- **Collection**: {project_config["collection_name"]}
- **Embedding Model**: {project_config["embedding_model"]}
- **Quantization**: {project_config["quantization"]}
- **Strategy**: {project_config["search_strategy"]}
"""
        with open(os.path.join(project_dir, "README.md"), "w") as f:
            f.write(readme_content)
        console.print(f"   [green]✓ Generated [bold cyan]README.md[/bold cyan][/green]")

        # --- PHASE 2: CONNECT ---
        console.print("\n[bold dim]STEP 1: CONNECT[/bold dim]")

        from .search_engine import MaxQEngine, CollectionStrategy, SearchRequest

        engine = MaxQEngine(
            qdrant_url=project_config["q_url"],
            qdrant_api_key=project_config["q_key"],
            openai_api_key=project_config["api_key"],
        )
        console.print("   [green]✓ Connected to Qdrant[/green]")

        # --- PHASE 3: ANALYZE DATA (if OpenAI key provided) ---
        dataset_name = project_config.get("dataset_name")
        folder_path = project_config.get("folder_path")
        embedding_col = None

        # Only analyze if we are NOT skipping data step
        if project_config["api_key"] and dataset_name and not project_config["skip_data_step"]:
            console.print("\n[bold dim]STEP 2: ANALYZING DATA[/bold dim]")
            try:
                with console.status("[bold green]Analyzing dataset structure...[/bold green]"):
                    from datasets import load_dataset

                    ds = load_dataset(
                        dataset_name, split="train", streaming=True, trust_remote_code=True
                    )
                    samples = []
                    for i, row in enumerate(ds):
                        if i >= 3:
                            break
                        samples.append({k: str(v)[:200] for k, v in row.items()})

                    analysis = engine.analyze_data_strategy(samples)

                if analysis.get("embedding_column"):
                    embedding_col = analysis["embedding_column"]
                    console.print(
                        f"   [green]✓ Will embed column: [bold cyan]'{embedding_col}'[/bold cyan][/green]"
                    )
                    console.print(f"   [dim]{analysis.get('reason')}[/dim]")

                    skipped = analysis.get("skipped_columns", [])
                    if skipped:
                        console.print(f"\n   [yellow]Skipped Columns:[/yellow]")
                        for col in skipped:
                            name = col.get("name", "unknown")
                            reason = col.get("reason", "no reason provided")
                            console.print(f"   • [dim]{name}: {reason}[/dim]")
            except Exception as e:
                console.print(f"   [yellow]⚠️ Could not analyze dataset: {str(e)[:100]}[/yellow]")
                console.print(f"   [dim]Continuing without auto-detection...[/dim]")

        # --- PHASE 4: CONFIGURATION ---
        limit = project_config["max_documents"]

        # Build CollectionStrategy
        use_quant = "Int8" in project_config["quantization"]

        config = CollectionStrategy(
            collection_name=project_config["collection_name"],
            estimated_doc_count=limit,
            use_quantization=use_quant,
            dense_model_name=project_config["embedding_model"],
        )

        console.print(f"\n[bold dim]STRATEGY[/bold dim]")
        console.print(f"   • Collection: {project_config['collection_name']}")
        console.print(f"   • Dense Model: {project_config['embedding_model'].split('/')[-1]}")
        console.print(f"   • Sparse Model: {config.sparse_model_name.split('/')[-1]}")
        console.print(f"   • Strategy: {project_config['search_strategy']}")
        console.print(f"   • Quantization: {project_config['quantization']}")
        console.print(f"   • {config.shard_number} shards")
        console.print(f"\n[dim]Note: First run may take a few minutes to download models.[/dim]")

        # --- PHASE 5: INGESTION ---
        console.print("\n[bold dim]STEP 3: INDEXING[/bold dim]")

        should_ingest = True

        if project_config["skip_data_step"]:
            should_ingest = False
            console.print("   [dim]Using existing collection (Ingestion skipped).[/dim]")
            try:
                count = engine.client.count(config.collection_name).count
            except:
                count = "Unknown"

        elif not dataset_name and not folder_path:
            should_ingest = False
            console.print("   [dim]No data source selected (Ingestion skipped).[/dim]")
            try:
                count = engine.client.count(config.collection_name).count
            except:
                count = "Unknown"

        elif engine.collection_exists(config.collection_name):
            console.print(
                f"   [yellow]Collection '{config.collection_name}' already exists.[/yellow]"
            )
            if (
                Prompt.ask("   [cyan]Re-index (overwrite)?[/cyan]", choices=["y", "n"], default="n")
                == "n"
            ):
                should_ingest = False
                console.print("   [dim]Skipping ingestion. Using existing collection.[/dim]")
                try:
                    count = engine.client.count(config.collection_name).count
                except:
                    count = "Unknown"

        if should_ingest:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("[cyan]Ingesting...[/cyan]", total=limit)

                def update_progress(n):
                    progress.update(task, advance=n)

                if dataset_name:
                    count = engine.ingest_hf(
                        dataset_name,
                        config,
                        limit,
                        embedding_column=embedding_col,
                        callback=update_progress,
                    )
                else:
                    count = engine.ingest_local(
                        folder_path, config, limit, glob_pattern="**/*.*", callback=update_progress
                    )

        console.print(f"\n[bold green]✨ Success![/bold green]")
        console.print(f"[bold white]📊 Indexed {count} documents[/bold white]")

        # --- PHASE 5.5: SAVE PROJECT TO STUDIO DATABASE ---
        try:
            from .server.database import ProjectStore, init_db
            from .server.models import Project, IndexedModelInfo
            from datetime import datetime
            import uuid

            init_db()

            # Create project in Studio database
            project = Project(
                id=str(uuid.uuid4()),
                name=project_config["name"],
                description=f"Created via CLI with {dataset_name or folder_path}",
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                task_type="search",
                embedding_model=project_config["embedding_model"],
            )
            ProjectStore.create(project)

            # Add indexed model info
            indexed_model = IndexedModelInfo(
                model_name=project_config["embedding_model"],
                collection_name=config.collection_name,
                indexed_at=datetime.now(),
                point_count=count if isinstance(count, int) else 0,
            )
            ProjectStore.add_indexed_model(project.id, indexed_model)

            console.print(f"   [green]✓ Project saved to Studio database[/green]")
        except Exception as e:
            console.print(f"   [yellow]⚠️ Could not save to Studio database: {e}[/yellow]")

        # --- PHASE 6: LAUNCH STUDIO ---
        console.print("\n[bold dim]STEP 4: LAUNCH STUDIO[/bold dim]")
        console.print("   [dim]Opening MaxQ Studio in your browser...[/dim]\n")

        # Launch Studio
        studio()

    except KeyboardInterrupt:
        console.print()
        console.print(f"[cyan]│[/]")
        console.print(f"[cyan]└[/]  [bold red]Operation cancelled[/bold red]")
        sys.exit(1)


# ============================================
# NEW: Run-based commands (from maxq3)
# ============================================


@app.command()
def worker():
    """
    Start the background job worker.

    The worker processes INDEX, EVAL, and REPORT jobs from the queue.
    Supports graceful shutdown with Ctrl+C (SIGTERM/SIGINT).
    """
    from maxq.worker.worker import run_worker

    run_worker()


@app.command("run")
def run_cmd(
    dataset: str = typer.Argument(..., help="Path to dataset JSONL file or hf://dataset_name"),
    collection: str = typer.Argument(..., help="Qdrant collection name"),
    chunk_size: int = typer.Option(800, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(120, help="Chunk overlap in characters"),
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Embedding model"),
    cloud_inference: bool = typer.Option(True, help="Use Qdrant Cloud Inference"),
):
    """
    Create and queue an index run.

    This queues a job to be processed by the worker.
    """
    from maxq.core.types import JobType, RunConfig
    from maxq.core import runs
    from maxq.db import sqlite as db
    from maxq.db.migrations import run_migrations

    run_migrations()

    # Create config
    config = RunConfig(
        collection=collection,
        dataset_path=dataset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
    )

    # Create run
    run_obj = runs.create_run(config)
    db.create_run_record(run_obj.run_id, config)

    # Create job
    payload = {"use_cloud_inference": cloud_inference}
    job_id = db.create_job(run_obj.run_id, JobType.INDEX, payload)

    console.print(f"[green]Created run: {run_obj.run_id}[/green]")
    console.print(f"[green]Queued job: {job_id}[/green]")
    console.print(f"\n[dim]Start the worker with: maxq worker[/dim]")


@app.command("status")
def status_cmd(
    run_id: str = typer.Argument(..., help="Run ID to check status for"),
):
    """
    Check run/job status.
    """
    from maxq.db import sqlite as db
    from maxq.db.migrations import run_migrations

    run_migrations()

    run_record = db.get_run(run_id)
    if not run_record:
        console.print(f"[red]Run {run_id} not found[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Run: {run_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", run_record["status"])
    table.add_row("Collection", run_record["collection"])
    table.add_row("Dataset", run_record["dataset_path"])
    table.add_row("Chunk Size", str(run_record["chunk_size"]))
    table.add_row("Model", run_record["embedding_model"])
    table.add_row("Created", run_record["created_at"])
    if run_record["error"]:
        table.add_row("Error", run_record["error"])

    console.print(table)

    # Show jobs
    jobs = db.get_jobs_for_run(run_id)
    if jobs:
        console.print("\n[bold]Jobs:[/bold]")
        jobs_table = Table()
        jobs_table.add_column("Job ID")
        jobs_table.add_column("Type")
        jobs_table.add_column("Status")
        jobs_table.add_column("Created")

        for job in jobs:
            status_color = {
                "queued": "yellow",
                "running": "cyan",
                "done": "green",
                "failed": "red",
                "cancelled": "dim",
            }.get(job["status"], "white")
            jobs_table.add_row(
                job["job_id"],
                job["job_type"],
                f"[{status_color}]{job['status']}[/{status_color}]",
                job["created_at"],
            )
        console.print(jobs_table)


@app.command("runs")
def runs_list(
    limit: int = typer.Option(20, help="Number of runs to show"),
):
    """
    List all runs.
    """
    from maxq.db import sqlite as db
    from maxq.db.migrations import run_migrations

    run_migrations()

    run_records = db.list_runs(limit=limit)

    if not run_records:
        console.print("[dim]No runs found[/dim]")
        return

    table = Table(title="Runs")
    table.add_column("Run ID")
    table.add_column("Status")
    table.add_column("Collection")
    table.add_column("Created")

    for record in run_records:
        status_color = {
            "queued": "yellow",
            "indexing": "cyan",
            "indexed": "green",
            "evaluating": "cyan",
            "evaluated": "green",
            "reporting": "cyan",
            "done": "bold green",
            "failed": "red",
        }.get(record["status"], "white")

        table.add_row(
            record["run_id"],
            f"[{status_color}]{record['status']}[/{status_color}]",
            record["collection"],
            record["created_at"][:19],
        )

    console.print(table)


@app.command("setup")
def setup_command():
    """
    Run the setup wizard (configure Qdrant connection).
    """
    use_studio = run_setup_wizard()
    console.print()
    console.print("[bold green]Setup complete![/bold green]")
    console.print()
    if use_studio:
        console.print("Run [cyan]maxq[/cyan] to open Studio")
    else:
        console.print("Run [cyan]maxq --help[/cyan] to see commands")


# =============================================================================
# Eval Commands (eval add, eval run, eval list)
# =============================================================================

eval_app = typer.Typer(help="Evaluation pack and run management.")
app.add_typer(eval_app, name="eval")


@eval_app.command("add")
def eval_add_command(
    source: str = typer.Argument(..., help="Path to eval pack file (JSON or JSONL)"),
    name: str = typer.Option(None, "--name", "-n", help="Name for the eval pack"),
):
    """
    Add an evaluation pack (queries + relevance judgments).

    File format (JSON):
        {"queries": [{"id": "q1", "query": "...", "relevant_doc_ids": ["doc1"]}]}

    Or JSONL (one query per line):
        {"id": "q1", "query": "...", "relevant_doc_ids": ["doc1"]}

    Examples:
        maxq eval add ./evals/my_eval.json --name my_eval
        maxq eval add ./queries.jsonl --name prod_queries
    """
    from maxq.core.evalpack import load_evalpack_from_file, save_evalpack

    print_header()

    # Use filename as name if not provided
    if not name:
        name = Path(source).stem

    try:
        pack = load_evalpack_from_file(source, name)
        save_evalpack(pack)

        console.print(f"\n[green]✓[/green] Added eval pack: [cyan]{name}[/cyan]")
        console.print(f"  Queries: {pack.metadata.num_queries}")
        console.print(f"  Source: {source}")
    except FileNotFoundError:
        console.print(f"[red]✗[/red] File not found: {source}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load eval pack: {e}")
        raise typer.Exit(1)


@eval_app.command("list")
def eval_list_command():
    """List all evaluation packs."""
    from maxq.core.evalpack import list_evalpacks

    print_header()

    packs = list_evalpacks()
    if not packs:
        console.print("[dim]No eval packs added yet.[/dim]")
        console.print("\nAdd one with: [cyan]maxq eval add <file.json> --name <name>[/cyan]")
        return

    table = Table(title="Evaluation Packs")
    table.add_column("Name", style="cyan")
    table.add_column("Queries", justify="right")
    table.add_column("Source", style="dim")
    table.add_column("Created")

    for p in packs:
        table.add_row(
            p.name,
            str(p.num_queries),
            p.source[:40] + "..." if len(p.source) > 40 else p.source,
            p.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@eval_app.command("run")
def eval_run_command(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to evaluate"),
    eval_pack: str = typer.Option(..., "--eval", "-e", help="Eval pack name"),
    preset: str = typer.Option("mxbai_bm25", "--preset", "-p", help="Search preset to use"),
    strategy: str = typer.Option(
        "hybrid", "--strategy", "-s", help="Search strategy (dense/sparse/hybrid)"
    ),
    limit: int = typer.Option(10, "--limit", "-k", help="Top-K results per query"),
    out: str = typer.Option(None, "--out", "-o", help="Output file for run artifact"),
):
    """
    Run evaluation against a collection and produce a scored run artifact.

    Examples:
        maxq eval run --collection kb --eval my_eval --preset mxbai_bm25
        maxq eval run -c kb -e my_eval --out runs/current.json
    """
    from maxq.core.evalpack import get_evalpack, get_reproducibility_metadata
    from maxq.core.eval import compute_metrics
    from maxq.core.types import QueryResult, SearchResult, Metrics, Run, RunConfig, RunStatus
    from maxq.core.runs import generate_run_id, create_run_dir, write_run_json, write_jsonl_artifact
    from maxq.search_engine import MaxQEngine, SearchRequest
    import time

    print_header()

    # Load eval pack
    pack = get_evalpack(eval_pack)
    if not pack:
        console.print(f"[red]✗[/red] Eval pack not found: {eval_pack}")
        console.print("\nAvailable packs:")
        from maxq.core.evalpack import list_evalpacks

        for p in list_evalpacks():
            console.print(f"  • {p.name}")
        raise typer.Exit(1)

    console.print(f"Eval pack: [cyan]{eval_pack}[/cyan] ({pack.metadata.num_queries} queries)")
    console.print(f"Collection: [cyan]{collection}[/cyan]")
    console.print(f"Strategy: {strategy}, Limit: {limit}")

    # Initialize engine
    try:
        engine = MaxQEngine()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    # Check collection exists
    if not engine.collection_exists(collection):
        console.print(f"[red]✗[/red] Collection not found: {collection}")
        raise typer.Exit(1)

    # Run evaluation
    console.print("\n[bold]Running evaluation...[/bold]")

    query_results = []
    latencies = []

    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating queries", total=len(pack.queries))

        for eq in pack.queries:
            start_time = time.time()

            # Search
            request = SearchRequest(query=eq.query, limit=limit, strategy=strategy)

            # We need a CollectionStrategy - create minimal one
            from maxq.search_engine import CollectionStrategy

            config = CollectionStrategy(collection_name=collection)

            try:
                results = engine.query(config, request)

                search_results = [
                    SearchResult(
                        id=str(r.id),
                        score=r.score,
                        doc_id=r.payload.get("doc_id", str(r.id)) if r.payload else str(r.id),
                        text=r.payload.get("_text", "")[:200] if r.payload else "",
                    )
                    for r in results
                ]
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Query failed: {eq.query[:30]}... - {e}")
                search_results = []

            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)

            query_results.append(
                QueryResult(
                    query_id=eq.id,
                    query=eq.query,
                    results=search_results,
                    relevant_doc_ids=eq.relevant_doc_ids,
                    relevant_ids=eq.relevant_ids,
                )
            )

            progress.advance(task)

    # Compute metrics
    metrics = compute_metrics(query_results, k_values=[5, 10, 20, 50])

    # Compute latency stats
    import statistics

    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else p50

    # Create run artifact
    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)

    run = Run(
        run_id=run_id,
        status=RunStatus.DONE,
        config=RunConfig(
            collection=collection,
            dataset_path=eval_pack,
            embedding_model=preset,
        ),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metrics=metrics,
    )

    # Add reproducibility metadata
    repro = get_reproducibility_metadata()

    # Write artifacts
    write_run_json(run)
    write_jsonl_artifact(run_id, "query_results.jsonl", [qr.model_dump() for qr in query_results])

    # Write repro metadata
    import json

    with open(run_dir / "reproducibility.json", "w") as f:
        json.dump(repro.model_dump(mode="json"), f, indent=2, default=str)

    # Copy to --out if specified
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(run_dir / "run.json", out_path)
        console.print(f"\n[dim]Run artifact: {out_path}[/dim]")

    # Print results
    console.print(f"\n[bold green]✓ Evaluation complete[/bold green]")
    console.print(f"\nRun ID: [cyan]{run_id}[/cyan]")

    console.print(f"\n[bold]Metrics:[/bold]")
    console.print(f"  NDCG@10:   {metrics.ndcg_at_k.get(10, 0):.4f}")
    console.print(f"  Recall@10: {metrics.recall_at_k.get(10, 0):.4f}")
    console.print(f"  MRR@10:    {metrics.mrr_at_k.get(10, 0):.4f}")

    console.print(f"\n[bold]Latency:[/bold]")
    console.print(f"  p50: {p50:.1f}ms")
    console.print(f"  p95: {p95:.1f}ms")

    console.print(f"\n[bold]Reproducibility:[/bold]")
    if repro.git_sha:
        dirty = " [yellow](dirty)[/yellow]" if repro.git_dirty else ""
        console.print(f"  Git: {repro.git_sha} ({repro.git_branch}){dirty}")
    console.print(f"  MaxQ: {repro.maxq_version}")

    console.print(f"\n[dim]Run directory: {run_dir}[/dim]")
    console.print(f"[dim]To save as baseline: maxq baseline set <name> --run {run_id}[/dim]")


@eval_app.command("show")
def eval_show_command(
    name: str = typer.Argument(..., help="Eval pack name"),
):
    """Show details of an eval pack."""
    from maxq.core.evalpack import get_evalpack

    print_header()

    pack = get_evalpack(name)
    if not pack:
        console.print(f"[red]✗[/red] Eval pack not found: {name}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Eval Pack: {name}[/bold]")
    console.print(f"  Queries: {pack.metadata.num_queries}")
    console.print(f"  Source: {pack.metadata.source}")
    console.print(f"  Created: {pack.metadata.created_at}")
    if pack.metadata.description:
        console.print(f"  Description: {pack.metadata.description}")

    console.print(f"\n[bold]Sample queries:[/bold]")
    for q in pack.queries[:5]:
        console.print(f"  • {q.query[:60]}...")
        console.print(f"    Relevant: {len(q.relevant_doc_ids)} docs")


@eval_app.command("remove")
def eval_remove_command(
    name: str = typer.Argument(..., help="Eval pack name to remove"),
):
    """Remove an eval pack."""
    from maxq.core.evalpack import delete_evalpack

    print_header()

    if delete_evalpack(name):
        console.print(f"[green]✓[/green] Removed eval pack: {name}")
    else:
        console.print(f"[red]✗[/red] Eval pack not found: {name}")
        raise typer.Exit(1)


# =============================================================================
# Baseline Commands (baseline set, baseline show, baseline list, baseline remove)
# =============================================================================

baseline_app = typer.Typer(help="Baseline management for regression testing.")
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("set")
def baseline_set_command(
    name: str = typer.Argument(..., help="Name for the baseline (e.g., 'main', 'production')"),
    run_id: str = typer.Option(..., "--run", "-r", help="Run ID to save as baseline"),
    description: str = typer.Option("", "--description", "-d", help="Description of the baseline"),
):
    """
    Save a run as a blessed baseline for regression testing.

    Examples:
        maxq baseline set main --run run_20251229_123456
        maxq baseline set production --run run_xxx --description "v1.0 release"
    """
    from maxq.core.baseline import save_baseline
    from maxq.core.runs import read_run_json

    print_header()

    try:
        run = read_run_json(run_id)
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Run not found: {run_id}")
        raise typer.Exit(1)

    if not run.metrics:
        console.print(f"[red]✗[/red] Run has no metrics. Run evaluation first.")
        raise typer.Exit(1)

    baseline = save_baseline(
        name=name,
        run_id=run_id,
        collection=run.config.collection,
        metrics=run.metrics,
        config=run.config.model_dump(),
        description=description,
    )

    console.print(f"\n[green]✓[/green] Saved baseline: [cyan]{name}[/cyan]")
    console.print(f"  Run: {run_id}")
    console.print(f"  Collection: {run.config.collection}")
    console.print(f"  NDCG@10: {baseline.metrics.ndcg_at_k.get(10, 0.0):.4f}")
    console.print(f"  Recall@10: {baseline.metrics.recall_at_k.get(10, 0.0):.4f}")


@baseline_app.command("list")
def baseline_list_command():
    """List all saved baselines."""
    from maxq.core.baseline import list_baselines

    print_header()

    baselines = list_baselines()
    if not baselines:
        console.print("[dim]No baselines saved yet.[/dim]")
        console.print("\nCreate one with: [cyan]maxq baseline set <name> --run <run_id>[/cyan]")
        return

    table = Table(title="Saved Baselines")
    table.add_column("Name", style="cyan")
    table.add_column("Run ID", style="dim")
    table.add_column("Collection")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Created")

    for b in baselines:
        ndcg = b.metrics.ndcg_at_k.get(10, 0.0) if b.metrics else 0.0
        table.add_row(
            b.name,
            b.run_id[:30] + "..." if len(b.run_id) > 30 else b.run_id,
            b.collection,
            f"{ndcg:.4f}",
            b.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@baseline_app.command("show")
def baseline_show_command(
    name: str = typer.Argument(..., help="Baseline name"),
):
    """Show details of a baseline."""
    from maxq.core.baseline import get_baseline

    print_header()

    baseline = get_baseline(name)
    if not baseline:
        console.print(f"[red]✗[/red] Baseline not found: {name}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Baseline: {name}[/bold]")
    console.print(f"  Run ID: {baseline.run_id}")
    console.print(f"  Collection: {baseline.collection}")
    console.print(f"  Created: {baseline.created_at}")
    if baseline.description:
        console.print(f"  Description: {baseline.description}")

    console.print(f"\n[bold]Metrics:[/bold]")
    for k in [5, 10, 20]:
        ndcg = baseline.metrics.ndcg_at_k.get(k, 0.0)
        recall = baseline.metrics.recall_at_k.get(k, 0.0)
        console.print(f"  NDCG@{k}: {ndcg:.4f}  Recall@{k}: {recall:.4f}")


@baseline_app.command("remove")
def baseline_remove_command(
    name: str = typer.Argument(..., help="Baseline name to remove"),
):
    """Remove a baseline."""
    from maxq.core.baseline import delete_baseline

    print_header()

    if delete_baseline(name):
        console.print(f"[green]✓[/green] Removed baseline: {name}")
    else:
        console.print(f"[red]✗[/red] Baseline not found: {name}")
        raise typer.Exit(1)


@app.command("ci")
def ci_command(
    run_id: str = typer.Argument(..., help="Run ID to check"),
    against: str = typer.Option(
        ..., "--against", "-a", help="Baseline name to compare against (e.g., 'main')"
    ),
    min_ndcg: float = typer.Option(None, "--min-ndcg", help="Minimum NDCG@10 threshold"),
    max_ndcg_drop: float = typer.Option(
        -0.02, "--max-ndcg-drop", help="Maximum allowed NDCG@10 drop from baseline"
    ),
    min_recall: float = typer.Option(None, "--min-recall", help="Minimum Recall@10 threshold"),
    max_recall_drop: float = typer.Option(
        -0.05, "--max-recall-drop", help="Maximum allowed Recall@10 drop from baseline"
    ),
    p95_ms_budget: float = typer.Option(
        None, "--p95-ms-budget", help="Maximum allowed P95 latency in milliseconds"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output report file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Run CI checks against a baseline. Exits non-zero on regression.

    Examples:
        maxq ci run_20251229_123456 --against main
        maxq ci run_20251229_123456 --against main --max-ndcg-drop -0.01
        maxq ci run_20251229_123456 --against main --min-ndcg 0.75
        maxq ci run_20251229_123456 --against main --p95-ms-budget 100
    """
    from maxq.core.baseline import run_ci_check, write_ci_report

    print_header()

    result = run_ci_check(
        run_id=run_id,
        baseline_name=against,
        min_ndcg_10=min_ndcg,
        max_ndcg_drop=max_ndcg_drop,
        min_recall_10=min_recall,
        max_recall_drop=max_recall_drop,
        max_p95_ms=p95_ms_budget,
    )

    if json_output:
        console.print(result.model_dump_json(indent=2))
    else:
        # Print status
        if result.passed:
            console.print(f"\n[bold green]✓ CI PASSED[/bold green]")
        else:
            console.print(f"\n[bold red]✗ CI FAILED[/bold red]")

        console.print(f"\nBaseline: [cyan]{against}[/cyan]")
        console.print(f"Run: [dim]{run_id}[/dim]")

        # Print checks
        if result.checks:
            console.print("\n[bold]Checks:[/bold]")
            for check in result.checks:
                emoji = "[green]✓[/green]" if check["passed"] else "[red]✗[/red]"
                console.print(
                    f"  {emoji} {check['check']}: {check['actual']:.4f} (threshold: {check['threshold']})"
                )

        # Print diff summary
        if result.diff:
            console.print(f"\n[bold]Diff Summary:[/bold]")
            console.print(f"  NDCG@10 delta: {result.diff.ndcg_10_delta:+.4f}")
            console.print(f"  Recall@10 delta: {result.diff.recall_10_delta:+.4f}")
            console.print(f"  Regressions: {result.diff.regressions}")
            console.print(f"  Improvements: {result.diff.improvements}")

        # Print worst regressions
        if result.diff and result.diff.worst_regressions:
            console.print(f"\n[bold]Top Regressions:[/bold]")
            for qd in result.diff.worst_regressions[:5]:
                query_short = qd.query[:40] + "..." if len(qd.query) > 40 else qd.query
                console.print(f"  • {query_short}")
                console.print(f"    NDCG: {qd.ndcg_delta:+.4f}, Recall: {qd.recall_delta:+.4f}")

    # Write report
    if output:
        report_path = Path(output)
        with open(report_path, "w") as f:
            f.write(result.markdown_report)
        console.print(f"\n[dim]Report written to: {report_path}[/dim]")
    else:
        # Write to run directory
        report_path = write_ci_report(run_id, result.markdown_report)
        console.print(f"\n[dim]Report: {report_path}[/dim]")

    # Exit with appropriate code
    if not result.passed:
        raise typer.Exit(1)


@app.command("diff")
def diff_command(
    run_a: str = typer.Argument(..., help="First run ID (or baseline:name)"),
    run_b: str = typer.Argument(..., help="Second run ID to compare"),
    worst: int = typer.Option(20, "--worst", "-w", help="Number of worst regressions to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Compare two runs and show per-query deltas.

    Examples:
        maxq diff run_20251229_111111 run_20251229_222222
        maxq diff baseline:main run_20251229_222222
        maxq diff baseline:main run_20251229_222222 --worst 10
    """
    from maxq.core.baseline import diff_runs, get_baseline

    print_header()

    # Parse run_a - could be "baseline:name" or a run_id
    baseline_name = None
    if run_a.startswith("baseline:"):
        baseline_name = run_a.split(":", 1)[1]
        baseline = get_baseline(baseline_name)
        if not baseline:
            console.print(f"[red]✗[/red] Baseline not found: {baseline_name}")
            raise typer.Exit(1)
        run_a = baseline.run_id

    diff = diff_runs(
        baseline_run_id=run_a,
        compare_run_id=run_b,
        baseline_name=baseline_name,
        worst_n=worst,
    )

    if json_output:
        console.print(diff.model_dump_json(indent=2))
        return

    # Print summary
    console.print(f"\n[bold]Comparison:[/bold]")
    if baseline_name:
        console.print(f"  Baseline: [cyan]{baseline_name}[/cyan] ({run_a[:30]}...)")
    else:
        console.print(f"  Run A: {run_a[:40]}...")
    console.print(f"  Run B: {run_b[:40]}...")

    console.print(f"\n[bold]Aggregate Deltas:[/bold]")

    # Color-code deltas
    def format_delta(val: float, invert: bool = False) -> str:
        if (val > 0 and not invert) or (val < 0 and invert):
            return f"[green]{val:+.4f}[/green]"
        elif (val < 0 and not invert) or (val > 0 and invert):
            return f"[red]{val:+.4f}[/red]"
        return f"{val:+.4f}"

    console.print(f"  NDCG@10:   {format_delta(diff.ndcg_10_delta)}")
    console.print(f"  Recall@10: {format_delta(diff.recall_10_delta)}")
    console.print(f"  MRR@10:    {format_delta(diff.mrr_10_delta)}")

    console.print(f"\n[bold]Query Analysis:[/bold]")
    console.print(f"  Total queries: {diff.total_queries}")
    console.print(f"  Regressions:   [red]{diff.regressions}[/red]")
    console.print(f"  Improvements:  [green]{diff.improvements}[/green]")
    console.print(f"  Unchanged:     {diff.unchanged}")

    # Show worst regressions
    if diff.worst_regressions:
        console.print(f"\n[bold]Top {len(diff.worst_regressions)} Regressions:[/bold]")

        table = Table()
        table.add_column("Query", style="dim", max_width=50)
        table.add_column("NDCG Δ", justify="right")
        table.add_column("Recall Δ", justify="right")
        table.add_column("Docs Removed", justify="right")

        for qd in diff.worst_regressions[:worst]:
            query_short = qd.query[:47] + "..." if len(qd.query) > 50 else qd.query
            table.add_row(
                query_short,
                f"[red]{qd.ndcg_delta:+.4f}[/red]",
                f"[red]{qd.recall_delta:+.4f}[/red]",
                str(len(qd.results_removed)),
            )

        console.print(table)

    # Show best improvements
    if diff.best_improvements:
        console.print(f"\n[bold]Top {len(diff.best_improvements)} Improvements:[/bold]")

        table = Table()
        table.add_column("Query", style="dim", max_width=50)
        table.add_column("NDCG Δ", justify="right")
        table.add_column("Recall Δ", justify="right")
        table.add_column("Docs Added", justify="right")

        for qd in diff.best_improvements[:worst]:
            query_short = qd.query[:47] + "..." if len(qd.query) > 50 else qd.query
            table.add_row(
                query_short,
                f"[green]{qd.ndcg_delta:+.4f}[/green]",
                f"[green]{qd.recall_delta:+.4f}[/green]",
                str(len(qd.results_added)),
            )

        console.print(table)


@app.command("pick")
def pick_command(
    run_ids: list[str] = typer.Argument(..., help="Run IDs to compare"),
    metric: str = typer.Option(
        "ndcg@10", "--metric", "-m", help="Metric to optimize (ndcg@10, recall@10, mrr@10)"
    ),
    constraint: str = typer.Option(
        None, "--constraint", "-c", help="Constraint (e.g., 'p95<150ms')"
    ),
):
    """
    Pick the best run from a set based on metric and constraints.

    Examples:
        maxq pick run_1 run_2 run_3 --metric ndcg@10
        maxq pick run_1 run_2 --metric recall@10 --constraint "p95<200ms"
    """
    from maxq.core.runs import read_run_json

    print_header()

    if len(run_ids) < 2:
        console.print("[red]✗[/red] Need at least 2 runs to compare")
        raise typer.Exit(1)

    # Parse metric
    metric_name = metric.lower().replace("@", "_at_")

    # Load runs and extract metrics
    runs_data = []
    for rid in run_ids:
        try:
            run = read_run_json(rid)
            if not run.metrics:
                console.print(f"[yellow]⚠[/yellow] Skipping {rid}: no metrics")
                continue

            # Extract the requested metric
            if "ndcg" in metric_name:
                k = int(metric_name.split("_")[-1]) if "_" in metric_name else 10
                value = run.metrics.ndcg_at_k.get(k, 0.0)
            elif "recall" in metric_name:
                k = int(metric_name.split("_")[-1]) if "_" in metric_name else 10
                value = run.metrics.recall_at_k.get(k, 0.0)
            elif "mrr" in metric_name:
                k = int(metric_name.split("_")[-1]) if "_" in metric_name else 10
                value = run.metrics.mrr_at_k.get(k, 0.0)
            else:
                console.print(f"[red]✗[/red] Unknown metric: {metric}")
                raise typer.Exit(1)

            runs_data.append(
                {
                    "run_id": rid,
                    "run": run,
                    "value": value,
                }
            )
        except FileNotFoundError:
            console.print(f"[yellow]⚠[/yellow] Skipping {rid}: not found")

    if not runs_data:
        console.print("[red]✗[/red] No valid runs found")
        raise typer.Exit(1)

    # Sort by metric (descending - higher is better)
    runs_data.sort(key=lambda x: x["value"], reverse=True)

    # Display results
    console.print(f"\n[bold]Runs ranked by {metric}:[/bold]\n")

    table = Table()
    table.add_column("Rank", justify="right")
    table.add_column("Run ID")
    table.add_column(metric.upper(), justify="right")
    table.add_column("Collection")

    for i, rd in enumerate(runs_data):
        style = "bold green" if i == 0 else ""
        rank = "👑 1" if i == 0 else str(i + 1)
        table.add_row(
            rank,
            rd["run_id"][:30] + "..." if len(rd["run_id"]) > 30 else rd["run_id"],
            f"{rd['value']:.4f}",
            rd["run"]["config"].collection,
            style=style,
        )

    console.print(table)

    winner = runs_data[0]
    console.print(f"\n[bold green]Winner:[/bold green] {winner['run_id']}")
    console.print(f"  {metric}: {winner['value']:.4f}")
    console.print(f"  Collection: {winner['run'].config.collection}")
    console.print(f"  Model: {winner['run'].config.embedding_model}")

    console.print(
        f"\n[dim]To save as baseline: maxq baseline <name> --run {winner['run_id']}[/dim]"
    )


# =============================================================================
# RAG Commands (rag eval, rag compare)
# =============================================================================

rag_app = typer.Typer(help="RAG pipeline evaluation and comparison.")
app.add_typer(rag_app, name="rag")


@rag_app.command("eval")
def rag_eval_command(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to search"),
    eval_pack: str = typer.Option(
        ..., "--eval", "-e", help="Eval pack name (must have expected_answer field)"
    ),
    pipeline: str = typer.Option(
        "standard", "--pipeline", "-p", help="Pipeline type (standard, speculative)"
    ),
    generator_model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="Generator model (or drafter for speculative)"
    ),
    verifier_model: str = typer.Option(
        "gpt-4o", "--verifier", "-v", help="Verifier model (speculative only)"
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of documents to retrieve"),
    num_drafts: int = typer.Option(5, "--num-drafts", help="Number of drafts (speculative only)"),
    docs_per_draft: int = typer.Option(
        2, "--docs-per-draft", help="Docs per draft (speculative only)"
    ),
    out: str = typer.Option(None, "--out", "-o", help="Output file for run artifact"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge evaluation"),
    judge_model: str = typer.Option("gpt-4o", "--judge-model", help="Model for LLM judge"),
):
    """
    Run RAG evaluation against a collection with LLM-as-Judge scoring.

    Supports two pipeline types:
    - standard: Basic retrieve-then-generate
    - speculative: Google's draft-then-verify approach (https://arxiv.org/abs/2407.08223)

    Examples:
        maxq rag eval -c docs -e qa_queries --pipeline standard
        maxq rag eval -c docs -e qa_queries --pipeline speculative --num-drafts 5
        maxq rag eval -c docs -e qa_queries --pipeline speculative --model gpt-4o-mini --verifier gpt-4o
    """
    from maxq.core.evalpack import get_evalpack
    from maxq.core.rag.pipeline import RAGMetrics
    from maxq.core.rag.standard import StandardRAG
    from maxq.core.rag.speculative import SpeculativeRAG
    from maxq.core.judge.llm_judge import LLMJudge
    from maxq.core.runs import generate_run_id, create_run_dir, write_jsonl_artifact
    from maxq.search_engine import MaxQEngine
    import statistics

    print_header()

    # Load eval pack
    pack = get_evalpack(eval_pack)
    if not pack:
        console.print(f"[red]✗[/red] Eval pack not found: {eval_pack}")
        raise typer.Exit(1)

    console.print(f"Eval pack: [cyan]{eval_pack}[/cyan] ({pack.metadata.num_queries} queries)")
    console.print(f"Collection: [cyan]{collection}[/cyan]")
    console.print(f"Pipeline: [cyan]{pipeline}[/cyan]")

    # Initialize engine/retriever
    try:
        engine = MaxQEngine()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    if not engine.collection_exists(collection):
        console.print(f"[red]✗[/red] Collection not found: {collection}")
        raise typer.Exit(1)

    # Create RAG pipeline
    if pipeline.lower() == "speculative":
        console.print(f"  Drafter: {generator_model}, Verifier: {verifier_model}")
        console.print(f"  Drafts: {num_drafts}, Docs per draft: {docs_per_draft}")
        rag_pipeline = SpeculativeRAG(
            retriever=engine.client,
            drafter_model=generator_model,
            verifier_model=verifier_model,
            collection_name=collection,
            top_k=top_k,
            num_drafts=num_drafts,
            docs_per_draft=docs_per_draft,
        )
    else:
        console.print(f"  Model: {generator_model}")
        rag_pipeline = StandardRAG(
            retriever=engine.client,
            generator_model=generator_model,
            collection_name=collection,
            top_k=top_k,
        )

    # Initialize judge
    judge = None
    if not no_judge:
        judge = LLMJudge(model=judge_model)
        console.print(f"  Judge: {judge_model}")

    # Run evaluation
    console.print("\n[bold]Running RAG evaluation...[/bold]")

    results = []
    latencies = []
    faithfulness_scores = []
    relevance_scores = []
    correctness_scores = []
    context_precision_scores = []

    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating queries", total=len(pack.queries))

        for eq in pack.queries:
            try:
                # Run RAG pipeline
                result = rag_pipeline.run(eq.query)
                latencies.append(result.total_latency_ms)

                # Get expected answer (if available in eval pack)
                expected_answer = getattr(eq, "expected_answer", None) or eq.metadata.get(
                    "expected_answer", ""
                )

                # Judge the result
                judge_results = {}
                if judge and result.answer:
                    context = "\n\n".join([d.text for d in result.retrieved_docs])

                    # Faithfulness - is answer grounded in context?
                    faith = judge.judge_faithfulness(
                        question=eq.query,
                        context=context,
                        answer=result.answer,
                    )
                    faithfulness_scores.append(faith.score)
                    judge_results["faithfulness"] = faith.score

                    # Relevance - does answer address the question?
                    rel = judge.judge_relevance(
                        question=eq.query,
                        answer=result.answer,
                    )
                    relevance_scores.append(rel.score)
                    judge_results["relevance"] = rel.score

                    # Context precision - are retrieved docs relevant?
                    ctx_prec = judge.judge_context_precision(
                        question=eq.query,
                        context=context,
                        expected_answer=expected_answer or "",
                    )
                    context_precision_scores.append(ctx_prec.score)
                    judge_results["context_precision"] = ctx_prec.score

                    # Correctness - if expected answer available
                    if expected_answer:
                        corr = judge.judge_correctness(
                            question=eq.query,
                            answer=result.answer,
                            expected_answer=expected_answer,
                        )
                        correctness_scores.append(corr.score)
                        judge_results["correctness"] = corr.score

                results.append(
                    {
                        "query_id": eq.id,
                        "query": eq.query,
                        "answer": result.answer,
                        "rationale": result.rationale,
                        "pipeline_type": result.pipeline_type,
                        "total_latency_ms": result.total_latency_ms,
                        "retrieval_latency_ms": result.retrieval_latency_ms,
                        "generation_latency_ms": result.generation_latency_ms,
                        "drafting_latency_ms": result.drafting_latency_ms,
                        "verification_latency_ms": result.verification_latency_ms,
                        "num_docs_retrieved": result.num_docs_retrieved,
                        "drafts": result.drafts,
                        "judge_scores": judge_results,
                    }
                )

            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Query failed: {eq.query[:30]}... - {e}")
                results.append(
                    {
                        "query_id": eq.id,
                        "query": eq.query,
                        "error": str(e),
                    }
                )

            progress.advance(task)

    # Compute aggregate metrics
    def safe_mean(lst: list) -> float:
        return statistics.mean(lst) if lst else 0.0

    def safe_percentile(lst: list, p: float) -> float:
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        idx = int(len(sorted_lst) * p / 100)
        return sorted_lst[min(idx, len(sorted_lst) - 1)]

    metrics = RAGMetrics(
        faithfulness=safe_mean(faithfulness_scores),
        relevance=safe_mean(relevance_scores),
        correctness=safe_mean(correctness_scores),
        context_precision=safe_mean(context_precision_scores),
        latency_p50_ms=safe_percentile(latencies, 50),
        latency_p95_ms=safe_percentile(latencies, 95),
        latency_p99_ms=safe_percentile(latencies, 99),
        latency_mean_ms=safe_mean(latencies),
        total_queries=len(pack.queries),
        successful_queries=len([r for r in results if "error" not in r]),
        failed_queries=len([r for r in results if "error" in r]),
    )

    # Create run artifact
    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)

    run_data = {
        "run_id": run_id,
        "type": "rag_eval",
        "pipeline": pipeline,
        "collection": collection,
        "eval_pack": eval_pack,
        "config": {
            "pipeline": pipeline,
            "generator_model": generator_model,
            "verifier_model": verifier_model if pipeline == "speculative" else None,
            "top_k": top_k,
            "num_drafts": num_drafts if pipeline == "speculative" else None,
            "docs_per_draft": docs_per_draft if pipeline == "speculative" else None,
            "judge_model": judge_model if not no_judge else None,
        },
        "metrics": metrics.model_dump(),
        "created_at": datetime.now().isoformat(),
    }

    # Write artifacts
    with open(run_dir / "run.json", "w") as f:
        json.dump(run_data, f, indent=2, default=str)

    write_jsonl_artifact(run_id, "rag_results.jsonl", results)

    # Copy to --out if specified
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        console.print(f"\n[dim]Run artifact: {out_path}[/dim]")

    # Print results
    console.print(f"\n[bold green]✓ RAG Evaluation complete[/bold green]")
    console.print(f"\nRun ID: [cyan]{run_id}[/cyan]")
    console.print(f"Pipeline: [cyan]{pipeline}[/cyan]")

    console.print(f"\n[bold]Answer Quality (LLM-judged):[/bold]")
    console.print(f"  Faithfulness:      {metrics.faithfulness:.3f}")
    console.print(f"  Relevance:         {metrics.relevance:.3f}")
    if correctness_scores:
        console.print(f"  Correctness:       {metrics.correctness:.3f}")
    console.print(f"  Context Precision: {metrics.context_precision:.3f}")

    console.print(f"\n[bold]Latency:[/bold]")
    console.print(f"  p50: {metrics.latency_p50_ms:.1f}ms")
    console.print(f"  p95: {metrics.latency_p95_ms:.1f}ms")
    console.print(f"  Mean: {metrics.latency_mean_ms:.1f}ms")

    console.print(f"\n[bold]Queries:[/bold]")
    console.print(f"  Total: {metrics.total_queries}")
    console.print(f"  Success: {metrics.successful_queries}")
    console.print(f"  Failed: {metrics.failed_queries}")

    console.print(f"\n[dim]Run directory: {run_dir}[/dim]")
    console.print(f"[dim]To compare: maxq rag compare <run_id_1> {run_id}[/dim]")


@rag_app.command("compare")
def rag_compare_command(
    run_a: str = typer.Argument(..., help="First RAG run ID (or path to run.json)"),
    run_b: str = typer.Argument(..., help="Second RAG run ID to compare"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Compare two RAG evaluation runs.

    Examples:
        maxq rag compare run_20251229_111111 run_20251229_222222
        maxq rag compare ./runs/standard.json ./runs/speculative.json
    """
    from maxq.core.runs import get_run_dir

    print_header()

    # Load runs
    def load_rag_run(run_ref: str) -> dict:
        # Check if it's a file path
        if os.path.exists(run_ref):
            with open(run_ref) as f:
                return json.load(f)
        # Otherwise, treat as run ID
        run_dir = get_run_dir(run_ref)
        run_file = run_dir / "run.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run not found: {run_ref}")
        with open(run_file) as f:
            return json.load(f)

    try:
        data_a = load_rag_run(run_a)
        data_b = load_rag_run(run_b)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    # Validate they're RAG runs
    if data_a.get("type") != "rag_eval" or data_b.get("type") != "rag_eval":
        console.print("[yellow]⚠[/yellow] One or both runs are not RAG evaluation runs")

    metrics_a = data_a.get("metrics", {})
    metrics_b = data_b.get("metrics", {})

    # Compute deltas
    comparison = {
        "run_a": {
            "run_id": data_a.get("run_id"),
            "pipeline": data_a.get("pipeline"),
            "collection": data_a.get("collection"),
        },
        "run_b": {
            "run_id": data_b.get("run_id"),
            "pipeline": data_b.get("pipeline"),
            "collection": data_b.get("collection"),
        },
        "deltas": {
            "faithfulness": metrics_b.get("faithfulness", 0) - metrics_a.get("faithfulness", 0),
            "relevance": metrics_b.get("relevance", 0) - metrics_a.get("relevance", 0),
            "correctness": metrics_b.get("correctness", 0) - metrics_a.get("correctness", 0),
            "context_precision": metrics_b.get("context_precision", 0)
            - metrics_a.get("context_precision", 0),
            "latency_p50_ms": metrics_b.get("latency_p50_ms", 0)
            - metrics_a.get("latency_p50_ms", 0),
            "latency_p95_ms": metrics_b.get("latency_p95_ms", 0)
            - metrics_a.get("latency_p95_ms", 0),
            "latency_mean_ms": metrics_b.get("latency_mean_ms", 0)
            - metrics_a.get("latency_mean_ms", 0),
        },
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
    }

    if json_output:
        console.print(json.dumps(comparison, indent=2))
        return

    # Print comparison
    console.print(f"\n[bold]RAG Pipeline Comparison[/bold]\n")

    # Pipeline info table
    info_table = Table(title="Runs", show_header=True)
    info_table.add_column("", style="bold")
    info_table.add_column("Run A", style="cyan")
    info_table.add_column("Run B", style="green")

    info_table.add_row("Run ID", data_a.get("run_id", "?")[:30], data_b.get("run_id", "?")[:30])
    info_table.add_row("Pipeline", data_a.get("pipeline", "?"), data_b.get("pipeline", "?"))
    info_table.add_row("Collection", data_a.get("collection", "?"), data_b.get("collection", "?"))

    config_a = data_a.get("config", {})
    config_b = data_b.get("config", {})
    info_table.add_row(
        "Model", config_a.get("generator_model", "?"), config_b.get("generator_model", "?")
    )

    console.print(info_table)

    # Metrics comparison
    console.print(f"\n[bold]Quality Metrics (higher is better):[/bold]")

    def format_delta(val: float, invert: bool = False) -> str:
        """Format delta with color. invert=True means lower is better."""
        if (val > 0.01 and not invert) or (val < -0.01 and invert):
            return f"[green]{val:+.3f}[/green]"
        elif (val < -0.01 and not invert) or (val > 0.01 and invert):
            return f"[red]{val:+.3f}[/red]"
        return f"{val:+.3f}"

    quality_table = Table(show_header=True)
    quality_table.add_column("Metric", style="bold")
    quality_table.add_column("Run A", justify="right")
    quality_table.add_column("Run B", justify="right")
    quality_table.add_column("Delta", justify="right")

    quality_metrics = ["faithfulness", "relevance", "correctness", "context_precision"]
    for metric in quality_metrics:
        val_a = metrics_a.get(metric, 0)
        val_b = metrics_b.get(metric, 0)
        delta = comparison["deltas"].get(metric, 0)
        quality_table.add_row(
            metric.replace("_", " ").title(),
            f"{val_a:.3f}",
            f"{val_b:.3f}",
            format_delta(delta),
        )

    console.print(quality_table)

    # Latency comparison
    console.print(f"\n[bold]Latency (lower is better):[/bold]")

    latency_table = Table(show_header=True)
    latency_table.add_column("Metric", style="bold")
    latency_table.add_column("Run A", justify="right")
    latency_table.add_column("Run B", justify="right")
    latency_table.add_column("Delta", justify="right")

    latency_metrics = ["latency_p50_ms", "latency_p95_ms", "latency_mean_ms"]
    for metric in latency_metrics:
        val_a = metrics_a.get(metric, 0)
        val_b = metrics_b.get(metric, 0)
        delta = comparison["deltas"].get(metric, 0)
        label = metric.replace("latency_", "").replace("_ms", "").upper()
        latency_table.add_row(
            label,
            f"{val_a:.1f}ms",
            f"{val_b:.1f}ms",
            format_delta(delta, invert=True),  # Lower latency is better
        )

    console.print(latency_table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")

    quality_delta = (
        comparison["deltas"]["faithfulness"]
        + comparison["deltas"]["relevance"]
        + comparison["deltas"]["context_precision"]
    ) / 3

    latency_delta = comparison["deltas"]["latency_mean_ms"]

    if quality_delta > 0.01 and latency_delta < 0:
        console.print(
            f"  [bold green]✓ Run B is better[/bold green] (higher quality, lower latency)"
        )
    elif quality_delta > 0.01:
        console.print(f"  [green]✓ Run B has higher quality[/green] (+{quality_delta:.3f} avg)")
        if latency_delta > 50:
            console.print(f"  [yellow]⚠ But {latency_delta:.0f}ms slower[/yellow]")
    elif quality_delta < -0.01:
        console.print(f"  [yellow]⚠ Run B has lower quality[/yellow] ({quality_delta:.3f} avg)")
    else:
        console.print(f"  Quality is similar between runs")

    if latency_delta < -50:
        console.print(f"  [green]✓ Run B is {-latency_delta:.0f}ms faster[/green]")
    elif latency_delta > 50:
        console.print(f"  [yellow]⚠ Run B is {latency_delta:.0f}ms slower[/yellow]")


@rag_app.command("list")
def rag_list_command(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs to show"),
):
    """List recent RAG evaluation runs."""
    from maxq.core.runs import RUNS_DIR

    print_header()

    if not RUNS_DIR.exists():
        console.print("[dim]No runs yet.[/dim]")
        return

    # Find all RAG runs
    rag_runs = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        if run_file.exists():
            try:
                with open(run_file) as f:
                    data = json.load(f)
                if data.get("type") == "rag_eval":
                    rag_runs.append(data)
                    if len(rag_runs) >= limit:
                        break
            except Exception:
                continue

    if not rag_runs:
        console.print("[dim]No RAG evaluation runs found.[/dim]")
        console.print("\nRun one with: [cyan]maxq rag eval -c <collection> -e <eval_pack>[/cyan]")
        return

    table = Table(title="RAG Evaluation Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Pipeline")
    table.add_column("Collection")
    table.add_column("Faithfulness", justify="right")
    table.add_column("p95 Latency", justify="right")
    table.add_column("Created")

    for run in rag_runs:
        metrics = run.get("metrics", {})
        created = run.get("created_at", "?")[:16]
        table.add_row(
            run.get("run_id", "?")[:30],
            run.get("pipeline", "?"),
            run.get("collection", "?"),
            f"{metrics.get('faithfulness', 0):.3f}",
            f"{metrics.get('latency_p95_ms', 0):.0f}ms",
            created,
        )

    console.print(table)


@app.command("init")
def init_command(
    directory: str = typer.Argument(".", help="Directory to initialize (default: current)"),
    studio: bool = typer.Option(False, "--studio", "-s", help="Launch Studio GUI directly"),
    cli: bool = typer.Option(False, "--cli", help="Run CLI wizard (skip GUI prompt)"),
    qdrant_url: str = typer.Option(None, "--qdrant-url", "-u", help="Qdrant URL (non-interactive)"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Qdrant API key"),
    collection: str = typer.Option(None, "--collection", "-c", help="Collection name"),
    docker: bool = typer.Option(False, "--docker", "-d", help="Auto-start Docker Qdrant"),
    scaffold: bool = typer.Option(False, "--scaffold", help="Create project files (.env.example, CI workflow)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """
    Initialize MaxQ - connect to Qdrant and configure your project.

    Interactive modes:
        maxq init              # Ask: Studio or CLI?
        maxq init --studio     # Launch GUI directly
        maxq init --cli        # CLI wizard

    Non-interactive (for AI/automation):
        maxq init --qdrant-url URL --collection NAME
        maxq init --docker --collection NAME
        maxq init --cli --docker

    Project scaffolding:
        maxq init --scaffold   # Create .env.example, CI workflow, sample evals
    """
    from pathlib import Path
    from maxq.core.onboarding import run_init

    print_header()

    target_dir = Path(directory).resolve()

    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"[green]Created directory:[/green] {target_dir}")

    # Change to target directory
    import os
    original_dir = os.getcwd()
    os.chdir(target_dir)

    try:
        # Run the onboarding wizard (or non-interactive setup)
        success = run_init(
            studio=studio,
            cli=cli,
            qdrant_url=qdrant_url,
            api_key=api_key,
            collection=collection,
            docker=docker,
        )

        # If --scaffold, also create project files
        if scaffold or (not studio and not cli and not qdrant_url and not docker):
            # Only scaffold if explicitly requested or no other options given
            if scaffold:
                _create_scaffold(target_dir, force)
    finally:
        os.chdir(original_dir)

    if not success and not scaffold:
        raise typer.Exit(1)


def _create_scaffold(target_dir, force: bool = False):
    """Create project scaffold files."""
    from pathlib import Path

    console.print(f"\n[bold]Creating project files in:[/bold] {target_dir}\n")

    files_created = []
    files_skipped = []

    # 1. Create .env.example
    env_example = target_dir / ".env.example"
    env_content = """# MaxQ Environment Configuration
# Copy this file to .env and fill in your API keys

# Required: Qdrant Cloud connection
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>

# Optional: OpenAI for RAG features
OPENAI_API_KEY=<your-openai-api-key>

# Optional: Linkup for web search augmented retrieval
LINKUP_API_KEY=<your-linkup-api-key>
"""

    if env_example.exists() and not force:
        files_skipped.append(".env.example")
    else:
        with open(env_example, "w") as f:
            f.write(env_content)
        files_created.append(".env.example")

    # 2. Create evals directory and sample eval pack
    evals_dir = target_dir / "evals"
    evals_dir.mkdir(exist_ok=True)

    sample_eval = evals_dir / "sample_queries.json"
    sample_eval_content = """{
  "name": "sample_eval",
  "description": "Sample evaluation pack - replace with your own queries",
  "queries": [
    {
      "id": "q1",
      "query": "What is vector search?",
      "relevant_doc_ids": ["doc_vector_search_intro"],
      "relevant_ids": []
    },
    {
      "id": "q2", 
      "query": "How do embeddings work?",
      "relevant_doc_ids": ["doc_embeddings_explained"],
      "relevant_ids": []
    },
    {
      "id": "q3",
      "query": "Best practices for semantic search",
      "relevant_doc_ids": ["doc_semantic_best_practices", "doc_search_optimization"],
      "relevant_ids": []
    }
  ],
  "metadata": {
    "created_by": "maxq init",
    "notes": "Replace these queries with real queries from your domain"
  }
}
"""

    if sample_eval.exists() and not force:
        files_skipped.append("evals/sample_queries.json")
    else:
        with open(sample_eval, "w") as f:
            f.write(sample_eval_content)
        files_created.append("evals/sample_queries.json")

    # 3. Create GitHub Actions workflow
    workflows_dir = target_dir / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    ci_workflow = workflows_dir / "maxq-ci.yml"
    ci_workflow_content = """# MaxQ CI - Vector Search Regression Testing
# This workflow runs on PRs that modify search-related code

name: MaxQ CI

on:
  pull_request:
    paths:
      - 'src/**'
      - 'config/**'
      - 'evals/**'
      - '.github/workflows/maxq-ci.yml'

jobs:
  maxq-regression:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install MaxQ
        run: pip install maxq
      
      - name: Run MaxQ evaluation
        env:
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
        run: |
          # Run evaluation against your collection
          maxq eval run \\
            --collection ${{ vars.MAXQ_COLLECTION || 'my_collection' }} \\
            --eval sample_eval \\
            --out runs/pr_${{ github.event.pull_request.number }}.json
      
      - name: CI Gate Check
        env:
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
        run: |
          # Compare against production baseline
          maxq ci runs/pr_${{ github.event.pull_request.number }}.json \\
            --against production \\
            --max-ndcg-drop -0.02 \\
            --max-recall-drop -0.05 \\
            --output maxq_report.md
      
      - name: Comment PR with results
        uses: actions/github-script@v7
        if: always()
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('maxq_report.md', 'utf8');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: report
            });
"""

    if ci_workflow.exists() and not force:
        files_skipped.append(".github/workflows/maxq-ci.yml")
    else:
        with open(ci_workflow, "w") as f:
            f.write(ci_workflow_content)
        files_created.append(".github/workflows/maxq-ci.yml")

    # 4. Create .gitignore additions (append if exists)
    gitignore = target_dir / ".gitignore"
    gitignore_additions = """
# MaxQ
.env
runs/
*.maxq.json
"""

    if gitignore.exists():
        # Check if already contains maxq entries
        existing = gitignore.read_text()
        if "# MaxQ" not in existing:
            with open(gitignore, "a") as f:
                f.write(gitignore_additions)
            files_created.append(".gitignore (updated)")
        else:
            files_skipped.append(".gitignore (already has MaxQ entries)")
    else:
        with open(gitignore, "w") as f:
            f.write(gitignore_additions.strip() + "\n")
        files_created.append(".gitignore")

    # Print results
    if files_created:
        console.print("[bold green]Created files:[/bold green]")
        for f in files_created:
            console.print(f"  [green]+[/green] {f}")

    if files_skipped:
        console.print("\n[bold yellow]Skipped (already exist):[/bold yellow]")
        for f in files_skipped:
            console.print(f"  [yellow]-[/yellow] {f}")
        console.print("[dim]Use --force to overwrite[/dim]")

    # Print next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Copy .env.example to .env and add your API keys")
    console.print("  2. Replace evals/sample_queries.json with your actual eval queries")
    console.print("  3. Add QDRANT_URL and QDRANT_API_KEY to your GitHub secrets")
    console.print("  4. Run [cyan]maxq doctor[/cyan] to verify your setup")
    console.print("\n[dim]Documentation: https://github.com/thierrypdamiba/maxq-project[/dim]")


# =============================================================================
# TEST COMMAND - Declarative YAML-based testing
# =============================================================================


@app.command("test")
def test_command(
    config: str = typer.Option("maxq.yaml", "--config", "-c", help="Path to maxq.yaml config"),
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Filter by tag (can specify multiple)"),
    query: str = typer.Option(None, "--query", "-q", help="Filter tests by query substring"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, github"),
    ci: bool = typer.Option(False, "--ci", help="CI mode - exit with non-zero on failures"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed results"),
):
    """
    Run declarative tests defined in maxq.yaml.

    Executes search queries and validates results against assertions like
    NDCG thresholds, recall requirements, latency bounds, and more.

    Examples:
        maxq test                         # Run all tests in maxq.yaml
        maxq test -c tests/search.yaml    # Use custom config
        maxq test --tag smoke             # Run only smoke tests
        maxq test --query "hiking"        # Run tests matching query
        maxq test --ci                    # CI mode (exit 1 on failure)
    """
    from pathlib import Path
    from rich.table import Table
    import json as json_lib

    print_header()

    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        console.print("\nTo create a config file, run:")
        console.print("  [cyan]maxq test init[/cyan]")
        raise typer.Exit(1)

    console.print(f"[bold]Running tests from:[/bold] {config_path}\n")

    try:
        from maxq.core.testconfig import load_config
        from maxq.core.runner import TestRunner

        cfg = load_config(config_path)
        runner = TestRunner(cfg)

        filter_tags = list(tag) if tag else None
        result = runner.run(filter_tags=filter_tags, filter_query=query)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error running tests:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

    # Output results
    if output == "json":
        # JSON output
        output_data = {
            "run_id": result.run_id,
            "passed": result.passed,
            "summary": result.summary,
            "tests": [
                {
                    "query": t.query,
                    "passed": t.passed,
                    "latency_ms": t.latency_ms,
                    "error": t.error,
                    "assertions": [
                        {
                            "type": a.assertion_type,
                            "passed": a.passed,
                            "message": a.message,
                        }
                        for a in t.assertion_results
                    ],
                }
                for t in result.tests
            ],
        }
        console.print(json_lib.dumps(output_data, indent=2))

    elif output == "github":
        # GitHub Actions format
        for t in result.tests:
            for a in t.assertion_results:
                if not a.passed:
                    console.print(f"::error::Test '{t.query}' failed: {a.message}")
        if not result.passed:
            console.print(f"::error::MaxQ tests failed: {result.failed_tests}/{result.total_tests} tests failed")

    else:
        # Table output (default)
        for test_result in result.tests:
            status = "[green]✓ PASS[/green]" if test_result.passed else "[red]✗ FAIL[/red]"

            # Test header
            console.print(f"\n{status} [bold]{test_result.query}[/bold]")

            if test_result.error:
                console.print(f"  [red]Error:[/red] {test_result.error}")
                continue

            console.print(f"  [dim]Latency: {test_result.latency_ms:.0f}ms | Results: {len(test_result.results)}[/dim]")

            # Show assertions
            if verbose or not test_result.passed:
                for assertion in test_result.assertion_results:
                    icon = "[green]✓[/green]" if assertion.passed else "[red]✗[/red]"
                    console.print(f"  {icon} {assertion.message}")

        # Summary table
        console.print("\n")
        summary_table = Table(title="Summary", show_header=False, box=None)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value")

        summary_table.add_row("Tests", f"{result.passed_tests}/{result.total_tests} passed")
        summary_table.add_row("Assertions", f"{result.passed_assertions}/{result.total_assertions} passed")
        summary_table.add_row(
            "Pass Rate",
            f"[green]{result.summary['pass_rate'] * 100:.1f}%[/green]"
            if result.passed
            else f"[red]{result.summary['pass_rate'] * 100:.1f}%[/red]"
        )
        summary_table.add_row("Duration", f"{result.summary['duration_ms']:.0f}ms")

        console.print(summary_table)

        # Final status
        if result.passed:
            console.print("\n[bold green]All tests passed![/bold green]")
        else:
            console.print(f"\n[bold red]{result.failed_tests} test(s) failed[/bold red]")

    # Exit with appropriate code
    if ci and not result.passed:
        raise typer.Exit(1)


@app.command("test-init")
def test_init_command(
    directory: str = typer.Argument(".", help="Directory to create maxq.yaml in"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing maxq.yaml"),
):
    """
    Create a starter maxq.yaml configuration file.

    The generated file includes example test cases with various assertion types
    that you can customize for your collection.

    Examples:
        maxq test-init             # Create maxq.yaml in current directory
        maxq test-init ./tests     # Create in tests/ directory
        maxq test-init --force     # Overwrite existing file
    """
    from pathlib import Path
    from maxq.core.testconfig import generate_example_config

    print_header()

    target_dir = Path(directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    config_file = target_dir / "maxq.yaml"

    if config_file.exists() and not force:
        console.print(f"[yellow]File already exists:[/yellow] {config_file}")
        console.print("[dim]Use --force to overwrite[/dim]")
        raise typer.Exit(1)

    config_content = generate_example_config()
    with open(config_file, "w") as f:
        f.write(config_content)

    console.print(f"[green]Created:[/green] {config_file}\n")
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Edit maxq.yaml with your collection name and test queries")
    console.print("  2. Add ground_truth document IDs for metric assertions")
    console.print("  3. Run [cyan]maxq test[/cyan] to execute the tests")
    console.print("\n[dim]Available assertion types:[/dim]")
    console.print("  not-empty, contains-id, count, latency")
    console.print("  ndcg, mrr, recall, precision, hit-rate")
    console.print("  contains-text, regex, field-equals, field-range")
    console.print("  llm-relevance, llm-rubric (require OPENAI_API_KEY)")
    console.print("  semantic-similarity, semantic-diversity")


@app.command("test-validate")
def test_validate_command(
    config: str = typer.Option("maxq.yaml", "--config", "-c", help="Path to maxq.yaml config"),
):
    """
    Validate a maxq.yaml configuration file.

    Checks for syntax errors, missing required fields, and invalid assertion types.

    Examples:
        maxq test-validate
        maxq test-validate -c custom.yaml
    """
    from pathlib import Path
    from maxq.core.testconfig import validate_config

    print_header()

    config_path = Path(config)

    is_valid, errors = validate_config(config_path)

    if is_valid:
        console.print(f"[green]✓ Valid:[/green] {config_path}")

        # Also show summary
        from maxq.core.testconfig import load_config
        cfg = load_config(config_path)
        console.print(f"\n[dim]Collection:[/dim] {cfg.provider.collection}")
        console.print(f"[dim]Model:[/dim] {cfg.provider.model}")
        console.print(f"[dim]Tests:[/dim] {len(cfg.tests)}")

        total_assertions = sum(len(t.assertions) for t in cfg.tests)
        console.print(f"[dim]Assertions:[/dim] {total_assertions}")
    else:
        console.print(f"[red]✗ Invalid:[/red] {config_path}\n")
        for error in errors:
            console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(1)


# ============================================
# Chat Command
# ============================================


@app.command()
def chat(
    collection: str = typer.Option(None, "--collection", "-c", help="Collection to chat with"),
    point_id: str = typer.Option(None, "--point-id", "-p", help="Specific point ID to examine"),
    message: str = typer.Option(None, "--message", "-m", help="Single message (non-interactive)"),
):
    """Chat with your Qdrant clusters, collections, or points."""
    from maxq.chat import ChatAgent, ChatRequest, ChatScope

    print_header()

    try:
        engine = MaxQEngine()
    except Exception as e:
        console.print(f"[red]✗ Failed to connect:[/red] {e}")
        raise typer.Exit(1)

    agent = ChatAgent(engine)

    # Determine scope
    if point_id and collection:
        scope = ChatScope.POINT
        console.print(f"[bold cyan]Chat mode:[/bold cyan] Point [dim]{point_id}[/dim] in [dim]{collection}[/dim]\n")
    elif collection:
        scope = ChatScope.COLLECTION
        console.print(f"[bold cyan]Chat mode:[/bold cyan] Collection [dim]{collection}[/dim]\n")
    else:
        scope = ChatScope.CLUSTER
        console.print("[bold cyan]Chat mode:[/bold cyan] Cluster overview\n")

    # Single message mode
    if message:
        req = ChatRequest(message=message, scope=scope, collection_name=collection, point_id=point_id)
        for token in agent.chat(req):
            console.print(token, end="")
        console.print()
        return

    # Interactive mode
    console.print("[dim]Type 'exit' or 'quit' to leave. Press Enter to send.[/dim]\n")
    history = []

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break

        if not user_input.strip():
            continue

        from maxq.chat import ChatMessage
        req = ChatRequest(
            message=user_input,
            scope=scope,
            collection_name=collection,
            point_id=point_id,
            history=history,
        )

        console.print("[bold green]MaxQ[/bold green] ", end="")
        response_text = ""
        for token in agent.chat(req):
            console.print(token, end="")
            response_text += token
        console.print("\n")

        history.append(ChatMessage(role="user", content=user_input))
        history.append(ChatMessage(role="assistant", content=response_text))

    console.print("\n[dim]Chat ended.[/dim]")


# ============================================
# Cleanup Command
# ============================================


@app.command()
def cleanup(
    collection: str = typer.Option(None, "--collection", "-c", help="Specific collection to analyze"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Preview vs execute actions"),
    duplicates: bool = typer.Option(False, "--duplicates", "-d", help="Also scan for duplicate points"),
):
    """Analyze and clean up your Qdrant cluster."""
    from maxq.cleanup import CleanupAgent

    print_header()

    try:
        engine = MaxQEngine()
    except Exception as e:
        console.print(f"[red]✗ Failed to connect:[/red] {e}")
        raise typer.Exit(1)

    agent = CleanupAgent(engine)

    console.print("[bold]Analyzing cluster...[/bold]\n")
    report = agent.analyze(collection)

    # Display collection stats
    table = Table(title="Collections", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan")
    table.add_column("Points", justify="right")
    table.add_column("Vectors", justify="right")
    table.add_column("Segments", justify="right")
    table.add_column("Status")

    for col in report.collections:
        status_style = "green" if col.status == "green" else "yellow" if col.status == "yellow" else "red" if col.error else "white"
        table.add_row(
            col.name,
            str(col.points_count),
            str(col.vectors_count),
            str(col.segments_count),
            f"[{status_style}]{col.error or col.status}[/{status_style}]",
        )

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {report.total_collections} collections, {report.total_points:,} points")

    # Empty collections
    if report.empty_collections:
        console.print(f"\n[yellow]⚠ Empty collections ({len(report.empty_collections)}):[/yellow]")
        for name in report.empty_collections:
            console.print(f"  • {name}")

    # Stale collections
    if report.stale_collections:
        console.print(f"\n[yellow]⚠ Potentially stale collections ({len(report.stale_collections)}):[/yellow]")
        for name in report.stale_collections:
            console.print(f"  • {name}")

    # Duplicates
    if duplicates:
        console.print("\n[bold]Scanning for duplicates...[/bold]")
        for col in report.collections:
            if col.points_count > 0:
                dups = agent.find_duplicates(col.name)
                if dups:
                    console.print(f"\n[yellow]⚠ {len(dups)} duplicate groups in '{col.name}':[/yellow]")
                    for d in dups[:5]:
                        console.print(f"  • {len(d.point_ids)} copies: {d.sample_text[:80]}...")

    # Suggested actions
    if report.suggested_actions:
        console.print(f"\n[bold]Suggested Actions ({len(report.suggested_actions)}):[/bold]")
        for action in report.suggested_actions:
            console.print(f"  • [cyan]{action.action}[/cyan] → {action.target}: {action.reason}")

        if dry_run:
            console.print("\n[dim]Run with --execute to apply these actions.[/dim]")
        else:
            console.print("\n[bold red]Executing cleanup actions...[/bold red]")
            results = agent.execute(report.suggested_actions, dry_run=False)
            for r in results:
                style = "green" if r["status"] == "completed" else "red"
                console.print(f"  [{style}]{r['status']}[/{style}] {r['message']}")

    # LLM summary
    try:
        summary = agent.summarize_with_llm(report)
        if summary:
            console.print(Panel(summary, title="AI Analysis", border_style="cyan"))
    except Exception:
        pass


if __name__ == "__main__":
    app()
