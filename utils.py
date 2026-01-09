"""Utility functions for configuration, file management, and logging."""
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import rich
from rich.console import Console

console = Console()


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Configuration file {config_path} not found![/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing configuration file: {e}[/red]")
        sys.exit(1)


def setup_logging(log_file: str = "./research_agent.log") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def sanitize_filename(topic: str) -> str:
    """Sanitize topic string for use in filename."""
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^\w\s-]', '', topic)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized[:100]  # Limit length


def get_report_filename(topic: str, base_dir: str = "./research_reports") -> Path:
    """Generate report filename with timestamp."""
    sanitized = sanitize_filename(topic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized}_{timestamp}.md"
    
    # Create directory if it doesn't exist
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    return Path(base_dir) / filename


def get_latest_report(base_dir: str = "./research_reports") -> Optional[Path]:
    """Get the most recently created report file."""
    reports_dir = Path(base_dir)
    if not reports_dir.exists():
        return None
    
    reports = list(reports_dir.glob("*.md"))
    if not reports:
        return None
    
    return max(reports, key=lambda p: p.stat().st_mtime)


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is running."""
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def start_ollama() -> bool:
    """Attempt to start Ollama service."""
    console.print("[yellow]Ollama not running. Attempting to start...[/yellow]")
    try:
        # Try to start Ollama in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Wait a bit for it to start
        import time
        time.sleep(3)
        
        if check_ollama_running():
            console.print("[green]✓ Ollama started successfully[/green]")
            return True
        else:
            console.print("[red]Failed to start Ollama. Please start it manually: ollama serve[/red]")
            return False
    except FileNotFoundError:
        console.print("[red]Ollama not found. Please install Ollama from https://ollama.ai[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error starting Ollama: {e}[/red]")
        return False
