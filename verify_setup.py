#!/usr/bin/env python3
"""Quick setup verification script."""
import sys
from rich.console import Console

console = Console()

def check_imports():
    """Check if all required packages can be imported."""
    console.print("[cyan]Checking Python package imports...[/cyan]")
    
    packages = [
        ("ollama", "ollama"),
        ("duckduckgo_search", "duckduckgo-search"),
        ("trafilatura", "trafilatura"),
        ("newspaper", "newspaper3k"),
        ("PyPDF2", "PyPDF2"),
        ("pdfplumber", "pdfplumber"),
        ("sentence_transformers", "sentence-transformers"),
        ("rich", "rich"),
        ("typer", "typer"),
        ("aiohttp", "aiohttp"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("torch", "torch"),
    ]
    
    failed = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            console.print(f"  [green]✓[/green] {package_name}")
        except ImportError:
            console.print(f"  [red]✗[/red] {package_name} (not installed)")
            failed.append(package_name)
    
    return failed

def check_ollama():
    """Check if Ollama is accessible."""
    console.print("\n[cyan]Checking Ollama connection...[/cyan]")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            console.print("  [green]✓[/green] Ollama is running")
            try:
                import ollama
                models = ollama.list()
                console.print("  [green]✓[/green] Ollama API accessible")
                return True
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] Ollama running but API error: {e}")
                return False
        else:
            console.print("  [red]✗[/red] Ollama not responding correctly")
            return False
    except Exception as e:
        console.print(f"  [red]✗[/red] Ollama not running: {e}")
        console.print("  [yellow]Install from: https://ollama.ai[/yellow]")
        return False

def check_config():
    """Check if config file exists."""
    console.print("\n[cyan]Checking configuration...[/cyan]")
    import os
    if os.path.exists("config.json"):
        console.print("  [green]✓[/green] config.json exists")
        return True
    else:
        console.print("  [red]✗[/red] config.json not found")
        return False

def main():
    """Run all checks."""
    console.print("[bold]Research Agent Setup Verification[/bold]\n")
    
    failed_packages = check_imports()
    ollama_ok = check_ollama()
    config_ok = check_config()
    
    console.print("\n[bold]Summary:[/bold]")
    
    if failed_packages:
        console.print(f"[red]Missing packages: {', '.join(failed_packages)}[/red]")
        console.print("[yellow]Install with: pip install -r requirements.txt[/yellow]")
    
    if not ollama_ok:
        console.print("[yellow]Ollama is not running. Start with: ollama serve[/yellow]")
    
    if not config_ok:
        console.print("[red]Configuration file missing![/red]")
    
    if not failed_packages and ollama_ok and config_ok:
        console.print("[green]✓ All checks passed! Setup looks good.[/green]")
        return 0
    else:
        console.print("[yellow]⚠ Some issues found. Please fix them before using the research agent.[/yellow]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
