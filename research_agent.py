#!/usr/bin/env python3
"""Main CLI application for the Local Research Agent."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from ollama_client import OllamaClient
from report_generator import ReportGenerator
from semantic_evaluator import SemanticEvaluator
from source_gatherer import SourceGatherer
from utils import (
    load_config,
    setup_logging,
    get_report_filename,
    get_latest_report,
    check_ollama_running,
    start_ollama
)

app = typer.Typer(help="Local Research Agent - AI-powered research assistant using Ollama")
console = Console()
logger = logging.getLogger(__name__)


def display_banner():
    """Display application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║         Local Research Agent - AI Research Assistant      ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def display_sources_table(sources):
    """Display sources in a formatted table."""
    table = Table(title="Gathered Sources", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("URL", style="blue", no_wrap=False)
    table.add_column("Similarity", justify="right", style="green")
    
    for i, source in enumerate(sources, 1):
        table.add_row(
            str(i),
            source.title[:60] + "..." if len(source.title) > 60 else source.title,
            source.url[:50] + "..." if len(source.url) > 50 else source.url,
            f"{source.similarity_score:.2f}"
        )
    
    console.print(table)


@app.command()
def main(
    topic: Optional[str] = typer.Argument(None, help="Research topic"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    regenerate: bool = typer.Option(False, "--regenerate", "-r", help="Regenerate last report"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated list of URLs to use as sources (bypasses search)")
):
    """Main CLI entry point."""
    # Setup
    config = load_config()
    setup_logging(config.get("output", {}).get("log_file", "./research_agent.log"))
    
    display_banner()
    
    # Handle regenerate flag
    if regenerate:
        latest_report = get_latest_report()
        if not latest_report:
            console.print("[red]No previous report found to regenerate[/red]")
            raise typer.Exit(1)
        
        console.print(f"[cyan]Regenerating report from: {latest_report.name}[/cyan]")
        
        # If topic not provided, try to extract from report or ask user
        if not topic:
            # Try to extract topic from report filename (simplified)
            report_name = latest_report.stem
            # Topic is usually before the last underscore and timestamp
            parts = report_name.rsplit("_", 2)
            if len(parts) >= 2:
                potential_topic = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
                if potential_topic:
                    console.print(f"[yellow]Inferred topic from filename: {potential_topic}[/yellow]")
                    console.print("[yellow]If this is incorrect, please provide the topic as an argument[/yellow]")
                    topic = potential_topic
                else:
                    console.print("[yellow]Please provide the original topic for regeneration:[/yellow]")
                    topic = input("Topic: ").strip()
            else:
                console.print("[yellow]Please provide the original topic for regeneration:[/yellow]")
                topic = input("Topic: ").strip()
            
            if not topic:
                console.print("[red]Topic required for regeneration[/red]")
                raise typer.Exit(1)
    
    # Validate topic
    if not topic:
        console.print("[red]Error: Research topic is required[/red]")
        console.print("[yellow]Usage: python research_agent.py 'your research topic'[/yellow]")
        raise typer.Exit(1)
    
    # Check Ollama
    if not check_ollama_running():
        if not start_ollama():
            console.print("[red]Please ensure Ollama is installed and running[/red]")
            raise typer.Exit(1)
    
    try:
        # Run async workflow
        asyncio.run(run_research_workflow(topic, model, verbose, manual_sources=sources))
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error in research workflow: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def run_research_workflow(topic: str, model: Optional[str], verbose: bool, manual_sources: Optional[str] = None):
    """Main research workflow."""
    config = load_config()
    research_config = config.get("research", {})
    similarity_threshold = research_config.get("source_similarity_threshold", 0.6)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Step 1: Gather sources
        task1 = progress.add_task("[cyan]Gathering sources...", total=None)
        gatherer = SourceGatherer()
        
        # Check if manual sources provided
        manual_urls = None
        if manual_sources:
            manual_urls = [url.strip() for url in manual_sources.split(",") if url.strip()]
            console.print(f"[cyan]Using {len(manual_urls)} manually provided sources[/cyan]")
        
        sources = await gatherer.gather_sources(topic, verbose=verbose, manual_urls=manual_urls)
        progress.update(task1, completed=True)
        
        if not sources:
            console.print()
            console.print(Panel(
                "[red]Failed to gather any sources[/red]\n\n"
                "Possible reasons:\n"
                "• Network connectivity issues\n"
                "• DuckDuckGo search temporarily unavailable\n"
                "• All sources failed to load\n\n"
                "[yellow]Solutions:[/yellow]\n"
                "1. Wait 10-15 minutes and try again (rate limiting)\n"
                "2. Use manual sources: --sources 'url1,url2,url3'\n"
                "3. Check your internet connection",
                title="Error",
                border_style="red"
            ))
            raise typer.Exit(1)
        
        if verbose:
            console.print(f"[green]✓ Gathered {len(sources)} sources[/green]")
        
        # Step 2: Evaluate sources semantically
        task2 = progress.add_task("[cyan]Evaluating source relevance...", total=None)
        evaluator = SemanticEvaluator()
        evaluated_sources = evaluator.evaluate_sources(topic, sources, threshold=similarity_threshold)
        progress.update(task2, completed=True)
        
        if not evaluated_sources:
            console.print(f"[red]No sources met the similarity threshold ({similarity_threshold})[/red]")
            raise typer.Exit(1)
        
        # Display sources
        console.print()
        display_sources_table(evaluated_sources)
        console.print()
        
        # Ask user if they want to proceed (optional - could be skipped)
        if verbose:
            console.print(f"[cyan]Proceeding with {len(evaluated_sources)} sources above threshold...[/cyan]")
        
        # Step 3: Generate report
        task3 = progress.add_task("[cyan]Generating research report...", total=None)
        generator = ReportGenerator(model=model, verbose=verbose)
        report = generator.generate_report(topic, evaluated_sources)
        progress.update(task3, completed=True)
        
        # Step 4: Save report
        output_dir = config.get("output", {}).get("directory", "./research_reports")
        report_path = get_report_filename(topic, output_dir)
        
        task4 = progress.add_task("[cyan]Saving report...", total=None)
        report_path.write_text(report, encoding="utf-8")
        progress.update(task4, completed=True)
        
        # Display results
        console.print()
        console.print(Panel(
            f"[green]✓ Research report generated successfully![/green]\n\n"
            f"Saved to: [cyan]{report_path}[/cyan]\n"
            f"Sources used: [yellow]{len(evaluated_sources)}[/yellow]",
            title="Success",
            border_style="green"
        ))
        
        # Display report preview
        console.print()
        console.print(Panel(
            Markdown(report[:1000] + "\n\n..."),
            title="Report Preview",
            border_style="blue"
        ))
        
        console.print(f"\n[dim]Full report available at: {report_path}[/dim]")


if __name__ == "__main__":
    app()
