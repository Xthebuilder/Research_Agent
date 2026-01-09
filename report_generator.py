"""Report generation module."""
import logging
from typing import List

from rich.console import Console
from rich.markdown import Markdown

from ollama_client import OllamaClient
from semantic_evaluator import SemanticEvaluator
from source_gatherer import Source
from utils import load_config

console = Console()
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates research reports using Ollama."""
    
    def __init__(self, model: str = None, verbose: bool = False):
        """Initialize report generator."""
        self.ollama_client = OllamaClient(model=model)
        self.evaluator = SemanticEvaluator()
        self.verbose = verbose
        
        config = load_config()
        research_config = config.get("research", {})
        self.quality_threshold = research_config.get("report_quality_threshold", 0.7)
        self.max_attempts = research_config.get("max_report_attempts", 3)
    
    def generate_report(self, topic: str, sources: List[Source]) -> str:
        """Generate research report from sources."""
        console.print("[cyan]Generating research report...[/cyan]")
        
        # Validate all sources have URLs
        valid_sources = []
        for source in sources:
            if not source.url or source.url.strip() == "":
                logger.warning(f"Source '{source.title}' has no URL, skipping")
                continue
            # Validate URL format
            if not (source.url.startswith("http://") or source.url.startswith("https://")):
                logger.warning(f"Source '{source.title}' has invalid URL format: {source.url}")
                continue
            valid_sources.append(source)
        
        if not valid_sources:
            raise ValueError("No valid sources with URLs found for report generation")
        
        if len(valid_sources) < len(sources):
            console.print(f"[yellow]Warning: {len(sources) - len(valid_sources)} sources had invalid URLs and were excluded[/yellow]")
        
        # Prepare source information
        source_texts = []
        citations = []
        
        for i, source in enumerate(valid_sources, 1):
            source_texts.append(f"Source {i}:\nTitle: {source.title}\nURL: {source.url}\nContent: {source.content[:2000]}\n")
            citations.append({
                "number": i,
                "title": source.title,
                "url": source.url
            })
        
        sources_content = "\n\n".join(source_texts)
        
        # Create structured prompt
        system_prompt = """You are a professional research analyst. Generate comprehensive, well-structured research reports based on provided sources. 
Your reports should be factual, well-cited, and organized."""
        
        user_prompt = f"""Research Topic: {topic}

Sources:
{sources_content}

Please generate a comprehensive research report in markdown format with the following structure:

# {topic}

## Executive Summary
[A concise summary of key findings and conclusions]

## Introduction
[Introduction to the topic and its significance]

## Findings
[Main body of the report with detailed findings, organized into subsections as appropriate. Include citations like [1], [2], etc. where you reference sources]

## Conclusion
[Summary of findings and conclusions]

## References
CRITICAL REQUIREMENTS FOR REFERENCES:
1. You MUST use the EXACT URLs provided in the sources above - do NOT make up URLs
2. Format each reference as a markdown link: [Title](URL)
3. Do NOT use placeholder text like "[Article URL]", "[Blog post URL]", "[Interview URL]", or any variation
4. Every single reference MUST have a clickable link using the actual URL from the sources
5. Match the title from the source to the URL - each source has a specific URL that must be used

Example format:
1. [Wikipedia: Vibe coding](https://en.wikipedia.org/wiki/Vibe_coding)
2. [Karpathy, A. (2025). Vibe coding introduction](https://actual-url-from-source.com/path)

Remember: Every reference number corresponds to a specific source with a specific URL. Use that exact URL."""

        # Generate report with retry logic
        for attempt in range(self.max_attempts):
            try:
                if self.verbose:
                    console.print(f"[dim]Generation attempt {attempt + 1}/{self.max_attempts}[/dim]")
                
                report = self.ollama_client.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    verbose=self.verbose
                )
                
                # Evaluate report quality
                quality_score = self.evaluator.evaluate_report_quality(topic, report)
                
                if self.verbose:
                    console.print(f"[dim]Report quality score: {quality_score:.2f} (threshold: {self.quality_threshold})[/dim]")
                
                if quality_score >= self.quality_threshold:
                    # Format citations properly
                    report = self._format_citations(report, citations)
                    # Validate all citations have URLs
                    report = self._validate_citation_urls(report, citations)
                    return report
                else:
                    if attempt < self.max_attempts - 1:
                        console.print(
                            f"[yellow]Report quality ({quality_score:.2f}) below threshold. "
                            f"Regenerating... (attempt {attempt + 2}/{self.max_attempts})[/yellow]"
                        )
                        # Enhance prompt for next attempt
                        user_prompt += "\n\nPlease provide a more comprehensive and detailed report that better addresses the research topic."
                    else:
                        console.print(
                            f"[yellow]Warning: Report quality ({quality_score:.2f}) below threshold, "
                            f"but max attempts reached. Proceeding with current report.[/yellow]"
                        )
                        report = self._format_citations(report, citations)
                        # Validate all citations have URLs
                        report = self._validate_citation_urls(report, citations)
                        return report
            except Exception as e:
                logger.error(f"Error generating report (attempt {attempt + 1}): {e}")
                if attempt == self.max_attempts - 1:
                    raise RuntimeError(f"Failed to generate report after {self.max_attempts} attempts: {e}")
        
        raise RuntimeError("Failed to generate report")
    
    def _format_citations(self, report: str, citations: List[dict]) -> str:
        """Ensure citations are properly formatted with clickable links."""
        import re
        
        # Create a mapping of titles to URLs for lookup
        title_to_url = {cite['title']: cite['url'] for cite in citations}
        # Also create reverse mapping for partial matches
        url_map = {cite['number']: cite['url'] for cite in citations}
        
        # Find the References section
        ref_pattern = r'(##\s+References\s*\n\n?)(.*?)(?=\n##|\Z)'
        match = re.search(ref_pattern, report, re.DOTALL | re.IGNORECASE)
        
        if match:
            # References section exists - replace it with properly formatted citations
            ref_header = match.group(1)
            ref_content = match.group(2)
            
            # Build new references section with markdown links
            new_refs = []
            for citation in citations:
                title = citation['title']
                url = citation['url']
                # Format as markdown link: [Title](URL)
                new_refs.append(f"{citation['number']}. [{title}]({url})")
            
            new_ref_section = ref_header + "\n".join(new_refs) + "\n"
            report = report[:match.start()] + new_ref_section + report[match.end():]
        else:
            # No References section - add one
            report += "\n\n## References\n\n"
            for citation in citations:
                # Format as markdown link
                report += f"{citation['number']}. [{citation['title']}]({citation['url']})\n"
        
        # Fix any placeholder URLs that might still exist
        # Look for patterns like "[Article URL]", "[Blog post URL]", etc.
        placeholder_patterns = [
            r'\[(?:Article|Blog post|Interview|Paper|Source|Document)\s+URL\]',
            r'\[URL\]',
            r'\[.*?URL.*?\]',
        ]
        
        for pattern in placeholder_patterns:
            # Try to match and replace with actual URLs
            # This is a fallback - the LLM should have used real URLs
            def replace_placeholder(m):
                # Try to find the citation number before this placeholder
                line = m.group(0)
                # Look for citation number in the line
                num_match = re.search(r'(\d+)\.', report[max(0, m.start()-50):m.start()])
                if num_match:
                    cite_num = int(num_match.group(1))
                    if cite_num in url_map:
                        return url_map[cite_num]
                return m.group(0)  # Return original if can't replace
            
            report = re.sub(pattern, replace_placeholder, report, flags=re.IGNORECASE)
        
        # Ensure all citations have valid URLs (validate)
        for citation in citations:
            url = citation['url']
            # Check if URL is in the report and is clickable
            if url not in report or f"]({url})" not in report:
                # Try to find and fix the citation
                cite_pattern = rf"{citation['number']}\.\s+.*?{re.escape(citation['title'])}.*?(?:-|–|:)\s*([^\n]+)"
                cite_match = re.search(cite_pattern, report, re.IGNORECASE)
                if cite_match:
                    old_url_part = cite_match.group(1).strip()
                    # Replace with markdown link if it's a placeholder
                    if '[' in old_url_part and 'URL' in old_url_part.upper():
                        new_line = f"{citation['number']}. [{citation['title']}]({url})"
                        report = report.replace(cite_match.group(0), new_line)
        
        return report
    
    def _validate_citation_urls(self, report: str, citations: List[dict]) -> str:
        """Validate and ensure all citations have clickable URLs."""
        import re
        
        # Check References section
        ref_pattern = r'(##\s+References\s*\n\n?)(.*?)(?=\n##|\Z)'
        match = re.search(ref_pattern, report, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return report
        
        ref_content = match.group(2)
        ref_header = match.group(1)
        
        # Check each citation
        new_ref_lines = []
        lines = ref_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line has a citation number
            cite_match = re.match(r'(\d+)\.\s*(.+)', line)
            if cite_match:
                cite_num = int(cite_match.group(1))
                cite_text = cite_match.group(2)
                
                # Find corresponding citation
                citation = next((c for c in citations if c['number'] == cite_num), None)
                if citation:
                    url = citation['url']
                    title = citation['title']
                    
                    # Check if URL is already in the line as a markdown link
                    if f"]({url})" in line or f"]({re.escape(url)})" in line:
                        new_ref_lines.append(line)
                    else:
                        # Check if there's a placeholder
                        if re.search(r'\[.*?URL.*?\]', line, re.IGNORECASE):
                            # Replace placeholder with actual URL
                            new_line = f"{cite_num}. [{title}]({url})"
                            new_ref_lines.append(new_line)
                            logger.info(f"Replaced placeholder URL in citation {cite_num} with actual URL")
                        elif url not in line:
                            # URL missing - add it as markdown link
                            # Try to extract title from line
                            title_match = re.search(r'\[([^\]]+)\]', line)
                            if title_match:
                                extracted_title = title_match.group(1)
                                new_line = f"{cite_num}. [{extracted_title}]({url})"
                            else:
                                # Use citation title
                                new_line = f"{cite_num}. [{title}]({url})"
                            new_ref_lines.append(new_line)
                            logger.info(f"Added missing URL to citation {cite_num}")
                        else:
                            # URL exists but might not be formatted as link
                            if not re.search(r'\[.*?\]\(.*?\)', line):
                                # Format as markdown link
                                new_line = f"{cite_num}. [{title}]({url})"
                                new_ref_lines.append(new_line)
                            else:
                                new_ref_lines.append(line)
                else:
                    new_ref_lines.append(line)
            else:
                new_ref_lines.append(line)
        
        # Rebuild References section
        new_ref_section = ref_header + "\n".join(new_ref_lines) + "\n"
        report = report[:match.start()] + new_ref_section + report[match.end():]
        
        return report
