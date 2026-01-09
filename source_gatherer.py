"""Source gathering module for web search and content extraction."""
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import trafilatura
from duckduckgo_search import DDGS
from PyPDF2 import PdfReader
from io import BytesIO
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from utils import load_config

console = Console()
logger = logging.getLogger(__name__)


class Source:
    """Represents a research source."""
    
    def __init__(self, url: str, title: str, content: str, similarity_score: float = 0.0):
        self.url = url
        self.title = title
        self.content = content
        self.similarity_score = similarity_score
    
    def __repr__(self) -> str:
        return f"Source(url={self.url[:50]}..., similarity={self.similarity_score:.2f})"


class SourceGatherer:
    """Gathers and extracts content from web sources."""
    
    def __init__(self):
        """Initialize source gatherer."""
        config = load_config()
        research_config = config.get("research", {})
        self.min_sources = research_config.get("min_sources", 5)
        self.max_attempts = research_config.get("max_source_attempts", 10)
        self.timeout = research_config.get("source_timeout", 30)
    
    async def _search_duckduckgo(self, topic: str, num_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo."""
        results = []
        try:
            with DDGS() as ddgs:
                # Try with different parameters to avoid rate limiting
                search_results = list(ddgs.text(
                    topic, 
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=None
                ))
                
                logger.info(f"DuckDuckGo returned {len(search_results)} raw results")
                
                count = 0
                for result in search_results:
                    if count >= num_results:
                        break
                    if result and isinstance(result, dict):
                        url = result.get("href") or result.get("url") or ""
                        title = result.get("title") or result.get("text") or "Untitled"
                        snippet = result.get("body") or result.get("snippet") or ""
                        
                        if url:
                            results.append({
                                "url": url,
                                "title": title,
                                "snippet": snippet
                            })
                            count += 1
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            raise
        return results
    
    async def _search_fallback(self, topic: str, num_results: int) -> List[Dict[str, str]]:
        """Fallback search using simple web scraping approach."""
        results = []
        try:
            # Use a simple approach: construct search URLs for common sources
            # This is a basic fallback - in production you'd use a proper API
            console.print("[yellow]Using fallback search method...[/yellow]")
            
            # Use URLs that are more likely to be extractable
            # Wikipedia pages are usually extractable
            from urllib.parse import quote
            topic_encoded = quote(topic.replace(' ', '_'))
            
            fallback_urls = [
                {
                    "url": f"https://en.wikipedia.org/wiki/{topic_encoded}",
                    "title": f"Wikipedia: {topic}",
                    "snippet": f"Information about {topic} from Wikipedia"
                },
                {
                    "url": f"https://www.britannica.com/search?query={quote(topic)}",
                    "title": f"Britannica: {topic}",
                    "snippet": f"Encyclopedia article about {topic}"
                }
            ]
            
            # Only return a couple of fallback results
            return fallback_urls[:min(2, num_results)]
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
    async def search_sources(self, topic: str, num_results: int = 10) -> List[Dict[str, str]]:
        """Search for sources using DuckDuckGo with retry logic and fallback."""
        max_retries = 2
        retry_delay = 5  # Longer delay to avoid rate limiting
        
        # Try DuckDuckGo first
        for attempt in range(max_retries):
            try:
                # Add delay between retries
                if attempt > 0:
                    console.print(f"[yellow]Waiting {retry_delay} seconds before retry...[/yellow]")
                    await asyncio.sleep(retry_delay)
                
                results = await self._search_duckduckgo(topic, num_results)
                
                if results:
                    logger.info(f"Found {len(results)} valid search results for: {topic}")
                    return results
                else:
                    logger.warning(f"No results from DuckDuckGo for: {topic} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        console.print(f"[yellow]No results, retrying in {retry_delay} seconds...[/yellow]")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "ratelimit" in error_msg or "429" in error_msg or "rate" in error_msg:
                    logger.warning(f"Rate limited by DuckDuckGo (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        console.print(f"[yellow]Rate limited. Waiting {retry_delay * 2} seconds...[/yellow]")
                        await asyncio.sleep(retry_delay * 2)
                    else:
                        console.print("[red]DuckDuckGo rate limit exceeded. Using fallback method.[/red]")
                        break
                else:
                    logger.error(f"DuckDuckGo search error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        console.print(f"[yellow]Search error: {e}[/yellow]")
        
        # If DuckDuckGo failed, try fallback
        console.print("[yellow]⚠ DuckDuckGo unavailable. Trying fallback search...[/yellow]")
        fallback_results = await self._search_fallback(topic, num_results)
        
        if fallback_results:
            console.print(f"[green]Found {len(fallback_results)} fallback sources[/green]")
            return fallback_results
        
        # Final failure
        console.print("[red]⚠ All search methods failed.[/red]")
        console.print("[yellow]This might be due to:")
        console.print("  • Rate limiting (wait 10-15 minutes)")
        console.print("  • Network connectivity issues")
        console.print("  • Temporary service unavailability[/yellow]")
        return []
    
    async def extract_content(self, url: str, session: aiohttp.ClientSession) -> Optional[Tuple[str, str]]:
        """Extract content from a URL (web page or PDF) using multiple methods."""
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                content_type = response.headers.get("Content-Type", "").lower()
                
                # Handle PDFs
                if "pdf" in content_type or url.lower().endswith(".pdf"):
                    pdf_content = await response.read()
                    try:
                        pdf_reader = PdfReader(BytesIO(pdf_content))
                        text_content = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
                        if text_content and len(text_content.strip()) > 50:
                            return (text_content, "PDF")
                    except Exception as e:
                        logger.warning(f"Failed to extract PDF content from {url}: {e}")
                    return None
                
                # Handle web pages - try multiple extraction methods
                html_content = await response.text()
                
                # Method 1: Try trafilatura (best for clean extraction)
                try:
                    extracted = trafilatura.extract(html_content, url=url, include_comments=False, include_tables=False)
                    if extracted and len(extracted.strip()) > 100:
                        logger.debug(f"Successfully extracted {len(extracted)} chars from {url} using trafilatura")
                        return (extracted, "Web")
                except Exception as e:
                    logger.debug(f"Trafilatura extraction failed for {url}: {e}")
                
                # Method 2: Try newspaper3k as fallback
                try:
                    import newspaper
                    article = newspaper.Article(url)
                    article.download(html=html_content)
                    article.parse()
                    if article.text and len(article.text.strip()) > 100:
                        content = f"{article.title}\n\n{article.text}" if article.title else article.text
                        logger.debug(f"Successfully extracted {len(content)} chars from {url} using newspaper3k")
                        return (content, "Web")
                except Exception as e:
                    logger.debug(f"Newspaper3k extraction failed for {url}: {e}")
                
                # Method 3: Try BeautifulSoup as final fallback
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        script.decompose()
                    
                    # Try to find main content
                    content = ""
                    
                    # Try article tag first
                    article = soup.find('article')
                    if article:
                        content = article.get_text(separator='\n', strip=True)
                    else:
                        # Try main tag
                        main = soup.find('main')
                        if main:
                            content = main.get_text(separator='\n', strip=True)
                        else:
                            # Try common content divs
                            for div in soup.find_all('div', class_=lambda x: x and any(
                                keyword in str(x).lower() for keyword in ['content', 'article', 'main', 'post', 'entry']
                            )):
                                text = div.get_text(separator='\n', strip=True)
                                if len(text) > len(content):
                                    content = text
                    
                    # If still no content, get body text
                    if not content or len(content.strip()) < 100:
                        body = soup.find('body')
                        if body:
                            content = body.get_text(separator='\n', strip=True)
                    
                    # Clean up content
                    if content:
                        # Remove excessive whitespace
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        content = '\n'.join(lines)
                        
                        # Get title if available
                        title = ""
                        if soup.title:
                            title = soup.title.get_text(strip=True)
                        elif soup.find('h1'):
                            title = soup.find('h1').get_text(strip=True)
                        
                        if title:
                            content = f"{title}\n\n{content}"
                        
                        if len(content.strip()) > 100:
                            logger.debug(f"Successfully extracted {len(content)} chars from {url} using BeautifulSoup")
                            return (content, "Web")
                        elif len(content.strip()) > 50:
                            # Lower threshold for some sites
                            logger.debug(f"Extracted {len(content)} chars from {url} (below 100 but acceptable)")
                            return (content, "Web")
                except Exception as e:
                    logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")
                
                # Method 4: Last resort - get metadata and any text
                try:
                    from trafilatura import extract_metadata
                    metadata = extract_metadata(html_content, url=url)
                    if metadata:
                        content = ""
                        if metadata.title:
                            content += metadata.title + "\n\n"
                        if metadata.description:
                            content += metadata.description + "\n\n"
                        if metadata.author:
                            content += f"Author: {metadata.author}\n\n"
                        if content and len(content.strip()) > 50:
                            return (content, "Web")
                except:
                    pass
                
                logger.warning(f"All extraction methods failed for {url}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting content from {url}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return None
    
    async def gather_sources(self, topic: str, verbose: bool = False, manual_urls: Optional[List[str]] = None) -> List[Source]:
        """Gather sources asynchronously with retry logic."""
        if manual_urls:
            # Use manually provided URLs
            if verbose:
                console.print(f"[cyan]Using {len(manual_urls)} manually provided URLs[/cyan]")
            search_results = [{"url": url, "title": f"Source {i+1}", "snippet": ""} 
                            for i, url in enumerate(manual_urls)]
        else:
            if verbose:
                console.print(f"[cyan]Searching for sources on: {topic}[/cyan]")
            
            # Initial search
            search_results = await self.search_sources(topic, num_results=self.max_attempts)
        
        if not search_results:
            logger.error(f"No search results returned for topic: {topic}")
            console.print("[red]No search results found[/red]")
            console.print("[yellow]This might be a temporary issue with DuckDuckGo search.[/yellow]")
            return []
        
        if verbose:
            console.print(f"[green]Found {len(search_results)} search results[/green]")
        else:
            logger.info(f"Found {len(search_results)} search results")
        
        sources: List[Source] = []
        failed_urls: List[Dict[str, str]] = []
        
        async with aiohttp.ClientSession() as session:
            # First pass: try to extract content from all results
            tasks = []
            for result in search_results[:self.max_attempts]:
                tasks.append(self._extract_with_result(result, session, verbose=verbose))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Track which search results failed
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Exception extracting from {search_results[i].get('url', 'unknown')}: {result}")
                    failed_urls.append(search_results[i])
                    continue
                if result:
                    sources.append(result)
                    if verbose:
                        console.print(f"[green]✓ Extracted content from: {result.url[:60]}...[/green]")
                else:
                    logger.warning(f"Failed to extract content from: {search_results[i].get('url', 'unknown')}")
                    failed_urls.append(search_results[i])
            
            # Retry failed URLs
            if len(sources) < self.min_sources and failed_urls:
                if verbose:
                    console.print(f"[yellow]Retrying {len(failed_urls)} failed sources...[/yellow]")
                
                retry_tasks = []
                for url_info in failed_urls[:self.max_attempts - len(sources)]:
                    retry_tasks.append(self._extract_with_result(url_info, session, verbose=verbose))
                
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                for result in retry_results:
                    if isinstance(result, Exception):
                        continue
                    if result:
                        sources.append(result)
        
        # If still not enough sources, search for more
        if len(sources) < self.min_sources:
            if verbose:
                console.print(f"[yellow]Only found {len(sources)} sources. Searching for more...[/yellow]")
            
            additional_results = await self.search_sources(
                topic, 
                num_results=(self.min_sources - len(sources)) * 2
            )
            
            # Filter out already processed URLs
            existing_urls = {s.url for s in sources}
            new_results = [r for r in additional_results if r["url"] not in existing_urls]
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for result in new_results[:self.min_sources - len(sources)]:
                    tasks.append(self._extract_with_result(result, session, verbose=verbose))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    if result:
                        sources.append(result)
                        if len(sources) >= self.min_sources:
                            break
        
        if verbose:
            console.print(f"[green]✓ Gathered {len(sources)} sources[/green]")
        else:
            logger.info(f"Successfully gathered {len(sources)} sources")
        
        # If we have some sources but less than minimum, still return them
        # (the caller can decide what to do)
        if sources:
            return sources[:self.max_attempts]  # Limit to max attempts
        else:
            logger.error("No sources could be extracted from any URLs")
            return []
    
    async def _extract_with_result(
        self, 
        result: Dict[str, str], 
        session: aiohttp.ClientSession,
        verbose: bool = False
    ) -> Optional[Source]:
        """Extract content and create Source object."""
        url = result.get("url", "")
        title = result.get("title", "Untitled")
        
        if not url:
            return None
        
        if verbose:
            console.print(f"[dim]Extracting content from: {url[:60]}...[/dim]")
        
        try:
            content_result = await self.extract_content(url, session)
            if content_result:
                content, source_type = content_result
                # Lower minimum for fallback sources or manual URLs
                min_length = 50 if "wikipedia" in url.lower() or "britannica" in url.lower() else 100
                if content and len(content.strip()) > min_length:
                    if verbose:
                        console.print(f"[green]✓ Extracted {len(content)} chars from: {title}[/green]")
                    return Source(url=url, title=title, content=content)
                else:
                    logger.debug(f"Content too short from {url}: {len(content.strip()) if content else 0} chars (min: {min_length})")
                    if verbose:
                        console.print(f"[yellow]⚠ Content too short from: {title}[/yellow]")
            else:
                if verbose:
                    console.print(f"[red]✗ Failed to extract content from: {title}[/red]")
        except Exception as e:
            logger.warning(f"Error in _extract_with_result for {url}: {e}")
            if verbose:
                console.print(f"[red]✗ Error extracting from {title}: {e}[/red]")
        
        return None
