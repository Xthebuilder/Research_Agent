"""Semantic evaluation module using sentence transformers."""
import logging
from typing import List

from sentence_transformers import SentenceTransformer
from rich.console import Console

from utils import load_config
from source_gatherer import Source

console = Console()
logger = logging.getLogger(__name__)


class SemanticEvaluator:
    """Evaluates semantic similarity using local embedding models."""
    
    def __init__(self):
        """Initialize semantic evaluator."""
        config = load_config()
        embedding_config = config.get("embedding", {})
        model_name = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        
        console.print(f"[cyan]Loading embedding model: {model_name}[/cyan]")
        try:
            self.model = SentenceTransformer(model_name)
            console.print("[green]✓ Embedding model loaded[/green]")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        try:
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            from torch.nn.functional import cosine_similarity
            similarity = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def evaluate_sources(self, topic: str, sources: List[Source], threshold: float = 0.6) -> List[Source]:
        """Evaluate sources and assign similarity scores."""
        console.print("[cyan]Evaluating source relevance...[/cyan]")
        
        # Embed the topic once
        topic_embedding = self.model.encode(topic, convert_to_tensor=True)
        
        # Calculate similarities
        for source in sources:
            try:
                # Use a combination of title and content snippet for evaluation
                source_text = f"{source.title}\n\n{source.content[:500]}"
                source_embedding = self.model.encode(source_text, convert_to_tensor=True)
                
                from torch.nn.functional import cosine_similarity
                similarity = cosine_similarity(topic_embedding.unsqueeze(0), source_embedding.unsqueeze(0)).item()
                source.similarity_score = max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.warning(f"Error evaluating source {source.url}: {e}")
                source.similarity_score = 0.0
        
        # Sort by similarity score (descending)
        sources.sort(key=lambda s: s.similarity_score, reverse=True)
        
        # Filter by threshold
        filtered_sources = [s for s in sources if s.similarity_score >= threshold]
        
        if len(filtered_sources) < len(sources):
            console.print(
                f"[yellow]Filtered out {len(sources) - len(filtered_sources)} sources "
                f"below threshold ({threshold})[/yellow]"
            )
        
        return filtered_sources
    
    def evaluate_report_quality(self, topic: str, report: str) -> float:
        """Evaluate report quality by comparing with original topic."""
        try:
            return self.calculate_similarity(topic, report)
        except Exception as e:
            logger.error(f"Error evaluating report quality: {e}")
            return 0.0
