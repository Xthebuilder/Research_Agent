"""Ollama client for LLM inference."""
import logging
from typing import Optional, List, Dict, Any

import ollama
from rich.console import Console

from utils import check_ollama_running, start_ollama, load_config

console = Console()
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize Ollama client."""
        config = load_config()
        ollama_config = config.get("ollama", {})
        
        self.base_url = base_url or ollama_config.get("base_url", "http://localhost:11434")
        self.model = model or ollama_config.get("default_model", "gpt-oss:20b")
        self.timeout = ollama_config.get("timeout", 300)
        
        # Ensure Ollama is running
        if not check_ollama_running(self.base_url):
            if not start_ollama():
                raise RuntimeError("Ollama is not running and could not be started")
        
        # Validate model exists
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate that the selected model exists in Ollama."""
        try:
            models_response = ollama.list()
            # Handle different response formats
            if isinstance(models_response, dict):
                models_list = models_response.get("models", [])
            else:
                models_list = models_response
            
            model_names = []
            for model in models_list:
                if isinstance(model, dict):
                    model_names.append(model.get("name", ""))
                else:
                    model_names.append(str(model))
            
            # Check if model exists (handle model:tag format)
            model_base = self.model.split(":")[0]
            if model_names and not any(m.startswith(model_base) for m in model_names if m):
                console.print(f"[yellow]Warning: Model '{self.model}' not found. Available models: {', '.join(model_names)}[/yellow]")
                console.print(f"[yellow]Attempting to use '{self.model}' anyway...[/yellow]")
        except Exception as e:
            logger.warning(f"Could not validate model: {e}")
    
    def generate(self, prompt: str, system: Optional[str] = None, verbose: bool = False) -> str:
        """Generate text using Ollama."""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            if verbose:
                console.print(f"[dim]Generating response with model: {self.model}[/dim]")
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def generate_stream(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text with streaming (for progress indication)."""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            full_response = ""
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    console.print(content, end="", style="dim")
            
            console.print()  # New line after streaming
            return full_response
        except Exception as e:
            logger.error(f"Error generating streamed response: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")
