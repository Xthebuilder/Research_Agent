 HEAD
# Research_Agent
Local Research Agent , uses Ollama as inference engine. Embedding model for semantic matching of topics for better reports. Prints report of topic with citations in markdown file at end of operation>
# Local Research Agent

A Python-based CLI application for conducting local research using Ollama as the inference engine. The agent gathers sources, evaluates their relevance using semantic embeddings, and generates comprehensive research reports.

## Features

- 🔍 **Intelligent Source Gathering**: Asynchronously retrieves and extracts content from web sources
- 🧠 **Semantic Evaluation**: Uses local embedding models to assess source relevance
- 📊 **Quality Assurance**: Automatic quality checks ensure high-quality reports
- 🎨 **Beautiful CLI**: Rich, colored interface with progress indicators
- ⚡ **Async Operations**: Fast, efficient processing with lazy loading
- 🔄 **Retry Logic**: Robust error handling with automatic retries
- 📝 **Structured Reports**: Well-formatted markdown reports with citations

## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and configured (see [Ollama Installation](https://ollama.ai))

### Setup

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install and set up Ollama:
```bash
# Follow instructions at https://ollama.ai
# Pull a model (e.g., llama2 or mistral):
ollama pull llama2
# or
ollama pull mistral
```

4. (Optional) Download the embedding model (will auto-download on first use):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

5. Verify your setup:
```bash
python verify_setup.py
```

## Usage

### Basic Usage

```bash
python research_agent.py "quantum computing applications"
```

### With Specific Model

```bash
python research_agent.py "climate change solutions" --model mistral
```

### Verbose Mode

```bash
python research_agent.py "AI ethics" --model llama2 --verbose
```

### Regenerate Last Report

```bash
python research_agent.py --regenerate
```

## Configuration

Edit `config.json` to customize:

- **Ollama settings**: Base URL, default model, timeout
- **Research parameters**: Number of sources, similarity thresholds, timeouts
- **Embedding model**: Choose different sentence-transformer models
- **Output settings**: Report directory, log file location

### Key Configuration Options

- `source_similarity_threshold`: Minimum similarity score for sources (default: 0.6)
- `report_quality_threshold`: Minimum quality score for reports (default: 0.7)
- `min_sources`: Minimum number of sources to gather (default: 5)
- `max_source_attempts`: Maximum attempts to find sources (default: 10)

## Workflow

1. **Source Gathering**: Asynchronously retrieves 5 sources using DuckDuckGo search
2. **Content Extraction**: Extracts full content from web pages and PDFs
3. **Semantic Evaluation**: Calculates similarity scores between sources and research topic
4. **Quality Check**: Ensures sources meet relevance threshold
5. **Report Generation**: Creates structured markdown report with citations
6. **Quality Validation**: Verifies report quality against original topic

## Output

Reports are saved to `./research_reports/` directory with the naming convention:
```
{sanitized_topic}_{timestamp}.md
```

Each report includes:
- Executive Summary
- Introduction
- Findings (main body)
- Conclusion
- References/Citations

## Troubleshooting

### Ollama Not Running

The application will attempt to auto-start Ollama if it's not running. If this fails:

```bash
# Start Ollama manually
ollama serve
```

### Model Not Found

Ensure the model is pulled in Ollama:

```bash
ollama pull llama2
# or your preferred model
```

### Network Issues

The application includes retry logic for network failures. If issues persist:
- Check your internet connection
- Verify DuckDuckGo search is accessible
- Check firewall settings

## Logging

Detailed logs are written to `./research_agent.log` for debugging purposes.

## License

This project is provided as-is for research and educational purposes.
 928b0b6 (Initial commit)
