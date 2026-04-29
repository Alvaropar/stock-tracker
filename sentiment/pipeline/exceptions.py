"""
Custom exception hierarchy for the pipeline.

All pipeline-specific exceptions inherit from PipelineError,
making it easy to catch any pipeline issue with a single except clause.
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class ConfigError(PipelineError):
    """Invalid configuration (bad model path, unknown market, etc.)."""


class ScraperError(PipelineError):
    """Error during news scraping (network timeout, parse failure, etc.)."""

    def __init__(self, source: str, message: str, original: Exception | None = None):
        self.source = source
        self.original = original
        super().__init__(f"[{source}] {message}")


class FilterError(PipelineError):
    """Error during relevance filtering."""


class SentimentError(PipelineError):
    """Error during sentiment analysis."""


class ModelLoadError(PipelineError):
    """Error loading ML model weights (OOM, missing files, etc.)."""

    def __init__(self, model_path: str, message: str, original: Exception | None = None):
        self.model_path = model_path
        self.original = original
        super().__init__(f"Failed to load model '{model_path}': {message}")


class PriceDataError(PipelineError):
    """Error fetching price/market data."""
