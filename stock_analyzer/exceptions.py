"""Custom exception hierarchy for Stock Analyzer."""


class StockAnalyzerError(Exception):
    """Base exception for all application errors."""


class DataFetchError(StockAnalyzerError):
    """Raised when market data cannot be retrieved."""


class AnalysisError(StockAnalyzerError):
    """Raised when the analysis pipeline fails."""


class SentimentError(StockAnalyzerError):
    """Raised when sentiment analysis fails."""


class MLError(StockAnalyzerError):
    """Raised when ML model operations fail."""


class ConfigurationError(StockAnalyzerError):
    """Raised when configuration is invalid or missing."""


class RiskLimitExceeded(StockAnalyzerError):
    """Raised when a trade would violate risk limits."""
