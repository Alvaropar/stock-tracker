"""
Sentiment scoring and metrics calculation.
Converts raw sentiment classifications into actionable metrics.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

from .visualizer import SentimentMetrics


class SentimentScorer:
    """
    Calculate sentiment scores and metrics from classified articles.
    """

    # Sentiment value mapping
    SENTIMENT_VALUES = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0
    }

    def __init__(self):
        pass

    def calculate_metrics(
        self,
        articles: List[Dict],
        sentiment_field: str = "sentiment"
    ) -> SentimentMetrics:
        """
        Calculate comprehensive sentiment metrics from classified articles.

        Args:
            articles: List of article dicts with 'datetime' and sentiment_field
            sentiment_field: Name of the field containing sentiment classification

        Returns:
            SentimentMetrics object with all calculated values
        """
        # Defensive: ensure articles is a list of dicts, not strings or other types
        if not isinstance(articles, list):
            return self._empty_metrics()
        articles = [a for a in articles if isinstance(a, dict)]
        if not articles:
            return self._empty_metrics()

        # Group articles by date
        daily_scores = self._group_by_date(articles, sentiment_field)

        # Calculate time-based metrics
        today = datetime.now().date()
        seven_days_ago = today - timedelta(days=7)
        thirty_days_ago = today - timedelta(days=30)

        # Today's metrics
        today_articles = [a for a in articles if self._get_date(a) == today]
        current_score, current_count = self._calculate_average(today_articles, sentiment_field)

        # 7-day metrics
        weekly_articles = [a for a in articles if self._get_date(a) >= seven_days_ago]
        weekly_score, weekly_count = self._calculate_average(weekly_articles, sentiment_field)

        # 30-day metrics
        monthly_articles = [a for a in articles if self._get_date(a) >= thirty_days_ago]
        monthly_score, monthly_count = self._calculate_average(monthly_articles, sentiment_field)

        # Momentum: compare recent 3 days vs previous 4 days
        momentum = self._calculate_momentum(articles, sentiment_field)

        # Dispersion (standard deviation of daily scores)
        dispersion = self._calculate_dispersion(daily_scores)

        # Volume trend
        volume_trend = self._calculate_volume_trend(daily_scores)

        return SentimentMetrics(
            current_score=current_score,
            current_count=current_count,
            weekly_score=weekly_score,
            weekly_count=weekly_count,
            monthly_score=monthly_score,
            monthly_count=monthly_count,
            momentum=momentum,
            dispersion=dispersion,
            volume_trend=volume_trend
        )

    def _empty_metrics(self) -> SentimentMetrics:
        """Return empty metrics when no data available."""
        return SentimentMetrics(
            current_score=0.0,
            current_count=0,
            weekly_score=0.0,
            weekly_count=0,
            monthly_score=0.0,
            monthly_count=0,
            momentum=0.0,
            dispersion=0.0,
            volume_trend="stable"
        )

    def _get_date(self, article: Dict) -> Optional[datetime.date]:
        """Extract date from article."""
        # Defensive: handle non-dict input
        if not isinstance(article, dict):
            return None
        dt_str = article.get("datetime", article.get("date", ""))
        if not dt_str:
            return None

        try:
            if isinstance(dt_str, datetime):
                return dt_str.date()
            elif "T" in str(dt_str):
                return datetime.fromisoformat(str(dt_str).replace("Z", "")).date()
            else:
                return datetime.strptime(str(dt_str)[:10], "%Y-%m-%d").date()
        except:
            return None

    def _group_by_date(
        self,
        articles: List[Dict],
        sentiment_field: str
    ) -> Dict[datetime.date, List[float]]:
        """Group sentiment scores by date."""
        daily = defaultdict(list)

        for article in articles:
            # Defensive: skip non-dict articles
            if not isinstance(article, dict):
                continue
            date = self._get_date(article)
            if date is None:
                continue

            sentiment = article.get(sentiment_field, "neutral").lower()
            score = self.SENTIMENT_VALUES.get(sentiment, 0.0)
            daily[date].append(score)

        return dict(daily)

    def _calculate_average(
        self,
        articles: List[Dict],
        sentiment_field: str
    ) -> Tuple[float, int]:
        """Calculate average sentiment score and count."""
        if not articles:
            return 0.0, 0

        scores = []
        for article in articles:
            # Defensive: skip non-dict articles
            if not isinstance(article, dict):
                continue
            sentiment = article.get(sentiment_field, "neutral").lower()
            score = self.SENTIMENT_VALUES.get(sentiment, 0.0)
            scores.append(score)

        if not scores:
            return 0.0, 0

        return sum(scores) / len(scores), len(scores)

    def _calculate_momentum(
        self,
        articles: List[Dict],
        sentiment_field: str
    ) -> float:
        """
        Calculate sentiment momentum.
        Compares recent period (last 3 days) vs previous period (4-7 days ago).
        """
        today = datetime.now().date()
        three_days_ago = today - timedelta(days=3)
        seven_days_ago = today - timedelta(days=7)

        # Recent articles (0-3 days)
        recent = [
            a for a in articles
            if isinstance(a, dict) and self._get_date(a) is not None and self._get_date(a) >= three_days_ago
        ]

        # Previous articles (4-7 days)
        previous = [
            a for a in articles
            if isinstance(a, dict) and self._get_date(a) is not None and
               three_days_ago > self._get_date(a) >= seven_days_ago
        ]

        recent_score, _ = self._calculate_average(recent, sentiment_field)
        previous_score, _ = self._calculate_average(previous, sentiment_field)

        return recent_score - previous_score

    def _calculate_dispersion(self, daily_scores: Dict[datetime.date, List[float]]) -> float:
        """
        Calculate sentiment dispersion (standard deviation).
        Lower dispersion = higher consensus.
        """
        # Defensive: handle non-dict input
        if not isinstance(daily_scores, dict) or not daily_scores:
            return 0.0

        # Flatten all scores
        all_scores = []
        for scores in daily_scores.values():
            all_scores.extend(scores)

        if len(all_scores) < 2:
            return 0.0

        try:
            return statistics.stdev(all_scores)
        except:
            return 0.0

    def _calculate_volume_trend(self, daily_scores: Dict[datetime.date, List[float]]) -> str:
        """
        Calculate article volume trend.
        Compares recent 3 days average vs previous week average.
        """
        # Defensive: handle non-dict input
        if not isinstance(daily_scores, dict) or not daily_scores:
            return "stable"

        today = datetime.now().date()
        sorted_dates = sorted(daily_scores.keys(), reverse=True)

        # Recent 3 days
        recent_dates = [d for d in sorted_dates if d >= today - timedelta(days=3)]
        recent_volume = sum(len(daily_scores[d]) for d in recent_dates) / max(len(recent_dates), 1)

        # Previous period (4-10 days)
        previous_dates = [
            d for d in sorted_dates
            if today - timedelta(days=10) <= d < today - timedelta(days=3)
        ]
        previous_volume = sum(len(daily_scores[d]) for d in previous_dates) / max(len(previous_dates), 1)

        if previous_volume == 0:
            return "stable"

        ratio = recent_volume / previous_volume

        if ratio > 1.3:
            return "increasing"
        elif ratio < 0.7:
            return "decreasing"
        else:
            return "stable"

    def get_recent_headlines_with_sentiment(
        self,
        articles: List[Dict],
        sentiment_field: str = "sentiment",
        max_count: int = 5
    ) -> List[Tuple[str, str]]:
        """
        Get recent headlines with their sentiment for display.

        Returns:
            List of (headline, sentiment) tuples
        """
        # Sort by datetime (newest first)
        sorted_articles = sorted(
            [a for a in articles if isinstance(a, dict)],
            key=lambda x: x.get("datetime", x.get("date", "")),
            reverse=True
        )

        results = []
        for article in sorted_articles[:max_count]:
            headline = article.get("title", article.get("headline", "No title"))
            sentiment = article.get(sentiment_field, "neutral")
            results.append((headline, sentiment))

        return results
