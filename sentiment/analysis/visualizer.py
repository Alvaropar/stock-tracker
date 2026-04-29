"""
Terminal-based sentiment visualization using Rich library.
Displays sentiment metrics in an elegant, quant-researcher style.
Includes matplotlib-based sentiment graphs.
"""
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding - set UTF-8 codepage
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.align import Align
from rich import box


@dataclass
class SentimentMetrics:
    """Container for sentiment analysis metrics."""
    current_score: float  # Today's sentiment (-1 to +1)
    current_count: int    # Today's article count
    weekly_score: float   # 7-day average
    weekly_count: int     # 7-day article count
    monthly_score: float  # 30-day average
    monthly_count: int    # 30-day article count
    momentum: float       # Score change (recent - older)
    dispersion: float     # Standard deviation (consensus measure)
    volume_trend: str     # "increasing", "stable", "decreasing"

    @property
    def signal(self) -> str:
        """Get trading signal interpretation."""
        if self.current_score > 0.5:
            return "STRONG_BULLISH"
        elif self.current_score > 0.2:
            return "BULLISH"
        elif self.current_score > -0.2:
            return "NEUTRAL"
        elif self.current_score > -0.5:
            return "BEARISH"
        else:
            return "STRONG_BEARISH"

    @property
    def signal_color(self) -> str:
        """Get color for signal."""
        signal_colors = {
            "STRONG_BULLISH": "bold green",
            "BULLISH": "green",
            "NEUTRAL": "yellow",
            "BEARISH": "red",
            "STRONG_BEARISH": "bold red"
        }
        return signal_colors.get(self.signal, "white")

    @property
    def momentum_arrow(self) -> Tuple[str, str]:
        """Get arrow indicator for momentum. Returns (arrow, color)."""
        if self.momentum > 0.1:
            return ("^^", "green")
        elif self.momentum > 0.02:
            return ("^", "green")
        elif self.momentum > -0.02:
            return ("->", "yellow")
        elif self.momentum > -0.1:
            return ("v", "red")
        else:
            return ("vv", "red")


class SentimentVisualizer:
    """
    Rich terminal-based sentiment visualization.
    Presents sentiment data in a professional quant-researcher style.
    """

    def __init__(self):
        # Use force_terminal=True for consistent output, legacy_windows=False for better Unicode
        self.console = Console(force_terminal=True, legacy_windows=False)

    def _score_to_bar(self, score: float, width: int = 20) -> Text:
        """Convert sentiment score to visual bar."""
        # Map score from [-1, 1] to [0, width]
        normalized = (score + 1) / 2
        filled = int(normalized * width)

        # Determine color based on score
        if score > 0.3:
            color = "green"
        elif score > 0:
            color = "bright_green"
        elif score > -0.3:
            color = "yellow"
        elif score > -0.5:
            color = "bright_red"
        else:
            color = "red"

        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (width - filled), style="dim")
        return bar

    def _format_score(self, score: float, include_sign: bool = True) -> Text:
        """Format a score with color coding."""
        if score > 0.3:
            color = "bold green"
        elif score > 0:
            color = "green"
        elif score > -0.3:
            color = "yellow"
        elif score > -0.5:
            color = "red"
        else:
            color = "bold red"

        if include_sign and score > 0:
            text = f"+{score:.2f}"
        else:
            text = f"{score:.2f}"

        return Text(text, style=color)

    def display_sentiment(
        self,
        asset_name: str,
        market: str,
        metrics: SentimentMetrics,
        recent_headlines: List[Tuple[str, str]] = None  # [(headline, sentiment), ...]
    ) -> None:
        """
        Display comprehensive sentiment analysis in terminal.

        Args:
            asset_name: Name of the asset (e.g., "GOLD", "600547")
            market: Market identifier ("US" or "CHINA")
            metrics: Calculated sentiment metrics
            recent_headlines: Optional list of (headline, sentiment) tuples
        """
        self.console.clear()

        # Build header
        header = self._build_header(asset_name, market)

        # Build main sentiment panel
        main_panel = self._build_main_panel(metrics)

        # Build metrics table
        metrics_table = self._build_metrics_table(metrics)

        # Build signal panel
        signal_panel = self._build_signal_panel(metrics)

        # Print everything
        self.console.print(header)
        self.console.print()
        self.console.print(main_panel)
        self.console.print()
        self.console.print(metrics_table)
        self.console.print()
        self.console.print(signal_panel)

        # Show recent headlines if provided
        if recent_headlines:
            self.console.print()
            self._display_recent_headlines(recent_headlines)

    def _build_header(self, asset_name: str, market: str) -> Panel:
        """Build the header panel."""
        # Use text labels instead of emojis for Windows compatibility
        market_label = "[US]" if market.upper() == "US" else "[CN]"
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        header_text = Text()
        header_text.append(f"{market_label} ", style="bold blue")
        header_text.append(f"{asset_name.upper()} ", style="bold white")
        header_text.append("SENTIMENT ANALYSIS", style="bold cyan")
        header_text.append(f"\n{date_str}", style="dim")

        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="cyan",
            padding=(0, 2)
        )

    def _build_main_panel(self, metrics: SentimentMetrics) -> Panel:
        """Build the main sentiment display panel."""
        content = Text()

        # Overall sentiment bar
        content.append("\n  Overall Sentiment:  ", style="bold")
        content.append_text(self._score_to_bar(metrics.current_score))
        content.append("  ")
        content.append_text(self._format_score(metrics.current_score))
        content.append(f" {metrics.signal.replace('_', ' ')}", style=metrics.signal_color)
        content.append("\n")

        return Panel(
            content,
            title="[bold]Sentiment Overview[/bold]",
            box=box.ROUNDED,
            padding=(0, 1)
        )

    def _build_metrics_table(self, metrics: SentimentMetrics) -> Table:
        """Build the metrics comparison table."""
        table = Table(
            title="Time-Series Analysis",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Period", style="dim", width=12)
        table.add_column("Score", justify="right", width=10)
        table.add_column("Trend", justify="center", width=8)
        table.add_column("Articles", justify="right", width=10)
        table.add_column("Confidence", justify="center", width=12)

        # Today
        arrow, arrow_color = metrics.momentum_arrow
        today_trend = Text(arrow, style=arrow_color)
        today_confidence = self._get_confidence_indicator(metrics.current_count, 5)
        table.add_row(
            "Today",
            self._format_score(metrics.current_score),
            today_trend,
            str(metrics.current_count),
            today_confidence
        )

        # 7-day trend indicator
        if metrics.weekly_score > 0:
            weekly_trend = Text("^", style="green")
        elif metrics.weekly_score < -0.1:
            weekly_trend = Text("v", style="red")
        else:
            weekly_trend = Text("->", style="yellow")
        weekly_confidence = self._get_confidence_indicator(metrics.weekly_count, 20)
        table.add_row(
            "7-Day Avg",
            self._format_score(metrics.weekly_score),
            weekly_trend,
            str(metrics.weekly_count),
            weekly_confidence
        )

        # 30-day trend indicator
        if metrics.monthly_score > 0:
            monthly_trend = Text("^", style="green")
        elif metrics.monthly_score < -0.1:
            monthly_trend = Text("v", style="red")
        else:
            monthly_trend = Text("->", style="yellow")
        monthly_confidence = self._get_confidence_indicator(metrics.monthly_count, 50)
        table.add_row(
            "30-Day Avg",
            self._format_score(metrics.monthly_score),
            monthly_trend,
            str(metrics.monthly_count),
            monthly_confidence
        )

        return table

    def _get_confidence_indicator(self, count: int, threshold: int) -> Text:
        """Get confidence indicator based on article count."""
        if count >= threshold:
            return Text("●●●", style="green")  # High confidence
        elif count >= threshold // 2:
            return Text("●●○", style="yellow")  # Medium confidence
        elif count > 0:
            return Text("●○○", style="red")  # Low confidence
        else:
            return Text("○○○", style="dim")  # No data

    def _build_signal_panel(self, metrics: SentimentMetrics) -> Panel:
        """Build the trading signal interpretation panel."""
        signal_text = Text()

        # Main signal
        signal_name = metrics.signal.replace("_", " ")
        signal_text.append("SIGNAL: ", style="bold")
        signal_text.append(signal_name, style=metrics.signal_color)
        signal_text.append("\n\n")

        # Interpretation
        interpretation = self._get_signal_interpretation(metrics)
        signal_text.append(interpretation, style="dim")

        # Additional metrics
        signal_text.append("\n\n")
        signal_text.append("Momentum: ", style="bold dim")
        signal_text.append_text(self._format_score(metrics.momentum))
        arrow, arrow_color = metrics.momentum_arrow
        signal_text.append(f" {arrow}", style=arrow_color)

        signal_text.append("    Dispersion: ", style="bold dim")
        dispersion_color = "green" if metrics.dispersion < 0.3 else "yellow" if metrics.dispersion < 0.5 else "red"
        signal_text.append(f"{metrics.dispersion:.2f}", style=dispersion_color)
        consensus = "High consensus" if metrics.dispersion < 0.3 else "Mixed signals" if metrics.dispersion < 0.5 else "Divergent views"
        signal_text.append(f" ({consensus})", style="dim")

        signal_text.append("    Volume: ", style="bold dim")
        volume_color = "green" if metrics.volume_trend == "increasing" else "red" if metrics.volume_trend == "decreasing" else "yellow"
        signal_text.append(metrics.volume_trend.capitalize(), style=volume_color)

        return Panel(
            signal_text,
            title="[bold]Trading Signal[/bold]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2)
        )

    def _get_signal_interpretation(self, metrics: SentimentMetrics) -> str:
        """Get human-readable interpretation of the signal."""
        interpretations = {
            "STRONG_BULLISH": (
                "Strong positive sentiment detected. News flow supports long positions.\n"
                "Consider: Entry points for bullish trades. Watch for potential reversal signals."
            ),
            "BULLISH": (
                "Moderately positive sentiment. News bias is favorable.\n"
                "Consider: Gradual position building. Monitor for momentum confirmation."
            ),
            "NEUTRAL": (
                "Mixed or neutral sentiment. No clear directional bias in news.\n"
                "Consider: Wait for clearer signals. Range-bound strategies may apply."
            ),
            "BEARISH": (
                "Moderately negative sentiment. News flow suggests caution.\n"
                "Consider: Reducing long exposure. Watch for potential short opportunities."
            ),
            "STRONG_BEARISH": (
                "Strong negative sentiment detected. News flow is predominantly negative.\n"
                "Consider: Defensive positioning. Short opportunities may exist."
            )
        }
        return interpretations.get(metrics.signal, "Unable to determine signal.")

    def _display_recent_headlines(self, headlines: List[Tuple[str, str]], max_display: int = 5) -> None:
        """Display recent headlines with sentiment."""
        table = Table(
            title="Recent Headlines",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold"
        )

        table.add_column("Sentiment", justify="center", width=10)
        table.add_column("Headline", width=70)

        for headline, sentiment in headlines[:max_display]:
            sentiment_lower = sentiment.lower()
            if sentiment_lower == "positive":
                sent_display = Text("POSITIVE", style="green")
            elif sentiment_lower == "negative":
                sent_display = Text("NEGATIVE", style="red")
            else:
                sent_display = Text("NEUTRAL", style="yellow")

            # Truncate headline if too long
            if len(headline) > 68:
                headline = headline[:65] + "..."

            table.add_row(sent_display, headline)

        self.console.print(table)

    def display_news_snippet(
        self,
        asset_name: str,
        market: str,
        articles: List[Dict],
        max_display: int = 5
    ) -> None:
        """
        Display a snippet of fetched news articles.

        Args:
            asset_name: Name of the asset
            market: Market identifier
            articles: List of article dictionaries with 'title', 'source', 'datetime' keys
            max_display: Maximum number of articles to display
        """
        # Use text labels instead of emojis for Windows compatibility
        market_label = "[US]" if market.upper() == "US" else "[CN]"

        # Header
        header = Text()
        header.append(f"\n{market_label} ", style="bold blue")
        header.append(f"{asset_name.upper()}", style="bold white")
        header.append(f" - Latest News ({len(articles)} articles fetched)\n", style="dim")
        self.console.print(header)

        # Articles table
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Date", width=12)
        table.add_column("Source", width=15)
        table.add_column("Headline", width=60)

        for article in articles[:max_display]:
            # Parse datetime
            dt_str = article.get("datetime", article.get("date", ""))
            if dt_str:
                try:
                    if "T" in str(dt_str):
                        dt = datetime.fromisoformat(str(dt_str).replace("Z", ""))
                        date_display = dt.strftime("%Y-%m-%d")
                    else:
                        date_display = str(dt_str)[:10]
                except:
                    date_display = str(dt_str)[:10]
            else:
                date_display = "Unknown"

            source = article.get("source", "Unknown")[:14]
            title = article.get("title", article.get("headline", "No title"))
            if len(title) > 58:
                title = title[:55] + "..."

            table.add_row(date_display, source, title)

        self.console.print(table)

        if len(articles) > max_display:
            self.console.print(f"  [dim]... and {len(articles) - max_display} more articles[/dim]\n")

        self.console.print("[dim]Type 'analyze' to run sentiment analysis[/dim]\n")

    def display_loading(self, message: str = "Loading...") -> None:
        """Display a loading message."""
        self.console.print(f"\n[cyan]{message}[/cyan]")

    def display_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(Panel(
            Text(message, style="red"),
            title="[bold red]Error[/bold red]",
            box=box.ROUNDED,
            border_style="red"
        ))

    def display_success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"[green]✓[/green] {message}")

    def generate_sentiment_graph(
        self,
        articles: List[Dict],
        asset_name: str,
        market: str,
        sentiment_field: str = "sentiment",
        output_dir: Optional[Path] = None,
        days: int = 30
    ) -> Optional[Path]:
        """
        Generate a sentiment time-series graph and save it to a file.

        Args:
            articles: List of article dicts with 'datetime'/'date' and sentiment_field
            asset_name: Name of the asset for the title
            market: Market identifier (US/CHINA)
            sentiment_field: Name of the field containing sentiment
            output_dir: Directory to save the plot (defaults to plots/)
            days: Number of days to include in the graph

        Returns:
            Path to the saved graph file, or None if generation failed
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for file output
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            self.console.print("[yellow]Warning: matplotlib not installed. Cannot generate graph.[/yellow]")
            return None

        if not articles:
            self.console.print("[yellow]No articles to plot.[/yellow]")
            return None

        # Sentiment values
        sentiment_values = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0
        }

        # Group articles by date
        daily_data = defaultdict(list)
        today = datetime.now().date()
        cutoff = today - timedelta(days=days)

        for article in articles:
            dt_str = article.get("datetime", article.get("date", ""))
            if not dt_str:
                continue

            try:
                if isinstance(dt_str, datetime):
                    date = dt_str.date()
                elif "T" in str(dt_str):
                    date = datetime.fromisoformat(str(dt_str).replace("Z", "")).date()
                else:
                    date = datetime.strptime(str(dt_str)[:10], "%Y-%m-%d").date()
            except:
                continue

            if date < cutoff:
                continue

            sentiment = article.get(sentiment_field, "neutral").lower()
            score = sentiment_values.get(sentiment, 0.0)
            daily_data[date].append(score)

        if not daily_data:
            self.console.print("[yellow]No valid data points for graph.[/yellow]")
            return None

        # Calculate daily averages and counts
        dates = sorted(daily_data.keys())
        avg_scores = [sum(daily_data[d]) / len(daily_data[d]) for d in dates]
        counts = [len(daily_data[d]) for d in dates]

        # Calculate 7-day moving average
        ma_scores = []
        for i, date in enumerate(dates):
            # Get scores from last 7 days
            week_scores = []
            for j in range(max(0, i - 6), i + 1):
                week_scores.extend(daily_data[dates[j]])
            if week_scores:
                ma_scores.append(sum(week_scores) / len(week_scores))
            else:
                ma_scores.append(0)

        # Create the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                                        gridspec_kw={'hspace': 0.1})

        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        fig.patch.set_facecolor('#1e1e2e')
        ax1.set_facecolor('#1e1e2e')
        ax2.set_facecolor('#1e1e2e')

        # --- Top plot: Sentiment scores ---
        # Plot daily scores as scatter points with color coding
        colors = ['#2ecc71' if s > 0.1 else '#e74c3c' if s < -0.1 else '#f39c12' for s in avg_scores]
        ax1.scatter(dates, avg_scores, c=colors, s=60, alpha=0.7, zorder=3, label='Daily Avg')

        # Plot 7-day moving average line
        ax1.plot(dates, ma_scores, color='#3498db', linewidth=2.5, alpha=0.9,
                 label='7-Day MA', zorder=2)

        # Fill between for visual emphasis
        ax1.fill_between(dates, ma_scores, 0, where=[s > 0 for s in ma_scores],
                         color='#2ecc71', alpha=0.15, interpolate=True)
        ax1.fill_between(dates, ma_scores, 0, where=[s <= 0 for s in ma_scores],
                         color='#e74c3c', alpha=0.15, interpolate=True)

        # Zero line
        ax1.axhline(y=0, color='#95a5a6', linestyle='-', linewidth=1, alpha=0.5)

        # Threshold lines
        ax1.axhline(y=0.2, color='#2ecc71', linestyle='--', linewidth=0.8, alpha=0.4)
        ax1.axhline(y=-0.2, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.4)

        # Labels and title
        market_label = "US" if market.upper() == "US" else "CHINA"
        ax1.set_title(f'{asset_name.upper()} Sentiment Analysis ({market_label})',
                      fontsize=14, fontweight='bold', color='white', pad=15)
        ax1.set_ylabel('Sentiment Score', fontsize=11, color='white')
        ax1.set_ylim(-1.1, 1.1)

        # Style axis
        ax1.tick_params(axis='both', colors='white', labelsize=9)
        ax1.spines['bottom'].set_color('#404040')
        ax1.spines['left'].set_color('#404040')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(MaxNLocator(10))
        ax1.set_xticklabels([])  # Hide x labels for top plot

        # Legend
        ax1.legend(loc='upper left', facecolor='#2d2d3d', edgecolor='#404040',
                   labelcolor='white', fontsize=9)

        # Add sentiment zone labels
        ax1.text(dates[-1], 0.8, 'BULLISH', fontsize=8, color='#2ecc71', alpha=0.7,
                 ha='right', va='center')
        ax1.text(dates[-1], -0.8, 'BEARISH', fontsize=8, color='#e74c3c', alpha=0.7,
                 ha='right', va='center')

        # --- Bottom plot: Volume bars ---
        bar_colors = ['#2ecc71' if avg_scores[i] > 0.1 else '#e74c3c' if avg_scores[i] < -0.1 else '#f39c12'
                      for i in range(len(dates))]
        ax2.bar(dates, counts, color=bar_colors, alpha=0.7, width=0.8)

        ax2.set_ylabel('Articles', fontsize=11, color='white')
        ax2.set_xlabel('Date', fontsize=11, color='white')
        ax2.tick_params(axis='both', colors='white', labelsize=9)
        ax2.spines['bottom'].set_color('#404040')
        ax2.spines['left'].set_color('#404040')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(MaxNLocator(10))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Tight layout
        plt.tight_layout()

        # Save the figure
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{asset_name.lower()}_{market.lower()}_sentiment_{timestamp}.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        self.console.print(f"[green]Graph saved to:[/green] {output_path}")
        return output_path

    def display_graph_summary(
        self,
        articles: List[Dict],
        sentiment_field: str = "sentiment",
        days: int = 7
    ) -> None:
        """
        Display a simple ASCII sentiment trend in the terminal.

        Args:
            articles: List of article dicts
            sentiment_field: Name of the sentiment field
            days: Number of days to show
        """
        sentiment_values = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

        # Group by date
        daily_data = defaultdict(list)
        today = datetime.now().date()
        cutoff = today - timedelta(days=days)

        for article in articles:
            dt_str = article.get("datetime", article.get("date", ""))
            if not dt_str:
                continue
            try:
                if isinstance(dt_str, datetime):
                    date = dt_str.date()
                elif "T" in str(dt_str):
                    date = datetime.fromisoformat(str(dt_str).replace("Z", "")).date()
                else:
                    date = datetime.strptime(str(dt_str)[:10], "%Y-%m-%d").date()
            except:
                continue

            if date < cutoff:
                continue

            sentiment = article.get(sentiment_field, "neutral").lower()
            score = sentiment_values.get(sentiment, 0.0)
            daily_data[date].append(score)

        if not daily_data:
            return

        # Build ASCII chart
        dates = sorted(daily_data.keys())
        self.console.print("\n[bold cyan]7-Day Sentiment Trend[/bold cyan]")
        self.console.print("[dim]" + "-" * 50 + "[/dim]")

        for date in dates[-7:]:
            scores = daily_data[date]
            avg = sum(scores) / len(scores)
            count = len(scores)

            # Create bar
            bar_len = int((avg + 1) * 10)  # Map -1..1 to 0..20
            if avg > 0.1:
                bar = "[green]" + "█" * bar_len + "[/green]"
            elif avg < -0.1:
                bar = "[red]" + "█" * bar_len + "[/red]"
            else:
                bar = "[yellow]" + "█" * bar_len + "[/yellow]"

            date_str = date.strftime("%m/%d")
            score_str = f"+{avg:.2f}" if avg >= 0 else f"{avg:.2f}"
            self.console.print(f"  {date_str} |{bar:20s}| {score_str} ({count} articles)")
