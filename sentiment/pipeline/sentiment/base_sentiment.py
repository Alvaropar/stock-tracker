"""
Abstract interface for sentiment models.

Any model that can classify headlines into sentiment labels
implements this interface.  This lets the rest of the pipeline
swap implementations (LoRA, zero-shot, distilled, etc.) without
touching orchestration or client code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseSentimentModel(ABC):
    """Contract for sentiment classifiers."""

    @abstractmethod
    def analyze(self, texts: List[str], progress_cb=None) -> List[str]:
        """
        Classify a list of headlines.

        Parameters
        ----------
        texts : list[str]
            Raw headline strings.
        progress_cb : callable, optional
            Called as ``progress_cb(done, total)`` after each headline.

        Returns
        -------
        list[str]
            One of ``"positive"``, ``"negative"``, ``"neutral"``
            per input text.
        """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (GPU/CPU)."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free accelerator memory."""
