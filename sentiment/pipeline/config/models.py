"""
Model configuration for filter and sentiment LLMs.

All paths are relative to PROJECT_ROOT by default.
Override via constructor arguments or environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(
    os.environ.get("PIPELINE_PROJECT_ROOT", "")
) if os.environ.get("PIPELINE_PROJECT_ROOT") else Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def discover_models(models_dir: Path | None = None) -> Dict:
    """
    Scan the models directory and return available base models and adapters.

    A directory is a **base model** if it contains ``config.json``
    (HuggingFace format) but NOT ``adapter_config.json``.

    A directory is a **LoRA adapter** if it contains ``adapter_config.json``.

    Returns
    -------
    dict
        {
            "base_models": [{"name": "Qwen3-8B", "path": "..."}],
            "adapters":    [{"name": "sft-qwen3", "path": "..."}],
        }
    """
    root = models_dir or MODELS_DIR
    base_models: List[Dict[str, str]] = []
    adapters: List[Dict[str, str]] = []

    if not root.exists():
        return {"base_models": base_models, "adapters": adapters}

    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        if (p / "config.json").exists() and not (p / "adapter_config.json").exists():
            base_models.append({"name": p.name, "path": str(p)})
        if (p / "adapter_config.json").exists():
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p.name
            adapters.append({"name": str(rel), "path": str(p)})

    return {"base_models": base_models, "adapters": adapters}


@dataclass
class ModelConfig:
    """Base model configuration shared by filter and sentiment models."""
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    device: str = "cuda"
    max_memory: Optional[Dict] = None

    def __post_init__(self):
        if self.max_memory is None:
            self.max_memory = {0: "6GiB", "cpu": "24GiB"}


@dataclass
class FilterModelConfig(ModelConfig):
    """Configuration for the relevance-filter LLM.

    The filter uses a base model with an *optional* LoRA adapter.
    If ``adapter_path`` is None, the base model is loaded directly.
    """
    # Which backend to use: "transformers", "ollama", "lmstudio", "llamacpp"
    backend: str = "transformers"
    # Transformers backend
    base_model_path: Path = field(default_factory=lambda: MODELS_DIR / "Qwen3-8B")
    adapter_path: Optional[Path] = None  # None = base model only
    # Ollama backend
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    # LM Studio backend
    lmstudio_host: str = "http://localhost:1234"
    lmstudio_model: str = "local-model"
    # llama.cpp backend
    llamacpp_model_path: Path = field(
        default_factory=lambda: MODELS_DIR / "Qwen3-8B-Q4_K_M.gguf"
    )
    llamacpp_n_gpu_layers: int = -1
    llamacpp_n_ctx: int = 4096
    llamacpp_n_batch: int = 512
    llamacpp_n_threads: int = 8
    # Generation
    max_tokens: int = 500
    temperature: float = 0.1
    batch_size: int = 5
    # Prompt
    system_prompt: str = (
        "You are a financial news classifier. "
        "You respond ONLY with JSON arrays, no other text."
    )
    user_prompt_template: str = (
        'Classify if each headline is relevant to {asset_description}.\n'
        'Reply with ONLY a JSON array of "yes" or "no" for each headline. '
        'Example: ["yes", "no", "yes"]\n\n{headlines}\n\nJSON array:'
    )

    @property
    def transformers_model_path(self) -> Path:
        """Backward-compat alias."""
        return self.base_model_path


@dataclass
class SentimentModelConfig(ModelConfig):
    """Configuration for the sentiment LLM (base model + optional LoRA adapter).

    If ``adapter_path`` is None, only the base model is loaded (no LoRA merge).
    """
    base_model_path: Path = field(default_factory=lambda: MODELS_DIR / "Qwen3-8B")
    adapter_path: Optional[Path] = field(default_factory=lambda: MODELS_DIR / "sft-qwen3")
    # Generation
    max_new_tokens: int = 50
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    # Prompt (matches fine-tuning format)
    instruction: str = (
        "You are an AI language model trained to detect the sentiment of "
        "each sentence for finance trends.\n"
        "Analyze the following sentence and determine if the sentiment is: "
        "positive or negative or neutral.\n"
        "Return only a single word, either positive or negative or neutral."
    )

    def format_prompt(self, headline: str) -> str:
        return (
            f"<|im_start|>user\n{self.instruction}\n{headline}<|im_end|>\n"
            f"<|im_start|>assistant\n/no_think"
        )
