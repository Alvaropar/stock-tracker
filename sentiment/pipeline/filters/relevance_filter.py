"""
LLM-based relevance filter.

Given a batch of headlines and an asset description, the filter asks
an LLM whether each headline is relevant.  Supports four backends
(transformers, ollama, lmstudio, llamacpp); the backend is selected
via ``FilterModelConfig.backend``.

The transformers backend supports an optional LoRA adapter.  If
``FilterModelConfig.adapter_path`` is None, only the base model is loaded.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config.models import FilterModelConfig

log = logging.getLogger("pipeline.filters")


class RelevanceFilter:
    """
    Classify headlines as relevant / not-relevant to a given asset.

    Usage
    -----
    >>> filt = RelevanceFilter(FilterModelConfig())
    >>> filt.set_asset_description("gold (price, mining, trade)")
    >>> results = filt.filter(["Gold rises 2%", "Apple beats earnings"])
    >>> # results == [True, False]
    """

    def __init__(self, config: Optional[FilterModelConfig] = None):
        self.cfg = config or FilterModelConfig()
        self._asset_description: str = "the financial asset"
        self._backend: Optional[_Backend] = None

    # ── public API ───────────────────────────────────────────────────

    def set_asset_description(self, description: str) -> None:
        self._asset_description = description

    def filter(self, headlines: List[str]) -> List[bool]:
        """Return a boolean mask: True = relevant."""
        log.info("Filtering %d headlines (batch_size=%d, backend=%s)",
                 len(headlines), self.cfg.batch_size, self.cfg.backend)
        results: List[bool] = []
        for i in range(0, len(headlines), self.cfg.batch_size):
            batch = headlines[i : i + self.cfg.batch_size]
            raw = self._classify_batch(batch)
            results.extend(r == "yes" for r in raw)
        relevant_count = sum(results)
        log.info("Filter result: %d/%d relevant", relevant_count, len(headlines))
        return results

    def load(self) -> None:
        """Eagerly load the model (optional; lazy-loaded on first call)."""
        self._ensure_backend()
        self._backend.load()

    def unload(self) -> None:
        """Free GPU/CPU memory."""
        if self._backend is not None:
            log.info("Unloading filter model")
            self._backend.unload()
            self._backend = None

    # ── internals ────────────────────────────────────────────────────

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        name = self.cfg.backend.lower()
        if name == "transformers":
            self._backend = _TransformersBackend(self.cfg)
        elif name == "ollama":
            self._backend = _OllamaBackend(self.cfg)
        elif name in ("lmstudio", "lm-studio", "lm_studio"):
            self._backend = _LMStudioBackend(self.cfg)
        elif name in ("llamacpp", "llama.cpp", "llama-cpp"):
            self._backend = _LlamaCppBackend(self.cfg)
        else:
            raise ValueError(f"Unknown filter backend: {self.cfg.backend}")

    def _classify_batch(self, headlines: List[str]) -> List[str]:
        if not headlines:
            return []
        self._ensure_backend()

        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
        user_msg = self.cfg.user_prompt_template.format(
            asset_description=self._asset_description,
            headlines=numbered,
        )
        text = self._backend.generate(self.cfg.system_prompt, user_msg)
        return _parse_results(text, len(headlines))


# ─── result parsing ──────────────────────────────────────────────────────────

def _parse_results(text: str, expected: int) -> List[str]:
    """Parse a JSON-array response of yes/no values."""
    text = text.strip()
    # Strip <think> blocks (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if "```" in text:
        for part in text.split("```"):
            if "[" in part and "]" in part:
                text = part.replace("json", "").strip()
                break

    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        raw = json.loads(text)
        out = []
        for r in raw:
            if isinstance(r, str):
                out.append("yes" if r.lower().strip() == "yes" else "no")
            elif isinstance(r, bool):
                out.append("yes" if r else "no")
            else:
                out.append("no")
        while len(out) < expected:
            out.append("no")
        return out[:expected]
    except json.JSONDecodeError:
        pass

    # Fallback: extract yes/no tokens
    found = re.findall(r"\b(yes|no)\b", text.lower())
    while len(found) < expected:
        found.append("no")
    return found[:expected]


# ─── backend implementations ─────────────────────────────────────────────────

class _Backend:
    """Thin interface wrapping model load/generate/unload."""
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, system: str, user: str) -> str: ...


class _TransformersBackend(_Backend):
    """HuggingFace transformers backend with optional LoRA adapter support."""

    def __init__(self, cfg: FilterModelConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_path = str(self.cfg.base_model_path)
        adapter_path = str(self.cfg.adapter_path) if self.cfg.adapter_path else None
        cuda = torch.cuda.is_available()

        bnb = None
        if self.cfg.use_4bit and cuda:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.cfg.bnb_4bit_compute_dtype),
                llm_int8_enable_fp32_cpu_offload=True,
            )

        # Tokenizer — prefer adapter dir if it exists, else base model
        tok_path = base_path
        if adapter_path and Path(adapter_path).exists():
            tok_path = adapter_path

        # Workaround for transformers 4.57+ / Qwen compatibility issue
        from transformers import PreTrainedTokenizerBase
        orig_set_special_tokens = PreTrainedTokenizerBase._set_model_specific_special_tokens
        def patched_set_special_tokens(self, special_tokens):
            if isinstance(special_tokens, list):
                special_tokens = {}
            return orig_set_special_tokens(self, special_tokens)
        PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_special_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(trust_remote_code=True, local_files_only=True)
        if cuda:
            load_kwargs.update(
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=(torch.bfloat16 if self.cfg.bnb_4bit_compute_dtype == "bfloat16"
                             else torch.float16),
                max_memory=self.cfg.max_memory,
            )
        else:
            load_kwargs.update(torch_dtype=torch.float32, low_cpu_mem_usage=True)

        base = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)

        # Optionally merge LoRA adapter
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        else:
            self.model = base

        self.model.eval()

    def unload(self) -> None:
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, system: str, user: str) -> str:
        import torch
        self.load()
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n/no_think\n"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        for _ in range(3):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    do_sample=self.cfg.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            if text and ("[" in text or "yes" in text.lower() or "no" in text.lower()):
                return text
        return text


class _OllamaBackend(_Backend):
    def __init__(self, cfg: FilterModelConfig):
        self.cfg = cfg

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def generate(self, system: str, user: str) -> str:
        import ollama
        for _ in range(5):
            resp = ollama.chat(
                model=self.cfg.ollama_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.cfg.temperature,
                         "num_predict": self.cfg.max_tokens},
            )
            text = resp["message"]["content"].strip()
            if text and ("[" in text or "yes" in text.lower() or "no" in text.lower()):
                return text
        return text


class _LMStudioBackend(_Backend):
    def __init__(self, cfg: FilterModelConfig):
        self.cfg = cfg

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def generate(self, system: str, user: str) -> str:
        import requests as _req
        resp = _req.post(
            f"{self.cfg.lmstudio_host}/v1/chat/completions",
            json={
                "model": self.cfg.lmstudio_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class _LlamaCppBackend(_Backend):
    def __init__(self, cfg: FilterModelConfig):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from llama_cpp import Llama
        self._model = Llama(
            model_path=str(self.cfg.llamacpp_model_path),
            n_gpu_layers=self.cfg.llamacpp_n_gpu_layers,
            n_ctx=self.cfg.llamacpp_n_ctx,
            n_batch=self.cfg.llamacpp_n_batch,
            n_threads=self.cfg.llamacpp_n_threads,
            verbose=False,
        )

    def unload(self) -> None:
        del self._model
        self._model = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def generate(self, system: str, user: str) -> str:
        self.load()
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        resp = self._model(
            prompt,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stop=["```", "\n\n", "<|im_end|>"],
        )
        return resp["choices"][0]["text"].strip()
