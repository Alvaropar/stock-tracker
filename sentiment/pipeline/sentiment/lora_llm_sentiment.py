"""
Local sentiment model: base LLM with optional LoRA adapter.

Runs entirely locally on PyTorch + HuggingFace.  Supports 4-bit
quantisation via bitsandbytes and automatic CPU fallback.

If ``SentimentModelConfig.adapter_path`` is ``None``, the base model
is loaded directly (no LoRA merge).  Otherwise base + adapter weights
are merged via PEFT.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from ..config.models import SentimentModelConfig
from .base_sentiment import BaseSentimentModel

log = logging.getLogger("pipeline.sentiment")


class LoRASentimentModel(BaseSentimentModel):
    """
    Sentiment classifier using a base causal-LM + optional LoRA adapter.

    >>> model = LoRASentimentModel(SentimentModelConfig())
    >>> model.analyze(["Gold prices surge to record high"])
    ['positive']
    """

    def __init__(self, config: Optional[SentimentModelConfig] = None):
        self.cfg = config or SentimentModelConfig()
        self._model = None
        self._tokenizer = None

    # ── public API ───────────────────────────────────────────────────

    def analyze(self, texts: List[str], progress_cb=None) -> List[str]:
        log.info("Analyzing sentiment for %d headlines", len(texts))
        self.load()
        results = []
        for i, t in enumerate(texts):
            results.append(self._classify_one(t))
            if progress_cb:
                progress_cb(i + 1, len(texts))
        counts = {s: results.count(s) for s in ("positive", "negative", "neutral")}
        log.info("Sentiment results: %s", counts)
        return results

    def load(self) -> None:
        if self._model is not None:
            return

        log.info("Loading sentiment model from '%s'%s",
                 self.cfg.base_model_path,
                 f" + adapter '{self.cfg.adapter_path}'" if self.cfg.adapter_path else " (base only)")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_path = str(self.cfg.base_model_path)
        adapter_path = str(self.cfg.adapter_path) if self.cfg.adapter_path else None
        cuda = torch.cuda.is_available()

        # Quantisation config
        bnb = None
        if self.cfg.use_4bit and cuda:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.cfg.bnb_4bit_compute_dtype),
                llm_int8_enable_fp32_cpu_offload=True,
            )

        # Tokenizer — prefer adapter dir (may have fine-tuned tokens),
        # fall back to base model
        tok_path = adapter_path if (adapter_path and Path(adapter_path).exists()) else base_path

        # Workaround for transformers 4.57+ / Qwen compatibility issue
        # where extra_special_tokens is a list instead of dict
        from transformers import PreTrainedTokenizerBase
        orig_set_special_tokens = PreTrainedTokenizerBase._set_model_specific_special_tokens
        def patched_set_special_tokens(self, special_tokens):
            if isinstance(special_tokens, list):
                special_tokens = {}
            return orig_set_special_tokens(self, special_tokens)
        PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_special_tokens

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tok_path, trust_remote_code=True)
        except Exception as e:
            log.warning("Tokenizer load failed: %s", e)
            raise
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Base model
        load_kw = dict(trust_remote_code=True)
        if cuda:
            load_kw.update(
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=(torch.bfloat16 if self.cfg.bnb_4bit_compute_dtype == "bfloat16"
                             else torch.float16),
                max_memory=self.cfg.max_memory,
            )
        else:
            load_kw.update(torch_dtype=torch.float32, low_cpu_mem_usage=True)

        base = AutoModelForCausalLM.from_pretrained(base_path, **load_kw)

        # Optionally merge LoRA adapter
        if adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        else:
            self._model = base

        self._model.eval()

    def unload(self) -> None:
        log.info("Unloading sentiment model")
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── internals ────────────────────────────────────────────────────

    def _classify_one(self, headline: str) -> str:
        import torch

        prompt = self.cfg.format_prompt(headline)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            if self.cfg.do_sample:
                gen_kwargs["temperature"] = self.cfg.temperature
                gen_kwargs["top_p"] = self.cfg.top_p
            out = self._model.generate(**inputs, **gen_kwargs)

        response = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
        return self._parse(response)

    @staticmethod
    def _parse(response: str) -> str:
        response = response.split("<|")[0].split("\n")[0].strip()
        if "positive" in response:
            return "positive"
        if "negative" in response:
            return "negative"
        if "neutral" in response:
            return "neutral"
        return "neutral"
