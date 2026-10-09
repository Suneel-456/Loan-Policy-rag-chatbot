"""LLM factory. The real model is an open-source instruction-tuned model run locally
with Hugging Face Transformers (CPU is fine for a 1.5B model). No API key is needed."""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

from src import config


class LocalChatLLM(LLM):
    """Thin LangChain wrapper around a local Hugging Face chat model."""

    model_id: str
    max_new_tokens: int = 256

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    # Real token counts of the most recent call (read by the bot for usage tracking).
    _last_usage: dict = PrivateAttr(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    @property
    def last_usage(self) -> dict:
        return dict(self._last_usage)

    def __init__(self, **data: Any):
        super().__init__(**data)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config.TORCH_THREADS > 0:
            torch.set_num_threads(config.TORCH_THREADS)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "local-hf-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic answers
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        self._last_usage = {"prompt_tokens": int(prompt_len), "completion_tokens": int(len(new_tokens))}
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ExtractiveFakeLLM(LLM):
    """Test-only stand-in: returns the first retrieved excerpt. Never used in production."""

    model_id: str = "fake-extractive"
    _last_usage: dict = PrivateAttr(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    @property
    def last_usage(self) -> dict:
        return dict(self._last_usage)

    @property
    def _llm_type(self) -> str:
        return "fake-extractive"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        start = prompt.find("Policy excerpts:")
        end = prompt.find("Question:")
        context = ""
        if start != -1 and end != -1:
            context = prompt[start + len("Policy excerpts:"):end].strip()
        first = context.split("\n\n")[0] if context else ""
        out = "[FAKE LLM] " + " ".join(first.split())[:300]
        # rough estimate (about 4 characters per token); real counts come from LocalChatLLM
        self._last_usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(out) // 4}
        return out


def get_llm():
    if config.USE_FAKE_MODELS:
        return ExtractiveFakeLLM()
    return LocalChatLLM(model_id=config.LLM_MODEL, max_new_tokens=config.MAX_NEW_TOKENS)


def get_small_llm():
    """The cheap model used by the router for simple questions (loaded lazily)."""
    if config.USE_FAKE_MODELS:
        return ExtractiveFakeLLM(model_id="fake-small")
    return LocalChatLLM(model_id=config.LLM_MODEL_SMALL, max_new_tokens=config.MAX_NEW_TOKENS)