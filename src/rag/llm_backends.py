"""
BhojRAG LLM Backends
======================
Pluggable LLM interface supporting API-based and local model backends.

Default: API-based (Gemini/OpenAI compatible) to reduce compute costs.
Local models (Llama 3, Mistral, Qwen) can be plugged in via config.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        ...


class APIBackend(BaseLLM):
    """
    API-based LLM backend using OpenAI-compatible chat completions.

    Works with:
        - OpenAI GPT models
        - Google Gemini (via OpenAI-compatible endpoint)
        - Any OpenAI-compatible API (vLLM, Ollama, etc.)

    API key is read from environment:
        - OPENAI_API_KEY (default)
        - GOOGLE_API_KEY (for Gemini)
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Determine API configuration
        if "gemini" in model.lower():
            try:
                import google.generativeai as genai

                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set.")
                genai.configure(api_key=api_key)
                self._client_type = "gemini"
                self._gemini_model = genai.GenerativeModel(model)
                logger.info(f"Initialized Gemini backend: {model}")
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai package required for Gemini. "
                    "Install with: pip install google-generativeai"
                ) from exc
        else:
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set.")
                kwargs = {"api_key": api_key}
                if api_base_url:
                    kwargs["base_url"] = api_base_url
                self._client = OpenAI(**kwargs)
                self._client_type = "openai"
                logger.info(f"Initialized OpenAI backend: {model}")
            except ImportError as exc:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from exc

    def generate(self, prompt: str) -> str:
        """Generate response via API call."""
        try:
            if self._client_type == "gemini":
                response = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
                )
                return response.text

            else:  # OpenAI-compatible
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return f"[Generation Error: {e}]"

    def get_model_name(self) -> str:
        return self.model


class LocalBackend(BaseLLM):
    """
    Local HuggingFace model backend.

    Supports:
        - Standard transformers pipeline
        - Optional quantization (4-bit, 8-bit via bitsandbytes)
        - Automatic device placement
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None
        self._quantization = quantization
        self._device = device

    def _load_pipeline(self):
        """Lazy-load the model pipeline."""
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        logger.info(f"Loading local model: {self.model_name}")

        model_kwargs = {"torch_dtype": torch.float16}

        if self._quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self._quantization == "8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs=model_kwargs,
            device_map=self._device,
        )
        logger.info(f"Model loaded: {self.model_name}")

    def generate(self, prompt: str) -> str:
        """Generate response using local model."""
        self._load_pipeline()

        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return f"[Generation Error: {e}]"

    def get_model_name(self) -> str:
        return self.model_name


def get_llm(
    backend: str = "api",
    model: Optional[str] = None,
    api_base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    quantization: Optional[str] = None,
    device: str = "auto",
) -> BaseLLM:
    """
    Factory function to create an LLM backend.

    Args:
        backend: "api" or "local"
        model: Model name/identifier
        api_base_url: Override API endpoint (for OpenAI-compatible servers)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        quantization: Quantization mode for local models ("4bit", "8bit", None)
        device: Device for local models ("auto", "cuda", "cpu")

    Returns:
        BaseLLM instance.
    """
    if backend == "api":
        return APIBackend(
            model=model or "gemini-1.5-flash",
            api_base_url=api_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend == "local":
        return LocalBackend(
            model_name=model or "meta-llama/Meta-Llama-3-8B-Instruct",
            quantization=quantization,
            max_new_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'api' or 'local'.")
