"""OpenAI-compatible VLLM vision client."""

from __future__ import annotations

import os
from pathlib import Path

from image_io import image_data_url


class VLLMClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model or os.getenv("VLLM_MODEL") or os.getenv("MODEL_ID") or "Qwen/Qwen3-VL-8B-Instruct"
        self.base_url = base_url or os.getenv("VLLM_BASE_URL") or os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("API_KEY") or "EMPTY"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai to call VLLM: pip install openai") from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def predict(self, image_path: str | Path, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""
