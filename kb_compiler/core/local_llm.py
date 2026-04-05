"""Local LLM client for MLX, Ollama, etc. with OpenAI-compatible API."""

import time
from collections.abc import AsyncIterator
from typing import Optional

from openai import AsyncOpenAI
from rich.console import Console

from kb_compiler.core.llm import LLMResponse, StreamingChunk, get_http_client

console = Console()


class LocalLLMClient:
    """Client for local LLM (MLX, Ollama, etc.) with OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://127.0.0.1:8017/v1",
        model: str = "Qwen3___5-27B-Claude-4___6-Opus-Distilled-MLX-6bit",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.http_client = get_http_client()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self.http_client,
        )
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._total_requests = 0
        self._total_tokens = 0

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a completion request with retry logic."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or 4096,
                )

                duration = (time.time() - start_time) * 1000
                self._total_requests += 1

                usage = response.usage
                if usage:
                    self._total_tokens += usage.total_tokens

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    duration_ms=duration,
                    model=response.model,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]Local LLM request failed: {e}, retrying...[/]")
                    time.sleep(self.retry_delay)
                else:
                    raise

        raise last_error or Exception("Max retries exceeded")

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream completion response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                stream=True,
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                is_finished = chunk.choices[0].finish_reason is not None
                yield StreamingChunk(content=content, is_finished=is_finished)

        except Exception as e:
            console.print(f"[red]Local LLM streaming error: {e}[/]")
            raise

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "model": self.model,
        }
