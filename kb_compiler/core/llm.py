"""Kimi LLM client with streaming, retry, and token tracking."""

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import AsyncOpenAI, RateLimitError
from rich.console import Console

console = Console()


def get_http_client():
    """Create httpx client with proxy disabled."""
    # Disable proxy by creating client with trust_env=False
    # This ignores HTTP_PROXY/HTTPS_PROXY environment variables
    return httpx.AsyncClient(
        trust_env=False,  # Ignore proxy environment variables
        timeout=httpx.Timeout(600.0, connect=30.0),
    )


def create_llm_client(
    api_key: str,
    base_url: str = "https://api.moonshot.cn/v1",
    model: str = "moonshot-v1-128k",
    code_mode: bool = False,
):
    """Factory function to create appropriate LLM client.

    Args:
        api_key: API key
        base_url: API base URL
        model: Model name
        code_mode: If True, use Kimi Code API (anthropic format)
    """
    if code_mode or "kimi.com/coding" in base_url:
        # Use Anthropic SDK for Kimi Code
        try:
            from anthropic import AsyncAnthropic
            return AnthropicClient(api_key=api_key, base_url=base_url, model=model)
        except ImportError:
            console.print("[yellow]anthropic SDK not installed. Using OpenAI compatibility mode.[/]")
            return KimiClient(api_key=api_key, base_url=base_url, model=model)
    else:
        return KimiClient(api_key=api_key, base_url=base_url, model=model)


@dataclass
class LLMResponse:
    """Structured LLM response."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    model: str = ""


@dataclass
class StreamingChunk:
    """A chunk of streaming response."""

    content: str
    is_finished: bool = False


class KimiClient:
    """Async client for Kimi API with retry logic."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "moonshot-v1-128k",
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
                    max_tokens=max_tokens,
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

            except RateLimitError as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                console.print(f"[yellow]Rate limited, waiting {wait_time}s...[/]")
                time.sleep(wait_time)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]Request failed: {e}, retrying...[/]")
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
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                is_finished = chunk.choices[0].finish_reason is not None
                yield StreamingChunk(content=content, is_finished=is_finished)

        except Exception as e:
            console.print(f"[red]Streaming error: {e}[/]")
            raise

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "model": self.model,
        }


# Default prompts for knowledge compiler
COMPILER_SYSTEM_PROMPT = """You are a Knowledge Compiler, an expert at transforming raw documents into structured, interconnected knowledge.

Your task is to:
1. Identify key concepts from source documents
2. Create well-structured concept articles
3. Establish connections between concepts using wiki-links [[Concept Name]]
4. Resolve contradictions between sources
5. Preserve specific data, numbers, and quotes

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- Each concept gets its own article
- Use wiki-links [[Concept]] to connect related ideas
- Maintain information density
- Note contradictions explicitly
- Mark open questions clearly"""


QUERIER_SYSTEM_PROMPT = """You are a Knowledge Querier, an expert at answering questions based on compiled wiki content.

Your task is to:
1. Synthesize information from provided wiki articles
2. Answer questions accurately and comprehensively
3. Cite sources using [[Concept Name]] format
4. Acknowledge when information is incomplete
5. Suggest related concepts for further exploration

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- Base answers strictly on provided context
- Use wiki-links to reference concepts
- Be precise with data and numbers
- Note any contradictions in sources"""


LINTER_SYSTEM_PROMPT = """You are a Wiki Linter, an expert at maintaining knowledge base health.

Your task is to:
1. Identify contradictions or outdated information
2. Find isolated concepts (no incoming links)
3. Suggest missing cross-references
4. Propose new concepts based on existing content
5. Flag unsupported claims

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- Be thorough but constructive
- Prioritize actionable suggestions
- Consider the knowledge graph structure"""


class AnthropicClient:
    """Anthropic API compatible client for Kimi Code."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.kimi.com/coding/",
        model: str = "kimi-k2-5",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        try:
            from anthropic import AsyncAnthropic
            self.http_client = get_http_client()
            self.client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                http_client=self.http_client,
            )
        except ImportError:
            raise ImportError(
                "anthropic SDK is required for Kimi Code mode. "
                "Install with: pip install anthropic"
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
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or 4096,
                    temperature=temperature,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                )

                duration = (time.time() - start_time) * 1000
                self._total_requests += 1

                # Extract text content
                content = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

                # Estimate tokens
                prompt_tokens = len(prompt.split()) + len((system_prompt or "").split())
                completion_tokens = len(content.split())
                total_tokens = prompt_tokens + completion_tokens
                self._total_tokens += total_tokens

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    duration_ms=duration,
                    model=response.model,
                )

            except Exception as e:
                last_error = e
                if "rate_limit" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** attempt)
                    console.print(f"[yellow]Rate limited, waiting {wait_time}s...[/]")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    console.print(f"[yellow]Request failed: {e}, retrying...[/]")
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
        try:
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield StreamingChunk(content=event.delta.text, is_finished=False)
                elif event.type == "message_stop":
                    yield StreamingChunk(content="", is_finished=True)

        except Exception as e:
            console.print(f"[red]Streaming error: {e}[/]")
            raise

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "model": self.model,
        }
