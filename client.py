"""
LLM Gateway Client — drop-in SDK for calling your gateway
Usage:
    from client import GatewayClient
    client = GatewayClient("http://localhost:8000")
    response = client.chat("Explain how transformers work in detail")
    print(response.content)
"""

import json
import httpx
from dataclasses import dataclass
from typing import Optional, Iterator


@dataclass
class GatewayResponse:
    content: str
    model: str
    provider: str
    difficulty: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int

    def __str__(self):
        return (
            f"[{self.difficulty.upper()} → {self.provider}/{self.model}] "
            f"{self.latency_ms}ms | {self.tokens_in}→{self.tokens_out} tokens "
            f"| ${self.cost_usd:.6f}\n\n{self.content}"
        )


class GatewayClient:
    """Synchronous gateway client with auto-routing."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        difficulty: Optional[str] = None,  # "simple" | "medium" | "complex" | "local"
        provider: Optional[str] = None,    # "anthropic" | "openai" | "google" | "ollama"
        privacy_mode: bool = False,
        max_tokens: int = 2048,
    ) -> GatewayResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "privacy_mode": privacy_mode,
        }
        if difficulty:
            payload["difficulty"] = difficulty
        if provider:
            payload["provider"] = provider

        with httpx.Client(timeout=self.timeout) as http:
            r = http.post(f"{self.base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()

        return GatewayResponse(
            content=data["content"],
            model=data["route"]["model"],
            provider=data["route"]["provider"],
            difficulty=data["route"]["difficulty"],
            tokens_in=data["usage"].get("prompt_tokens", 0),
            tokens_out=data["usage"].get("completion_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            latency_ms=data.get("latency_ms", 0),
        )

    def stream_chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        difficulty: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Iterator[str]:
        """Yields text chunks as they arrive."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"messages": messages, "stream": True}
        if difficulty:
            payload["difficulty"] = difficulty
        if provider:
            payload["provider"] = provider

        with httpx.Client(timeout=self.timeout) as http:
            with http.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data:
                            yield data["delta"]
                        elif data.get("done"):
                            break

    def preview_route(self, prompt: str) -> dict:
        """See how a prompt would be routed without calling any model."""
        with httpx.Client(timeout=10) as http:
            r = http.get(f"{self.base_url}/v1/route/preview", params={"prompt": prompt})
            r.raise_for_status()
            return r.json()

    def stats(self) -> dict:
        with httpx.Client(timeout=10) as http:
            r = http.get(f"{self.base_url}/v1/stats")
            r.raise_for_status()
            return r.json()


# ─── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = GatewayClient()

    print("=" * 60)
    print("ROUTING PREVIEW (no API calls)")
    print("=" * 60)

    examples = [
        "What is Python?",
        "Translate 'hello' to Spanish",
        "Explain how attention mechanisms work in transformers",
        "Write a complete REST API in FastAPI with auth, pagination, tests",
        "Debug this async Python code and explain all issues found",
    ]

    for prompt in examples:
        route = client.preview_route(prompt)
        print(f"\n[{route['difficulty'].upper():8}] → {route['provider']}/{route['model']}")
        print(f"           Prompt: {prompt[:60]}")

    print("\n" + "=" * 60)
    print("LIVE COMPLETIONS")
    print("=" * 60)

    # Simple → auto-routes to cheapest/fastest
    print("\n[1] Auto-route (simple task):")
    try:
        r = client.chat("What is 42 × 7?")
        print(r)
    except Exception as e:
        print(f"  Error: {e} (start the gateway first)")

    # Force complex difficulty
    print("\n[2] Force complex difficulty:")
    try:
        r = client.chat(
            "Design a distributed rate limiter",
            difficulty="complex",
        )
        print(r)
    except Exception as e:
        print(f"  Error: {e}")

    # Privacy mode → always local Ollama
    print("\n[3] Privacy mode (Ollama only):")
    try:
        r = client.chat("Summarize my meeting notes", privacy_mode=True)
        print(r)
    except Exception as e:
        print(f"  Error: {e}")

    # Streaming
    print("\n[4] Streaming response:")
    try:
        for chunk in client.stream_chat("Count from 1 to 5"):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"  Error: {e}")
