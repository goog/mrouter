"""
LLM API Gateway - Intelligent Multi-Provider Router
Supports: OpenAI, Anthropic, Google Gemini, Ollama (local)
Auto-routes based on task complexity classification
"""

import os
import re
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm-gateway")


# ─── Enums & Constants ─────────────────────────────────────────────────────────
class TaskDifficulty(str, Enum):
    SIMPLE = "simple"       # fast, cheap model (haiku / gpt-4o-mini / gemini-flash)
    MEDIUM = "medium"       # balanced model (sonnet / gpt-4o / gemini-pro)
    COMPLEX = "complex"     # most capable model (opus / gpt-4 / gemini-ultra)
    LOCAL = "local"         # always use local Ollama (privacy mode)


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


# ─── Model Registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[Provider, dict[TaskDifficulty, str]] = {
    Provider.ANTHROPIC: {
        TaskDifficulty.SIMPLE:  "claude-haiku-4-5-20251001",
        TaskDifficulty.MEDIUM:  "claude-sonnet-4-6",
        TaskDifficulty.COMPLEX: "claude-opus-4-6",
        TaskDifficulty.LOCAL:   "claude-haiku-4-5-20251001",
    },
    Provider.OPENAI: {
        TaskDifficulty.SIMPLE:  "gpt-4o-mini",
        TaskDifficulty.MEDIUM:  "gpt-4o",
        TaskDifficulty.COMPLEX: "o1-preview",
        TaskDifficulty.LOCAL:   "gpt-4o-mini",
    },
    Provider.GOOGLE: {
        TaskDifficulty.SIMPLE:  "gemini-1.5-flash",
        TaskDifficulty.MEDIUM:  "gemini-1.5-pro",
        TaskDifficulty.COMPLEX: "gemini-ultra",
        TaskDifficulty.LOCAL:   "gemini-1.5-flash",
    },
    Provider.OLLAMA: {
        TaskDifficulty.SIMPLE:  "llama3.2:3b",
        TaskDifficulty.MEDIUM:  "llama3.1:8b",
        TaskDifficulty.COMPLEX: "llama3.1:70b",
        TaskDifficulty.LOCAL:   "llama3.1:8b",
    },
    Provider.OPENROUTER: {
        TaskDifficulty.SIMPLE:  "minimax/minimax-m2.7",
        TaskDifficulty.MEDIUM:  "anthropic/claude-haiku-4.5",
        TaskDifficulty.COMPLEX: "xiaomi/mimo-v2-pro",
        TaskDifficulty.LOCAL:   "gpt-4o-mini",
    },
}

# Cost per 1K tokens (input/output USD) — for budget tracking
MODEL_COST: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001":    (0.00025, 0.00125),
    "claude-sonnet-4-6":            (0.003,   0.015),
    "claude-opus-4-6":              (0.015,   0.075),
    "gpt-4o-mini":                  (0.00015, 0.0006),
    "gpt-4o":                       (0.005,   0.015),
    "o1-preview":                   (0.015,   0.06),
    "gemini-1.5-flash":             (0.000075, 0.0003),
    "gemini-1.5-pro":               (0.00125,  0.005),
    "gemini-ultra":                 (0.0,      0.0),   # not public
    "llama3.2:3b":                  (0.0,      0.0),
    "llama3.1:8b":                  (0.0,      0.0),
    "llama3.1:70b":                 (0.0,      0.0),
}


_TOKEN_RE = re.compile(
    r"""
    [A-Za-z]+(?:'[A-Za-z]+)?   # words with optional apostrophe
    | \d+                       # numbers
    | [^\w\s]                   # individual punctuation/symbols
    | [\u4e00-\u9fff]            # each CJK char ≈ 1 token
    """,
    re.VERBOSE,
)

def count_tokens(text: str) -> int:
    """Fast approximation — no model needed. Within ~20% of BPE for English."""
    # Rule of thumb: ~4 chars/token for English, 1 char/token for CJK
    pieces = _TOKEN_RE.findall(text)
    # Long words get split further by BPE; approximate that
    extra = sum(max(0, len(p) // 6) for p in pieces if p.isalpha())
    return len(pieces) + extra
# ─── Complexity Classifier ─────────────────────────────────────────────────────


class ComplexityClassifier:

    SIMPLE_PATTERNS = [
        "translate", "翻译", "summarize in one", "一句话", "what is", "是什么",
        "define", "definition of", "simple question", "yes or no",
        "spell check", "grammar check", "简单", "format this",
    ]

    COMPLEX_PATTERNS = [
        "write a complete", "implement", "analyze in depth", "深度分析", "distributed", "with tests",
        "step by step", "step-by-step", "architecture", "design system", "query info",
        "compare and contrast", "pros and cons", "debug", "优化",
        "proof", "mathematical", "legal", "medical advice",
        "write a research", "comprehensive", "详细",
    ]

    MEDIUM_PATTERNS = [
        "explain", "解释", "how does", "how to", "如何",
        "write a", "help me", "review", "suggest", "list",
    ]

    @staticmethod
    def _compile_patterns(patterns):
        compiled = []
        for pat in patterns:
            # 中文检测（简单 heuristic）
            if re.search(r'[\u4e00-\u9fff]', pat):
                compiled.append(re.compile(re.escape(pat), re.IGNORECASE))
            else:
                # 英文：加词边界
                regex = r'\b' + re.escape(pat) + r'\b'
                compiled.append(re.compile(regex, re.IGNORECASE))
        return compiled

    @staticmethod
    def _count_matches(regex_list, text: str) -> int:
        return sum(1 for r in regex_list if r.search(text))

    @classmethod
    def classify(cls, prompt: str, hint=None):
        if hint:
            return hint

        text = prompt.lower()
        token_count = count_tokens(text)
        logger.info(f"token count: {token_count}")
        # Short prompt
        if token_count < 20:
            if cls._count_matches(cls.SIMPLE_REGEX, text):
                logger.info("✅ short prompt: simple_regex matched")
                return TaskDifficulty.SIMPLE
            if token_count < 10 and not cls._count_matches(cls.MEDIUM_REGEX, text) and not cls._count_matches(cls.COMPLEX_REGEX, text):
                return TaskDifficulty.SIMPLE

        logger.info("✅ complex is judging---")
        # Complex
        complex_hits = cls._count_matches(cls.COMPLEX_REGEX, text)
        if complex_hits >= 2 or token_count > 300:
            return TaskDifficulty.COMPLEX

        # Simple
        simple_hits = cls._count_matches(cls.SIMPLE_REGEX, text)
        if simple_hits >= 1 and token_count < 80:
            return TaskDifficulty.SIMPLE

        # Long → complex
        if token_count > 150:
            return TaskDifficulty.COMPLEX

        return TaskDifficulty.MEDIUM


# Pre-compile patterns once at module level (avoids __func__ hack)
ComplexityClassifier.SIMPLE_REGEX = ComplexityClassifier._compile_patterns(ComplexityClassifier.SIMPLE_PATTERNS)
ComplexityClassifier.COMPLEX_REGEX = ComplexityClassifier._compile_patterns(ComplexityClassifier.COMPLEX_PATTERNS)
ComplexityClassifier.MEDIUM_REGEX = ComplexityClassifier._compile_patterns(ComplexityClassifier.MEDIUM_PATTERNS)


# ─── Provider clients ──────────────────────────────────────────────────────────
class BaseProviderClient:
    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        raise NotImplementedError

    async def stream(self, model: str, messages: list[dict], **kwargs):
        raise NotImplementedError


class AnthropicClient(BaseProviderClient):
    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str):
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def complete(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs) -> dict:
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.BASE_URL, headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return {
                "content": data["content"][0]["text"],
                "usage": {
                    "prompt_tokens": data["usage"]["input_tokens"],
                    "completion_tokens": data["usage"]["output_tokens"],
                }
            }

    async def stream(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs):
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", self.BASE_URL, headers=self.headers, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            if obj.get("type") == "content_block_delta":
                                yield obj["delta"].get("text", "")
                        except Exception:
                            pass


class OpenAIClient(BaseProviderClient):
    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs) -> dict:
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.BASE_URL, headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                }
            }

    async def stream(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs):
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", self.BASE_URL, headers=self.headers, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            pass


class OpenrouterClient(BaseProviderClient):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs) -> dict:
        logger.info("openrouter completing")
        difficulty = kwargs.get("diff", "simple")
        if difficulty == "complex":
            payload = {"model": model, "max_tokens": max_tokens, "messages": messages, "reasoning": {"enabled": True} }
        else:
            payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.BASE_URL, headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data

    async def stream(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs):
        print("stream handle")
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", self.BASE_URL, headers=self.headers, json=payload) as resp:
                async for line in resp.aiter_lines():
                    print(f"line: {line}")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield line
                        except Exception:
                            pass

class GoogleClient(BaseProviderClient):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _url(self, model: str) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

    async def complete(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs) -> dict:
        contents = [{"role": m["role"].replace("assistant", "model"), "parts": [{"text": m["content"]}]}
                    for m in messages]
        payload = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self._url(model), json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return {
                "content": text,
                "usage": {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                }
            }

    async def stream(self, model: str, messages: list[dict], **kwargs):
        result = await self.complete(model, messages, **kwargs)
        yield result["content"]


class OllamaClient(BaseProviderClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def complete(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs) -> dict:
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"num_predict": max_tokens}}
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return {
                "content": data["message"]["content"],
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
            }

    async def stream(self, model: str, messages: list[dict], max_tokens: int = 2048, **kwargs):
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"num_predict": max_tokens}}
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            obj = json.loads(line)
                            yield obj.get("message", {}).get("content", "")
                            if obj.get("done"):
                                break
                        except Exception:
                            pass


# ─── Request / Response models ─────────────────────────────────────────────────
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(pattern="^(user|assistant|system|tool)$")
    content: str | None = None


class GatewayRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: list[ChatMessage]
    difficulty: Optional[TaskDifficulty] = None   # override auto-classification
    provider: Optional[Provider] = None           # force a specific provider
    stream: bool = False
    max_tokens: int = 2048
    privacy_mode: bool = False                    # force LOCAL/Ollama


@dataclass
class RouteDecision:
    difficulty: TaskDifficulty
    provider: Provider
    model: str
    reasoning: str


@dataclass
class GatewayStats:
    total_requests: int = 0
    requests_by_difficulty: dict = field(default_factory=lambda: {d.value: 0 for d in TaskDifficulty})
    requests_by_provider: dict = field(default_factory=lambda: {p.value: 0 for p in Provider})
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    errors: int = 0
    avg_latency_ms: float = 0.0
    _latencies: list = field(default_factory=list)


# ─── Gateway ───────────────────────────────────────────────────────────────────
class LLMGateway:
    def __init__(self):
        self.stats = GatewayStats()
        self.classifier = ComplexityClassifier()
        self._build_clients()
        self.request_log: list[dict] = []

    def _build_clients(self):
        self.clients: dict[Provider, Optional[BaseProviderClient]] = {
            Provider.ANTHROPIC: None,
            Provider.OPENAI: None,
            Provider.GOOGLE: None,
            Provider.OLLAMA: None,
        }
        if key := os.getenv("ANTHROPIC_API_KEY"):
            self.clients[Provider.ANTHROPIC] = AnthropicClient(key)
            logger.info("✅ Anthropic provider ready")
        if key := os.getenv("OPENAI_API_KEY"):
            self.clients[Provider.OPENAI] = OpenAIClient(key)
            logger.info("✅ OpenAI provider ready")
        if key := os.getenv("OPENROUTER_API_KEY"):
            self.clients[Provider.OPENROUTER] = OpenrouterClient(key)
            logger.info("✅ Openrouter provider ready")
        if key := os.getenv("GOOGLE_API_KEY"):
            self.clients[Provider.GOOGLE] = GoogleClient(key)
            logger.info("✅ Google provider ready")
        # Ollama is always available if running locally
        self.clients[Provider.OLLAMA] = OllamaClient(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    def _available_providers(self) -> list[Provider]:
        return [p for p, c in self.clients.items() if c is not None]

    def _select_provider(self, preferred: Optional[Provider]) -> Provider:
        available = self._available_providers()
        if preferred and preferred in available:
            return preferred
        # Priority order: anthropic → openai → google → ollama
        for p in [Provider.ANTHROPIC, Provider.OPENAI, Provider.OPENROUTER, Provider.GOOGLE, Provider.OLLAMA]:
            if p in available:
                return p
        raise HTTPException(503, "No LLM providers configured. Set API keys as env vars.")

    def route(self, req: GatewayRequest) -> RouteDecision:
        # Privacy mode forces local
        if req.privacy_mode:
            diff = TaskDifficulty.LOCAL
            provider = Provider.OLLAMA
        else:
            # Last user message drives classification
            last_user = next(
                (m.content for m in reversed(req.messages) if m.role == "user"), ""
            )
            diff = self.classifier.classify(last_user, req.difficulty)
            provider = self._select_provider(req.provider)

        model = MODEL_REGISTRY[provider][diff]
        reasoning = (
            f"Difficulty: {diff.value} | Provider: {provider.value} | Model: {model}"
        )
        return RouteDecision(difficulty=diff, provider=provider, model=model, reasoning=reasoning)

    def _calc_cost(self, model: str, in_tokens: int, out_tokens: int) -> float:
        rates = MODEL_COST.get(model, (0.0, 0.0))
        return (in_tokens / 1000) * rates[0] + (out_tokens / 1000) * rates[1]

    def _update_stats(self, decision: RouteDecision, usage: dict, latency_ms: float):
        s = self.stats
        s.total_requests += 1
        s.requests_by_difficulty[decision.difficulty.value] += 1
        s.requests_by_provider[decision.provider.value] += 1
        s.total_tokens_in += usage.get("prompt_tokens", 0)
        s.total_tokens_out += usage.get("completion_tokens", 0)
        cost = self._calc_cost(
            decision.model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        s.total_cost_usd += cost
        s._latencies.append(latency_ms)
        s.avg_latency_ms = sum(s._latencies[-100:]) / len(s._latencies[-100:])

    async def complete(self, req: GatewayRequest) -> dict:
        decision = self.route(req)
        client = self.clients[decision.provider]
        messages = [m.model_dump() for m in req.messages]

        t0 = time.perf_counter()
        try:
            result = await client.complete(
                decision.model, messages, max_tokens=req.max_tokens, diff=decision.difficulty.value
            )
        except Exception as e:
            self.stats.errors += 1
            raise HTTPException(502, f"Provider error ({decision.provider.value}): {e}")

        latency_ms = (time.perf_counter() - t0) * 1000
        self._update_stats(decision, result.get("usage", {}), latency_ms)

        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "difficulty": decision.difficulty.value,
            "provider": decision.provider.value,
            "model": decision.model,
            "latency_ms": round(latency_ms),
            "tokens_in": result.get("usage", {}).get("prompt_tokens", 0),
            "tokens_out": result.get("usage", {}).get("completion_tokens", 0),
            "cost_usd": round(self._calc_cost(
                decision.model,
                result.get("usage", {}).get("prompt_tokens", 0),
                result.get("usage", {}).get("completion_tokens", 0),
            ), 6),
        }
        self.request_log.append(log_entry)
        if len(self.request_log) > 500:
            self.request_log = self.request_log[-500:]
            
        return result

    async def stream_complete(self, req: GatewayRequest):
        decision = self.route(req)
        client = self.clients[decision.provider]
        messages = [m.model_dump() for m in req.messages]

        meta = json.dumps({"route": asdict(decision)})
        #yield f"data: {meta}\n\n"

        t0 = time.perf_counter()
        try:
            async for chunk in client.stream(decision.model, messages, max_tokens=req.max_tokens):
                if chunk:
                    logger.info(f"chunk {chunk}")
                    yield f"{chunk}\n\n"
                yield "data: [DONE]\n\n"  # jay added
        except Exception as e:
            self.stats.errors += 1
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        latency_ms = (time.perf_counter() - t0) * 1000
        self._update_stats(decision, {}, latency_ms)
        yield f"data: {json.dumps({'done': True, 'latency_ms': round(latency_ms)})}\n\n"


# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="LLM API Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

gateway = LLMGateway()


@app.post("/v1/chat/completions")
async def chat_completions(req: GatewayRequest):
    """OpenAI-compatible chat completions endpoint with intelligent routing."""
    #body = await req.json()
    #print(body)
    #return {"ok": True}
    #logger.info(f"✅ req data {req}")
    if req.stream:
        return StreamingResponse(
            gateway.stream_complete(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",   # disables nginx buffering
            }
        )
    return await gateway.complete(req)


@app.get("/v1/route/preview")
async def route_preview(prompt: str, provider: Optional[Provider] = None):
    """Preview routing decision without calling any model."""
    diff = ComplexityClassifier.classify(prompt)
    prov = gateway._select_provider(provider)
    model = MODEL_REGISTRY[prov][diff]
    return {
        "prompt_preview": prompt[:120],
        "difficulty": diff.value,
        "provider": prov.value,
        "model": model,
    }


@app.get("/v1/stats")
async def get_stats():
    """Aggregated gateway statistics."""
    s = gateway.stats
    return {
        "total_requests": s.total_requests,
        "requests_by_difficulty": s.requests_by_difficulty,
        "requests_by_provider": s.requests_by_provider,
        "total_tokens_in": s.total_tokens_in,
        "total_tokens_out": s.total_tokens_out,
        "total_cost_usd": round(s.total_cost_usd, 6),
        "avg_latency_ms": round(s.avg_latency_ms, 1),
        "errors": s.errors,
    }


@app.get("/v1/logs")
async def get_logs(limit: int = 50):
    """Recent request logs."""
    return {"logs": gateway.request_log[-limit:]}


@app.get("/v1/providers")
async def list_providers():
    """List configured providers and their models."""
    result = {}
    for provider in gateway._available_providers():
        result[provider.value] = {
            diff.value: model
            for diff, model in MODEL_REGISTRY[provider].items()
        }
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "providers": [p.value for p in gateway._available_providers()]}


if __name__ == "__main__":
    import uvicorn
    print("""
╔══════════════════════════════════════════════════════╗
║        LLM API Gateway  v1.0                         ║
║                                                      ║
║  Set env vars before starting:                       ║
║    ANTHROPIC_API_KEY=sk-ant-...                      ║
║    OPENAI_API_KEY=sk-...                             ║
║    GOOGLE_API_KEY=...                                ║
║    OLLAMA_BASE_URL=http://localhost:11434 (optional) ║
║                                                      ║
║  Endpoints:                                          ║
║    POST /v1/chat/completions  — main chat API        ║
║    GET  /v1/route/preview     — test routing logic   ║
║    GET  /v1/stats             — usage statistics     ║
║    GET  /v1/logs              — recent request logs  ║
║    GET  /v1/providers         — available models     ║
║    GET  /health               — health check         ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("gateway:app", host="0.0.0.0", port=8000, reload=True)
