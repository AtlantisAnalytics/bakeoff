"""Fake OpenAI-compatible server for tests, built on httpx.MockTransport."""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

NEEDLE_RE = re.compile(r"access code for the vault is (\S+)\.")


@dataclass
class FakeReply:
    content: str = "ok"
    reasoning: str = ""
    finish_reason: str | None = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: int = 200
    body: str | None = None  # error body for non-200 statuses
    chunk_delay_s: float = 0.0
    chunks: list[dict] | None = None  # raw chunk payloads, sent verbatim instead of content
    raw: bytes | None = None  # raw response body, sent verbatim as the SSE stream
    fail_after_chunks: int | None = None  # break the stream after this many chunks


def default_responder(messages: list[dict]) -> FakeReply:
    text = "\n".join(str(m.get("content", "")) for m in messages)
    m = NEEDLE_RE.search(text)
    return FakeReply(content=m.group(1) if m else "ok")


def _words(text: str) -> list[str]:
    parts = re.findall(r"\S+\s*", text)
    return parts


@dataclass
class FakeOpenAI:
    models: list[str] = field(default_factory=lambda: ["served-model-name"])
    responder: Callable[[list[dict]], FakeReply] | None = None
    support_stream_options: bool = True
    context_limit: int | None = None
    truncate_silently: bool = False
    fail_first: int = 0
    extras: dict[str, dict | str] | None = None

    def __post_init__(self):
        self.requests: list[dict] = []
        self._chat_calls = 0
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/v1/models":
            data = [{"id": m, "object": "model"} for m in self.models]
            return httpx.Response(200, json={"object": "list", "data": data})
        if request.method == "POST" and path == "/v1/chat/completions":
            return self._chat(json.loads(request.content))
        if request.method == "GET" and self.extras and path in self.extras:
            body = self.extras[path]
            if isinstance(body, str):
                return httpx.Response(200, text=body)
            return httpx.Response(200, json=body)
        return httpx.Response(404, text="not found")

    def _chat(self, body: dict) -> httpx.Response:
        self.requests.append(body)
        self._chat_calls += 1
        if self._chat_calls <= self.fail_first:
            return httpx.Response(503, text="unavailable")
        if "stream_options" in body and not self.support_stream_options:
            return httpx.Response(400, text="Unknown parameter: stream_options")
        messages = body.get("messages", [])
        prompt_tokens_est = sum(len(str(m.get("content", ""))) for m in messages) // 4
        if self.context_limit is not None and prompt_tokens_est > self.context_limit:
            if not self.truncate_silently:
                return httpx.Response(400, text="This model's maximum context length is exceeded")
            prompt_tokens_est = self.context_limit
        reply = (self.responder or default_responder)(messages)
        if reply.status != 200:
            return httpx.Response(reply.status, text=reply.body or "error")
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=self._stream(reply, prompt_tokens_est, include_usage),
        )

    async def _stream(self, reply: FakeReply, prompt_est: int, include_usage: bool):
        def sse(obj) -> bytes:
            return b"data: " + json.dumps(obj).encode() + b"\n\n"

        def chunk(delta: dict, finish=None) -> dict:
            return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        if reply.raw is not None:
            yield reply.raw
            return
        sent = 0
        n_content = 0
        if reply.chunks is not None:
            payloads = reply.chunks
        else:
            payloads = [chunk({"reasoning_content": w}) for w in _words(reply.reasoning)]
            words = _words(reply.content)
            n_content = len(words) + len(_words(reply.reasoning))
            payloads += [chunk({"content": w}) for w in words]
            payloads.append(chunk({}, reply.finish_reason))
        for p in payloads:
            if reply.fail_after_chunks is not None and sent >= reply.fail_after_chunks:
                raise httpx.ReadError("stream broke")
            if reply.chunk_delay_s:
                await asyncio.sleep(reply.chunk_delay_s)
            yield sse(p)
            sent += 1
        has_usage = any("usage" in p for p in payloads)
        if include_usage and not has_usage:
            pt = reply.prompt_tokens if reply.prompt_tokens is not None else prompt_est
            if self.truncate_silently and self.context_limit is not None:
                pt = min(pt, self.context_limit)
            ct = reply.completion_tokens if reply.completion_tokens is not None else n_content
            yield sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": pt + ct,
                    },
                }
            )
        yield b"data: [DONE]\n\n"
