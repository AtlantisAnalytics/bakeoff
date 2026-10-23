"""Async OpenAI-compatible streaming client with retries and runtime probes."""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
CACHE_CONFIG_METRIC = "vllm:cache_config_info"
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
ERROR_BODY_MAX = 500


def normalize_base_url(url: str) -> tuple[str, str]:
    """Return (api_base ending in /v1, root without /v1)."""
    root = url.strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    root = root.rstrip("/")
    return root + "/v1", root


def split_think(content: str) -> tuple[str, str]:
    """Split a leading <think> block off content: (answer, think_text)."""
    stripped = content.lstrip()
    if not stripped.startswith(THINK_OPEN):
        if THINK_CLOSE in content and THINK_OPEN not in content.partition(THINK_CLOSE)[0]:
            head, _, tail = content.partition(THINK_CLOSE)
            return tail.strip(), head.strip()  # template put <think> in the prompt
        return content, ""
    rest = stripped[len(THINK_OPEN) :]
    head, sep, tail = rest.partition(THINK_CLOSE)
    if not sep:
        return "", rest.strip()
    return tail.strip(), head.strip()


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_source: str = "estimated"
    chunk_count: int = 0
    t_start: float = 0.0
    t_first: float | None = None
    t_last: float | None = None
    status: int | None = None
    error: str | None = None

    @property
    def ttft_s(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_start

    @property
    def total_s(self) -> float | None:
        return None if self.t_last is None else self.t_last - self.t_start


class _Retryable(Exception):
    def __init__(self, error: str, status: int | None = None):
        super().__init__(error)
        self.error = error
        self.status = status


class _StreamOptionsRejected(Exception):
    pass


class EndpointClient:
    """Async client for one OpenAI-compatible endpoint.

    `timeout_s` is applied per operation (connect, each read, write), as httpx does; it is not a
    cap on the total duration of a streamed response.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff_s: tuple[float, ...] = (1.0, 2.0),
    ):
        self.api_base, self.root = normalize_base_url(base_url)
        netloc = urlsplit(self.root if "//" in self.root else "//" + self.root).netloc
        # URL forms are replaced anywhere; the bare netloc only at token edges; very short
        # strings are never replaced (they would mangle ordinary text)
        pats = [re.escape(t) for t in {self.api_base, self.root, base_url.strip()} if t]
        if len(netloc) >= 4:
            pats.append(rf"(?<![\w.-]){re.escape(netloc)}(?![\w.-])")
        if api_key and len(api_key) >= 4:
            pats.append(re.escape(api_key))
        pats.sort(key=len, reverse=True)
        self._own = re.compile("|".join(pats), re.I) if pats else None
        self.model = model
        self.backoff_s = backoff_s
        self.usage_supported = True
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(timeout_s), headers=headers
        )

    async def __aenter__(self) -> "EndpointClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._http.aclose()

    def _clean(self, text: str) -> str:
        """Hide this client's own endpoint (and key) in text that may be stored, then truncate."""
        if self._own is not None:
            text = self._own.sub("<endpoint>", text)
        return text[:ERROR_BODY_MAX]

    async def _get_json(self, url: str):
        try:
            r = await self._http.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def list_models(self) -> list[str]:
        r = await self._http.get(f"{self.api_base}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        extra: dict | None = None,
    ) -> ChatResult:
        last = ChatResult()
        for attempt in range(len(self.backoff_s) + 1):
            if attempt:
                await asyncio.sleep(self.backoff_s[attempt - 1])
            try:
                try:
                    return await self._chat_once(
                        messages, max_tokens, temperature, seed, extra, self.usage_supported
                    )
                except _StreamOptionsRejected:
                    self.usage_supported = False
                    return await self._chat_once(
                        messages, max_tokens, temperature, seed, extra, False
                    )
            except _Retryable as e:
                last = ChatResult(status=e.status, error=e.error, t_start=time.perf_counter())
        return last

    async def _chat_once(
        self, messages, max_tokens, temperature, seed, extra, with_usage: bool
    ) -> ChatResult:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
        }
        if with_usage:
            body["stream_options"] = {"include_usage": True}
        if extra:
            body.update(extra)
        res = ChatResult()
        raw_content: list[str] = []
        reasoning: list[str] = []
        usage: dict | None = None
        res.t_start = time.perf_counter()
        try:
            async with self._http.stream(
                "POST", f"{self.api_base}/chat/completions", json=body
            ) as resp:
                res.status = resp.status_code
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    if resp.status_code == 400 and with_usage and "stream_options" in text.lower():
                        raise _StreamOptionsRejected
                    if resp.status_code >= 500:
                        raise _Retryable(self._clean(text), resp.status_code)
                    res.error = self._clean(text)
                    return res
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    err = obj.get("error")
                    if err:
                        msg = err.get("message") if isinstance(err, dict) else err
                        res.status = None
                        res.error = self._clean(str(msg if msg else err))
                        return res
                    if isinstance(obj.get("usage"), dict) and obj["usage"]:
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            delta = {}
                        c = delta.get("content") or ""
                        r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if c or r:
                            now = time.perf_counter()
                            if res.t_first is None:
                                res.t_first = now
                            res.t_last = now
                            res.chunk_count += 1
                        if c:
                            raw_content.append(c)
                        if r:
                            reasoning.append(r)
                        if choice.get("finish_reason"):
                            res.finish_reason = choice["finish_reason"]
        except (_StreamOptionsRejected, _Retryable):
            raise
        except httpx.TimeoutException as e:
            raise _Retryable(f"timeout: {type(e).__name__}") from e
        except httpx.TransportError as e:
            raise _Retryable(self._clean(f"transport error: {type(e).__name__}: {e}")) from e
        except httpx.HTTPError as e:
            res.status = None
            res.error = self._clean(f"http error: {type(e).__name__}: {e}")
            return res
        if res.t_first is None and not usage:
            res.status = None
            res.error = "empty stream"
            return res
        content, think = split_think("".join(raw_content))
        res.content = content
        res.reasoning = "\n".join(p for p in ("".join(reasoning), think) if p)
        if usage:
            res.usage_source = "reported"
            res.prompt_tokens = usage.get("prompt_tokens")
            res.completion_tokens = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details") or {}
            res.reasoning_tokens = details.get("reasoning_tokens")
        else:
            res.usage_source = "estimated"
            res.completion_tokens = res.chunk_count
            res.prompt_tokens = None
        return res

    async def runtime_info(self, runtime: str) -> dict:
        info = {"version": None, "context_length": None, "size_vram": None}
        try:
            if runtime == "vllm":
                v = await self._get_json(f"{self.root}/version")
                info["version"] = _str_or_none((v or {}).get("version"))
                models = await self._get_json(f"{self.api_base}/models")
                data = (models or {}).get("data") or []
                mine = [m for m in data if m.get("id") == self.model] or data
                for m in mine:
                    if isinstance(m.get("max_model_len"), int):
                        info["context_length"] = m["max_model_len"]
                        break
            elif runtime == "llamacpp":
                p = await self._get_json(f"{self.root}/props") or {}
                info["version"] = _str_or_none(p.get("build_info"))
                n_ctx = (p.get("default_generation_settings") or {}).get("n_ctx", p.get("n_ctx"))
                info["context_length"] = n_ctx if isinstance(n_ctx, int) else None
            elif runtime == "exllamav3":  # TabbyAPI: the loaded model's card; no version
                card = await self._get_json(f"{self.api_base}/model") or {}
                n_ctx = (card.get("parameters") or {}).get("max_seq_len")
                info["context_length"] = n_ctx if isinstance(n_ctx, int) else None
            elif runtime == "ollama":
                v = await self._get_json(f"{self.root}/api/version")
                info["version"] = _str_or_none((v or {}).get("version"))
                ps = await self._get_json(f"{self.root}/api/ps") or {}
                names = {self.model, f"{self.model}:latest"}
                for m in ps.get("models") or []:
                    if m.get("name") in names or m.get("model") in names:
                        cl, sv = m.get("context_length"), m.get("size_vram")
                        info["context_length"] = cl if isinstance(cl, int) else None
                        info["size_vram"] = sv if isinstance(sv, int) else None
                        break
        except (AttributeError, TypeError):
            return {"version": None, "context_length": None, "size_vram": None}
        return info

    async def cache_config(self) -> dict[str, str] | None:
        """vLLM's `cache_config_info` labels (its CacheConfig, e.g. num_gpu_blocks, block_size)."""
        try:
            r = await self._http.get(f"{self.root}/metrics")
            if r.status_code != 200:
                return None
            for line in r.text.splitlines():
                if line.startswith(CACHE_CONFIG_METRIC + "{"):
                    body = line[len(CACHE_CONFIG_METRIC) + 1 :]
                    return {k: _unescape_label(v) for k, v in _LABEL.findall(body)}
        except httpx.HTTPError:
            return None
        return None


def _unescape_label(value: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), value)


def _str_or_none(v) -> str | None:
    return None if v is None else str(v)
