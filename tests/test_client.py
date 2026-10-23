import asyncio
import json

import httpx
import pytest
from fake_openai import FakeOpenAI, FakeReply

from bakeoff.client import EndpointClient, normalize_base_url, split_think

MSGS = [{"role": "user", "content": "x" * 48}]


def run(coro):
    return asyncio.run(coro)


def make(fake, **kw):
    kw.setdefault("backoff_s", (0.0, 0.0))
    return EndpointClient("http://h/v1", "m", transport=fake.transport, **kw)


async def _chat(fake, n=1, **kw):
    async with make(fake, **kw) as c:
        out = [await c.chat(MSGS, 16) for _ in range(n)]
        return c, out


@pytest.mark.parametrize("url", ["http://h/v1", "http://h/v1/", "http://h", "http://h/"])
def test_normalize(url):
    assert normalize_base_url(url) == ("http://h/v1", "http://h")


def test_split_think():
    assert split_think("<think>hmm</think>\nAnswer") == ("Answer", "hmm")
    assert split_think("<think>never closed") == ("", "never closed")
    assert split_think("plain") == ("plain", "")


def test_stream_timing_and_usage():
    fake = FakeOpenAI(
        responder=lambda m: FakeReply(
            "a b c", prompt_tokens=12, completion_tokens=3, chunk_delay_s=0.01
        )
    )
    _, (r,) = run(_chat(fake))
    assert r.content == "a b c" and r.prompt_tokens == 12 and r.usage_source == "reported"
    assert r.completion_tokens == 3 and r.finish_reason == "stop" and r.error is None
    assert r.ttft_s >= 0.01 and r.t_last >= r.t_first and r.chunk_count == 3
    assert r.total_s >= r.ttft_s
    assert fake.requests[0]["stream"] is True
    assert fake.requests[0]["stream_options"] == {"include_usage": True}
    assert fake.requests[0]["seed"] == 0 and fake.requests[0]["temperature"] == 0.0


def test_reasoning_and_inline_think_separated():
    fake = FakeOpenAI(responder=lambda m: FakeReply("<think>x</think>y", reasoning="r1 r2"))
    _, (r,) = run(_chat(fake))
    assert r.content == "y"
    assert "r1 r2" in r.reasoning and "x" in r.reasoning


def test_ollama_reasoning_field_and_reasoning_tokens():
    chunks = [
        {"choices": [{"delta": {"reasoning": "think"}}]},
        {"choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 9,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        },
    ]
    fake = FakeOpenAI(responder=lambda m: FakeReply(chunks=chunks))
    _, (r,) = run(_chat(fake))
    assert r.reasoning == "think" and r.content == "A"
    assert r.reasoning_tokens == 7 and r.completion_tokens == 9 and r.usage_source == "reported"
    assert r.t_first is not None


def test_unterminated_think_gives_empty_content():
    fake = FakeOpenAI(responder=lambda m: FakeReply("<think>cut off", finish_reason="length"))
    _, (r,) = run(_chat(fake))
    assert r.content == "" and r.reasoning == "cut off" and r.finish_reason == "length"


def test_stream_options_rejected_fallback():
    fake = FakeOpenAI(support_stream_options=False)
    c, (r1, r2) = run(_chat(fake, n=2))
    assert r1.content == "ok" and r1.usage_source == "estimated"
    assert r1.prompt_tokens is None and r1.completion_tokens == r1.chunk_count == 1
    assert c.usage_supported is False
    assert "stream_options" in fake.requests[0]
    assert "stream_options" not in fake.requests[1]
    assert len(fake.requests) == 3  # first call retried once, second call sent once
    assert "stream_options" not in fake.requests[2] and r2.error is None


def test_estimated_tokens_count_reasoning_and_content_chunks():
    fake = FakeOpenAI(
        support_stream_options=False, responder=lambda m: FakeReply("a b", reasoning="r1 r2 r3")
    )
    _, (r,) = run(_chat(fake))
    assert r.usage_source == "estimated" and r.chunk_count == 5 and r.completion_tokens == 5
    assert r.content == "a b" and r.reasoning == "r1 r2 r3"


def _role():
    return {"choices": [{"delta": {"role": "assistant"}}]}


def _content(t):
    return {"choices": [{"delta": {"content": t}}]}


def _fin():
    return {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def _usage():
    return {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _sse(*objs):
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs)


def test_empty_streams_are_errors_not_retried():
    for reply in (
        FakeReply(raw=b"data: [DONE]\n\n"),
        FakeReply(raw=b""),
        FakeReply(raw=b'{"choices": []}'),
        FakeReply(raw=_sse(_role(), _fin()) + b"data: [DONE]\n\n"),
        FakeReply(raw=_sse(_role())),
    ):
        fake = FakeOpenAI(responder=lambda m, reply=reply: reply)
        _, (r,) = run(_chat(fake))
        assert r.error == "empty stream" and r.status is None
        assert len(fake.requests) == 1


def test_in_stream_error_event():
    ev = {"error": {"message": "boom " + "x" * 600, "type": "server_error"}}
    fake = FakeOpenAI(responder=lambda m: FakeReply(chunks=[_content("a"), ev]))
    _, (r,) = run(_chat(fake))
    assert r.error.startswith("boom") and len(r.error) == 500 and len(fake.requests) == 1


def test_ttft_skips_role_only_first_chunk():
    fake = FakeOpenAI(
        responder=lambda m: FakeReply(chunks=[_role(), _content("hi"), _fin()], chunk_delay_s=0.05)
    )
    _, (r,) = run(_chat(fake))
    assert r.content == "hi" and r.ttft_s >= 0.095  # role chunk (delay 1) + content (delay 2)


def test_t_last_ignores_delayed_usage_chunk():
    fake = FakeOpenAI(
        responder=lambda m: FakeReply(chunks=[_content("hi"), _fin(), _usage()], chunk_delay_s=0.03)
    )
    _, (r,) = run(_chat(fake))
    assert r.t_last == r.t_first  # one content chunk; later chunks do not move t_last
    assert r.usage_source == "reported"


def test_raw_sse_variants_parse():
    raw = (
        b": ping\n\n"
        b'data:{"choices":[{"delta":{"content":"he"}}]}\n\n'
        b"\n\n"
        b": ping\n"
        b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n'
        b"data: not json\n\n"
        b"data: [1, 2]\n\n"
        b'data: {"choices":[null,{"delta":null}]}\n\n'
        b"data: [DONE]\n\n"
    )
    fake = FakeOpenAI(responder=lambda m: FakeReply(raw=raw))
    _, (r,) = run(_chat(fake))
    assert r.error is None and r.content == "hello" and r.finish_reason == "stop"


def test_decoding_error_becomes_error():
    def handler(request):
        return httpx.Response(200, headers={"content-encoding": "gzip"}, content=b"not gzip at all")

    async def go():
        async with EndpointClient(
            "http://h", "m", transport=httpx.MockTransport(handler), backoff_s=()
        ) as c:
            return await c.chat(MSGS, 4)

    r = run(go())
    assert r.error and r.status is None


def test_dangling_close_tag_and_reasoning_join():
    assert split_think("reasoning</think>answer") == ("answer", "reasoning")
    assert split_think("a <think>b</think> c") == ("a <think>b</think> c", "")
    fake = FakeOpenAI(responder=lambda m: FakeReply("<think>x</think>y", reasoning="r"))
    _, (r,) = run(_chat(fake))
    assert r.reasoning == "r\nx"


def test_400_without_stream_options_is_not_fallback():
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=400, body="bad"))
    c, (r,) = run(_chat(fake))
    assert c.usage_supported is True and len(fake.requests) == 1


def test_retries_5xx_then_succeeds():
    fake = FakeOpenAI(fail_first=2)
    _, (r,) = run(_chat(fake))
    assert r.error is None and r.content == "ok" and len(fake.requests) == 3


def test_gives_up_after_retries():
    fake = FakeOpenAI(fail_first=5)
    _, (r,) = run(_chat(fake))
    assert r.error and r.status == 503 and len(fake.requests) == 3


def test_4xx_not_retried():
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=400, body="bad"))
    _, (r,) = run(_chat(fake))
    assert len(fake.requests) == 1 and r.status == 400 and r.error == "bad"


def test_4xx_body_truncated():
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=422, body="z" * 900))
    _, (r,) = run(_chat(fake))
    assert len(r.error) == 500


def test_midstream_failure_retries_fresh():
    calls = []

    def responder(m):
        calls.append(1)
        if len(calls) == 1:
            return FakeReply("a b c", fail_after_chunks=2)
        return FakeReply("a b c")

    fake = FakeOpenAI(responder=responder)
    _, (r,) = run(_chat(fake))
    assert r.error is None and r.content == "a b c" and len(fake.requests) == 2


def test_midstream_failure_exhausts_to_error():
    fake = FakeOpenAI(responder=lambda m: FakeReply("a b c", fail_after_chunks=2))
    _, (r,) = run(_chat(fake))
    assert r.error and len(fake.requests) == 3


def test_transport_error_returns_result():
    def boom(request):
        raise httpx.ConnectError("down")

    async def go():
        async with EndpointClient(
            "http://h", "m", transport=httpx.MockTransport(boom), backoff_s=(0.0,)
        ) as c:
            return await c.chat(MSGS, 4)

    r = run(go())
    assert r.error and r.status is None


def test_api_key_header_only_when_given():
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    async def go(key):
        async with EndpointClient(
            "http://h", "m", api_key=key, transport=httpx.MockTransport(handler)
        ) as c:
            return await c.list_models()

    assert run(go("sekret")) == ["m"] and run(go(None)) == ["m"]
    assert seen == ["Bearer sekret", None]


def test_list_models_and_root_probes_use_root():
    fake = FakeOpenAI(models=["a", "b"], extras={"/version": {"version": "0.9.1"}})

    async def go():
        async with EndpointClient("http://h/v1/", "a", transport=fake.transport) as c:
            return await c.list_models(), await c.runtime_info("vllm")

    models, info = run(go())
    assert models == ["a", "b"] and info["version"] == "0.9.1"


def test_runtime_info_vllm_max_model_len():
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 8192}]})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.9"})
        return httpx.Response(404)

    async def go():
        async with EndpointClient("http://h", "m", transport=httpx.MockTransport(handler)) as c:
            return await c.runtime_info("vllm")

    assert run(go()) == {"version": "0.9", "context_length": 8192, "size_vram": None}


def test_runtime_info_llamacpp():
    fake = FakeOpenAI(
        extras={
            "/props": {
                "build_info": "b1234",
                "default_generation_settings": {"n_ctx": 4096},
            }
        }
    )

    async def go():
        async with make(fake) as c:
            return await c.runtime_info("llamacpp")

    assert run(go()) == {"version": "b1234", "context_length": 4096, "size_vram": None}


def test_runtime_info_ollama():
    fake = FakeOpenAI(
        extras={
            "/api/version": {"version": "0.5.1"},
            "/api/ps": {
                "models": [
                    {"name": "other:latest", "context_length": 1, "size_vram": 1},
                    {"name": "m", "context_length": 4096, "size_vram": 123},
                ]
            },
        }
    )

    async def go():
        async with make(fake) as c:
            return await c.runtime_info("ollama")

    assert run(go()) == {"version": "0.5.1", "context_length": 4096, "size_vram": 123}


def test_runtime_info_missing_endpoints():
    fake = FakeOpenAI()

    async def go():
        async with make(fake) as c:
            return [await c.runtime_info(r) for r in ("vllm", "llamacpp", "ollama", "weird")]

    for info in run(go()):
        assert info == {"version": None, "context_length": None, "size_vram": None}


def test_fake_server_basics():
    fake = FakeOpenAI(extras={"/x": "text"})

    async def probe():
        async with httpx.AsyncClient(transport=fake.transport, base_url="http://h") as c:
            return (await c.get("/nope")).status_code, (await c.get("/x")).text

    assert run(probe()) == (404, "text")
    needle = [{"role": "user", "content": "The access code for the vault is FOO-1. Done"}]

    async def go():
        async with make(fake) as cl:
            return await cl.chat(needle, 8)

    assert run(go()).content == "FOO-1"
    assert fake.requests[0]["messages"] == needle


def test_fake_context_limit():
    long = [{"role": "user", "content": "x" * 400}]  # 100 est tokens

    async def go(**kw):
        fake = FakeOpenAI(context_limit=50, **kw)
        async with make(fake) as c:
            return await c.chat(long, 8)

    r = run(go())
    assert r.status == 400 and "maximum context length" in r.error
    r = run(go(truncate_silently=True))
    assert r.error is None and r.prompt_tokens == 50


def test_error_text_hides_the_endpoint_before_truncation():
    url = "http://lab-box-7.lan:11434"  # scrub-ok
    body = "x" * 495 + f"{url}/v1/chat"  # a late URL must not survive as a cut-off fragment
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=400, body=body))

    async def go():
        async with EndpointClient(f"{url}/v1", "m", transport=fake.transport) as c:
            return await c.chat(MSGS, 4)

    r = run(go())
    assert len(r.error) <= 500 and "lab-box" not in r.error and "http://" not in r.error
    assert r.error.endswith("<endpoint>/v1/chat"[:5]) or "<endpoint>" in r.error


def test_error_text_hides_bare_netloc_root_and_key():
    key = "sk-secret-1234"
    body = f"cannot reach lab-box-7.lan:11434 at http://lab-box-7.lan:11434 with {key}"  # scrub-ok
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=400, body=body))

    async def go():
        async with EndpointClient(
            "http://lab-box-7.lan:11434/v1",  # scrub-ok
            "m",
            api_key=key,
            transport=fake.transport,
        ) as c:
            return await c.chat(MSGS, 4)

    r = run(go())
    assert r.error == "cannot reach <endpoint> at <endpoint> with <endpoint>"


def test_transport_error_text_is_redacted():
    def boom(request):
        raise httpx.ConnectError("connect to http://lab-box-7.lan:11434 failed")  # scrub-ok

    async def go():
        async with EndpointClient(
            "http://lab-box-7.lan:11434",  # scrub-ok
            "m",
            transport=httpx.MockTransport(boom),
            backoff_s=(),
        ) as c:
            return await c.chat(MSGS, 4)

    assert "lab-box" not in run(go()).error


def test_short_host_does_not_mangle_error_text():
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=400, body="the hash h failed"))
    _, (r,) = run(_chat(fake))
    assert r.error == "the hash h failed"


CACHE_LINE = (
    'vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",'
    'num_gpu_blocks="27000"} 1.0\n'
)


def _cache_config(text, status=200):
    async def go():
        fake = FakeOpenAI(extras={"/metrics": text} if status == 200 else None)
        async with make(fake) as c:
            return await c.cache_config()

    return run(go())


def test_cache_config_parsed():
    got = _cache_config("# HELP vllm:cache_config_info x\n" + CACHE_LINE)
    assert got["block_size"] == "16"
    assert got["num_gpu_blocks"] == "27000"
    assert got["cache_dtype"] == "auto"


def test_cache_config_missing():
    assert _cache_config("vllm:kv_cache_usage_perc 0.5\n") is None
    assert _cache_config("", status=404) is None


def test_cache_config_escaped_labels():
    line = 'vllm:cache_config_info{prefix="a\\"b,c",block_size="16",num_gpu_blocks="27000"} 1.0\n'
    got = _cache_config(line)
    assert got["prefix"] == 'a"b,c'
    assert got["block_size"] == "16"
    assert got["num_gpu_blocks"] == "27000"
