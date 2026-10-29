from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from principia.llm import LLMClient, LLMConfig
from principia.local.service import LocalDiscoveryService
from principia.providers import ModelPolicy, OpenAICompatibleProvider
from principia.providers.credentials import ProviderCredentialStore


@pytest.mark.parametrize("model", ["gpt-6-astra", "fixture-model"])
def test_chat_payload_matches_model_capabilities(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        "principia.llm.httpx.Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setenv("PRINCIPIA_THINKING_MODE", "disabled")
    client = LLMClient(
        LLMConfig.from_model(
            f"custom:{model}", api_key="fixture-secret", base_url="https://example.test/v1"
        )
    )
    assert client.chat_json("Return JSON.", "Test", max_tokens=1024) == {"ok": True}
    payload = requests[0]
    assert payload["model"] == model
    assert payload["response_format"] == {"type": "json_object"}
    if model == "gpt-6-astra":
        assert payload["max_completion_tokens"] == 1024
        assert not {"max_tokens", "temperature", "thinking"} & payload.keys()
    else:
        assert payload["max_tokens"] == 1024
        assert payload["temperature"] == 0.2
        assert payload["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("always_truncated", [False, True])
def test_astra_truncation_retries_keep_completion_token_parameter(
    always_truncated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        truncated = always_truncated or len(requests) == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "partial" if truncated else '{"ok": true}'},
                        "finish_reason": "length" if truncated else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        "principia.llm.httpx.Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    client = LLMClient(
        LLMConfig.from_model(
            "custom:gpt-6-astra",
            api_key="fixture-secret",
            base_url="https://example.test/v1",
            max_retries=1,
        )
    )
    if always_truncated:
        with pytest.raises(RuntimeError, match="output reached"):
            client.chat_json("Return JSON.", "Test", max_tokens=1024)
    else:
        assert client.chat_json("Return JSON.", "Test", max_tokens=1024) == {"ok": True}
    assert [p["max_completion_tokens"] for p in requests] == [1024, 2048]
    assert all("max_tokens" not in p and "temperature" not in p for p in requests)
    assert client.usage_totals()["calls"] == 2
    assert client.usage_totals()["completion_tokens"] == 40


@pytest.mark.parametrize("model", ["gpt-6-astra", "fixture-model"])
def test_local_discovery_provider_uses_model_parameters(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRINCIPIA_THINKING_MODE", raising=False)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    provider = OpenAICompatibleProvider(
        ModelPolicy(
            mode="remote",
            provider="custom",
            model=model,
            base_url="https://example.test/v1",
            remote_egress_confirmed=True,
        ),
        api_key="fixture-secret",
        thinking_budget=2048,
        transport=httpx.MockTransport(handler),
    )
    try:
        provider._request([{"role": "user", "content": "JSON"}], max_tokens=4096)
    finally:
        provider.close()
    payload = requests[0]
    if model == "gpt-6-astra":
        assert payload["max_completion_tokens"] == 4096
        assert not {"temperature", "max_tokens", "thinking_budget"} & payload.keys()
    else:
        assert payload["max_tokens"] == 4096
        assert payload["temperature"] == 0.1
        assert payload["thinking_budget"] == 2048


@pytest.mark.parametrize("status_code, expected", [(200, "connected"), (429, "rate_limited")])
def test_environment_gateway_connection_does_not_persist_a_siliconflow_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int, expected: str
) -> None:
    monkeypatch.setenv("PRINCIPIA_LLM_BASE_URL", "https://gateway.example.test/v1")
    monkeypatch.setenv("PRINCIPIA_LLM_API_KEY", "fixture-secret")
    monkeypatch.setenv("PRINCIPIA_LLM_MODEL", "gpt-6-astra")
    service = object.__new__(LocalDiscoveryService)
    service.credentials = ProviderCredentialStore(tmp_path)
    requests = []

    def fake_post(url: str, **kwargs) -> httpx.Response:
        requests.append(url)
        assert kwargs["follow_redirects"] is False
        assert kwargs["json"]["model"] == "gpt-6-astra"
        assert kwargs["json"]["max_completion_tokens"] == 64
        assert "temperature" not in kwargs["json"]
        return httpx.Response(status_code, json={"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr("principia.local.service.httpx.post", fake_post)
    result = service.test_provider_connection("siliconflow")
    assert result["category"] == expected
    assert result["ok"] is (status_code == 200)
    assert requests == ["https://gateway.example.test/v1/chat/completions"]
    assert not service.credentials.path.exists()
    assert service.provider_profile().base_url == "https://gateway.example.test/v1"


def test_environment_gateway_redirect_is_not_reported_as_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRINCIPIA_LLM_BASE_URL", "https://gateway.example.test/v1")
    monkeypatch.setenv("PRINCIPIA_LLM_API_KEY", "fixture-secret")
    service = object.__new__(LocalDiscoveryService)
    service.credentials = ProviderCredentialStore(tmp_path)
    monkeypatch.setattr(
        "principia.local.service.httpx.post",
        lambda *args, **kwargs: httpx.Response(302, headers={"Location": "https://other.test"}),
    )
    result = service.test_provider_connection("siliconflow")
    assert result["category"] == "provider_unavailable"
    assert not result.get("ok", False)
    assert not service.credentials.path.exists()
