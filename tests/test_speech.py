"""语音 Token 请求的网络容错测试，不访问阿里云。"""

import asyncio

import edge_tts
import httpx

import speech


def test_token_transport_error_is_retried(monkeypatch):
    attempts = []
    client_options = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"Token": {"Id": "test-token", "ExpireTime": 4_000_000_000}}

    class FakeClient:
        def __init__(self, **kwargs):
            client_options.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ConnectError("temporary TLS EOF")
            return FakeResponse()

    monkeypatch.setattr(speech, "_check_credentials", lambda: None)
    monkeypatch.setattr(speech, "_token_cache", {"token": "", "expire_time": 0})
    monkeypatch.setattr(speech.httpx, "Client", FakeClient)
    monkeypatch.setattr(speech, "MAX_RETRIES", 3)
    monkeypatch.setattr(speech, "RETRY_DELAY", 0)

    assert speech._get_aliyun_token() == "test-token"
    assert len(attempts) == 2
    assert client_options[0]["trust_env"] is False
    assert client_options[0]["http2"] is False
    assert client_options[0]["verify"] is speech.SSL_VERIFY


def test_tts_uses_python_api_without_cli(monkeypatch):
    class FakeCommunicate:
        def __init__(self, *, text, voice, rate):
            assert text == "测试文本"
            assert voice == speech.TTS_VOICE
            assert rate == speech.TTS_RATE

        async def save(self, output_file):
            with open(output_file, "wb") as stream:
                stream.write(b"fake-mp3")

    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)
    assert asyncio.run(speech.synthesize_speech("测试文本")) == b"fake-mp3"
