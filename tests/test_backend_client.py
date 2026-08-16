"""backend_client 单元测试（httpx 可用时运行）。"""

import time

import pytest

httpx = pytest.importorskip("httpx")

import backend_client as bc


def test_extract_data_ok():
    assert bc.extract_data({"code": 200, "data": {"a": 1}}) == {"a": 1}


def test_extract_data_error():
    with pytest.raises(RuntimeError):
        bc.extract_data({"code": 500, "message": "boom"})


def test_extract_records_list_and_page():
    assert bc.extract_records([1, 2]) == [1, 2]
    assert bc.extract_records({"records": [1]}) == [1]
    assert bc.extract_records({"total": 5}) == []


def test_session_expiry(monkeypatch):
    bc.set_session("s1", {"token": "t", "expires_at": time.time() + 100})
    assert bc.get_session("s1") is not None
    future = time.time() + 200
    monkeypatch.setattr(time, "time", lambda: future)
    assert bc.get_session("s1") is None


def test_purge_expired_sessions():
    bc.set_session("s2", {"token": "t", "expires_at": time.time() + 100})
    bc.set_session("s3", {"token": "t", "expires_at": time.time() - 10})
    assert bc.purge_expired_sessions() == 1
    assert bc.get_session("s2") is not None
    assert bc.get_session("s3") is None


def test_get_auth_headers_requires_login():
    with pytest.raises(PermissionError):
        bc.get_auth_headers("no-such-session")


def test_is_retryable(monkeypatch):
    from httpx import HTTPStatusError, Request, Response
    req = Request("GET", "http://x")
    assert bc._is_retryable(httpx.TimeoutException("t"))
    assert bc._is_retryable(httpx.ConnectError("c", request=req))
    r500 = HTTPStatusError("e", request=req, response=Response(500, request=req))
    assert bc._is_retryable(r500)
    r400 = HTTPStatusError("e", request=req, response=Response(400, request=req))
    assert not bc._is_retryable(r400)
