"""Faithful bounded readers exercise framing without socket timing dependencies."""

from __future__ import annotations

import gzip
from email.message import Message
from io import BytesIO
from types import SimpleNamespace

import pytest

import model_router as mr
import model_router.headers as router_headers
import model_router.http as router_http


class BodyReader(BytesIO):
    def __init__(self, raw: bytes, fragment: int | None = None) -> None:
        super().__init__(raw)
        self.fragment = fragment
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        assert size >= 0, "Unbounded/negative stream read"
        self.read_sizes.append(size)
        return super().read(min(size, self.fragment) if self.fragment else size)

    def readline(self, size: int = -1) -> bytes:
        assert 0 < size <= mr.MAX_CHUNK_LINE + 1, "Unbounded framing read"
        return super().readline(size)


def body_handler(
    raw: bytes,
    headers: list[tuple[str, str]],
    limit: int = 64,
    fragment: int | None = None,
) -> mr.Handler:
    handler = mr.Handler.__new__(mr.Handler)
    handler.headers = Message()
    for name, value in headers:
        handler.headers[name] = value
    handler.app = SimpleNamespace(config=SimpleNamespace(max_body_bytes=limit))
    handler.rfile = BodyReader(raw, fragment)
    return handler


CHUNKED = [("Transfer-Encoding", "chunked")]




@pytest.mark.parametrize("raw", [
    b"2\n{}\r\n0\r\n\r\n", b"2\r\n{", b"2\r\n{}", b"2\r\n{}\r",
    b"2\r\n{}xx0\r\n\r\n", b"2\r\n{}\n0\r\n\r\n", b"2\r\n{}\r\n",
    b"0\r\n", b"0\r\n\n", b"0\r\nX-Foo: bar\r\n",
    b"0\r\nBad Trailer\r\n\r\n", b"0\r\n X-Foo: folded\r\n\r\n",
    b"0\r\nX-Foo: bad\x00\r\n\r\n", b"0\r\nContent-Length: 0\r\n\r\n",
    b"0\r\nTransfer-Encoding: identity\r\n\r\n",
    b"0\r\nContent-Encoding: gzip\r\n\r\n",
])
def test_malformed_or_truncated_chunks(raw: bytes) -> None:
    handler = body_handler(raw, CHUNKED)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 400


@pytest.mark.parametrize("value", [
    "-1", "+1", "1_0", "0x10", "1, 1", "", "١", "1.0", "\v1", "1\v", "\xa01",
])
def test_invalid_content_length_is_rejected_before_read(value: str) -> None:
    handler = body_handler(b"x" * 100_000, [("Content-Length", value)])
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 400
    assert handler.rfile.tell() == 0




def test_truncated_fixed_body_even_when_prefix_is_valid_json() -> None:
    handler = body_handler(b"{}", [("Content-Length", "3")])
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 400


@pytest.mark.parametrize("chunked", [False, True])
def test_exact_body_limit_and_partial_reader(chunked: bool) -> None:
    raw = b"x" * 64
    if chunked:
        wire = b'0020;flag;label="a; b\\\"c"\r\n' + raw[:32]
        wire += b"\r\n20 ; next = yes\r\n" + raw[32:] + b"\r\n0\r\nX-Note: ignored\r\n\r\n"
        headers = CHUNKED
    else:
        wire, headers = raw, [("Content-Length", "00064")]
    handler = body_handler(wire, headers, fragment=3)
    assert handler._read_body() == raw
    assert handler.rfile.tell() == len(wire)


@pytest.mark.parametrize("wire,headers,expected_read", [
    (b"x" * 100_000, [("Content-Length", "100000")], 0),
    (b"x" * 100_000, [("Content-Length", "9" * 5000)], 0),
    (b"41\r\n" + b"x" * 100_000, CHUNKED, 4),
    (b"f" * 1000 + b"\r\n" + b"x" * 100_000, CHUNKED, 1002),
    (b"20\r\n" + b"x" * 32 + b"\r\n21\r\n" + b"x" * 100_000, CHUNKED, 42),
])
def test_large_body_rejected_before_over_limit_read(
    wire: bytes, headers: list[tuple[str, str]], expected_read: int,
) -> None:
    handler = body_handler(wire, headers)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 413
    assert handler.rfile.tell() == expected_read
    assert all(0 <= size <= 64 for size in handler.rfile.read_sizes)






def test_gzip_accepts_exact_decompressed_limit() -> None:
    raw = b"x" * 64
    assert router_http._gunzip_limited(gzip.compress(raw), 64) == raw
    assert router_http._gunzip_limited(gzip.compress(b""), 64) == b""


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_gzip_expansion_cannot_bypass_small_body_limit(chunked: bool, truncated: bool) -> None:
    zipped = gzip.compress(b"x" * 100_000)
    if truncated:
        zipped = zipped[:-8]
    assert len(zipped) < 256
    if chunked:
        wire = f"{len(zipped):x}\r\n".encode() + zipped + b"\r\n0\r\n\r\n"
        headers = CHUNKED
    else:
        wire, headers = zipped, [("Content-Length", str(len(zipped)))]
    handler = body_handler(wire, headers + [("Content-Encoding", "gzip")], limit=256)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 413


