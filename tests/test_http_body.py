"""Faithful bounded readers exercise framing without socket timing dependencies."""

from __future__ import annotations

import gzip
from email.message import Message
from io import BytesIO
from types import SimpleNamespace

import pytest

import model_router as mr


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


@pytest.mark.parametrize("size", [
    b"-1", b"-0", b"+1", b"+0", b"0x1", b"1_0", b"", b"xyz",
    b" 1", b"1 ", b"1;", b"1;foo=", b'1;foo="unclosed',
    b"1;foo=bad value", b"1;foo=\x00", b"1\nignored",
])
def test_invalid_chunk_size_is_rejected_before_payload_read(size: bytes) -> None:
    wire = size + b"\r\n" + b"x" * 100_000
    handler = body_handler(wire, CHUNKED)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 400
    assert handler.rfile.tell() <= len(size) + 2
    assert handler.rfile.read_sizes == []


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


@pytest.mark.parametrize("headers,status", [
    (CHUNKED + [("Content-Length", "0")], 400),
    (CHUNKED * 2, 400),
    ([("Content-Length", "2"), ("Content-Length", "2")], 400),
    ([("Content-Length", "2"), ("Content-Length", "3")], 400),
    ([("Transfer-Encoding", "gzip, chunked")], 400),
    ([("Transfer-Encoding", "chunked, gzip")], 400),
    ([("Transfer-Encoding", "chunked, chunked")], 400),
    ([("Transfer-Encoding", "notchunked")], 400),
    ([("Transfer-Encoding", "identity")], 400),
    ([("Transfer-Encoding", "chunked;foo=bar")], 400),
    ([("Transfer-Encoding", "\vchunked")], 400),
    ([("Transfer-Encoding", "")], 400),
    ([("Content-Encoding", "gzip, gzip")], 415),
    ([("Content-Encoding", "gzip"), ("Content-Encoding", "gzip")], 415),
    ([("Content-Encoding", "br")], 415),
    ([("Content-Encoding", "gzip\v")], 415),
])
def test_unsupported_or_conflicting_framing_rejected_before_read(
    headers: list[tuple[str, str]], status: int,
) -> None:
    handler = body_handler(b"x" * 100_000, headers)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == status
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


@pytest.mark.parametrize("wire,limit", [
    (b"1;name=" + b"x" * 100_000, 64),
    (b"0\r\nX-Name: " + b"x" * 100_000, 64),
    ((b"1;name=" + b"x" * 900 + b"\r\nx\r\n") * 100, 128),
    (b"0\r\n" + (b"X-Name: " + b"x" * 900 + b"\r\n") * 100, 64),
    (b"0\r\n" + b"X-Name: x\r\n" * 101 + b"\r\n", 64),
    (b"1\r\nx\r\n" * 20_000, 32_000),
])
def test_chunk_metadata_is_bounded(wire: bytes, limit: int) -> None:
    handler = body_handler(wire, CHUNKED, limit)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 413
    # Payload bytes and framing are bounded independently.
    assert handler.rfile.tell() <= mr.MAX_CHUNK_METADATA + limit + 1


@pytest.mark.parametrize("wire", [
    b"", b"not gzip", gzip.compress(b"{}")[:5], gzip.compress(b"{}")[:-8],
    gzip.compress(b"{}")[:-1],
    gzip.compress(b"{}")[:-8] + b"\0" * 8,  # Invalid CRC/length.
    gzip.compress(b"{}") + gzip.compress(b" "),
    gzip.compress(b"{}") + gzip.compress(b""),
    gzip.compress(b"{}") + b"junk", gzip.compress(b"{}") + b"\0",
])
def test_gzip_requires_one_complete_valid_member(wire: bytes) -> None:
    handler = body_handler(wire, [("Content-Length", str(len(wire))),
                                  ("Content-Encoding", "gzip")], limit=128)
    with pytest.raises(mr.RouterError) as error:
        handler._read_body()
    assert error.value.status == 400


def test_gzip_accepts_exact_decompressed_limit() -> None:
    raw = b"x" * 64
    assert mr._gunzip_limited(gzip.compress(raw), 64) == raw
    assert mr._gunzip_limited(gzip.compress(b""), 64) == b""


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


def test_static_provider_headers_cannot_mislabel_reserialized_json() -> None:
    provider = mr.Provider(
        name="test", dialect=mr.DIALECT_OPENAI_CHAT, base_url="http://localhost",
        api_key="dummy-key", timeout=30, litellm_provider="openai",
        litellm_kwargs={"extra_headers": {"Repr-Digest": "stale"}}, drop_params=True,
        headers={"Content-Encoding": "gzip", "Content-Type": "text/plain",
                 "Content-Length": "1", "Transfer-Encoding": "chunked",
                 "Content-Digest": "stale"},
    )
    translated = provider.litellm_call_kwargs()
    for headers in (mr._upstream_request_headers(provider, mr.DIALECT_OPENAI_CHAT, {}),
                    translated["headers"], translated["extra_headers"]):
        lowered = {name.lower(): value for name, value in headers.items()}
        assert lowered["content-type"] == "application/json"
        assert not (lowered.keys() & {"content-encoding", "content-length", "content-digest",
                                      "repr-digest", "transfer-encoding"})
