"""Direct fx/Cline compatibility against the real router and fake upstream."""
from uuid import uuid4

import httpx
import pytest

from model_router.client_compat import prepare_chat
from test_model_router import _config, router_config, upstream


def test_marked_chat_affinity_normalization_and_streaming(upstream):
    raw = _config(upstream)
    raw['server']['auth_mode'] = 'none'
    raw['server']['bearer_session_compat'] = True
    first, second = ['router-session-' + str(uuid4()) for _ in range(2)]
    messages = [{'role': 'system', 'content': 'first'},
                {'role': 'system', 'content': 'second'},
                {'role': 'user', 'content': 'hello'},
                {'role': 'system', 'content': 'later'}]
    with router_config(raw) as base:
        for session, stream in [(first, False), (first, True), (second, False)]:
            response = httpx.post(base + '/v1/chat/completions', json={
                'model': 'm-chat', 'messages': messages, 'stream': stream,
            }, headers={'Authorization': 'Bearer ' + session}, timeout=30)
            assert response.status_code == 200, response.text
            if stream:
                assert 'data:' in response.text and '[DONE]' in response.text
            request = upstream.last_request('/v1/chat/completions')
            assert request['headers']['x-opencode-session'] == session
            assert request['headers']['authorization'] != 'Bearer ' + session
            assert request['body']['messages'] == [
                {'role': 'system', 'content': 'first\n\nsecond'}, *messages[2:]]
        assert httpx.get(base + '/v1/models').status_code == 200


def test_unmarked_payload_is_unchanged_and_explicit_header_wins():
    payload = {'messages': [{'role': 'system', 'content': 'a'}, {'role': 'system', 'content': 'b'}]}
    headers = {'Authorization': 'Bearer secret'}
    assert prepare_chat(payload, headers) == (payload, headers)
    session = 'router-session-' + str(uuid4())
    _, result = prepare_chat(payload, {'Authorization': 'Bearer ' + session, 'X-OpenCode-Session': 'native'})
    assert result == {'X-OpenCode-Session': 'native'}
    named = {'messages': [{'role': 'system', 'content': 'a', 'name': 'named'}, *payload['messages']]}
    assert prepare_chat(named, {'Authorization': 'Bearer ' + session})[0] == named


def test_compatibility_rejects_authenticated_config(upstream):
    raw = _config(upstream, api_keys=['secret'])
    raw['server']['bearer_session_compat'] = True
    with pytest.raises(SystemExit, match='requires unauthenticated'):
        with router_config(raw):
            pass


def test_compatibility_is_disabled_by_default(upstream):
    raw = _config(upstream)
    raw['server']['auth_mode'] = 'none'
    session = 'router-session-' + str(uuid4())
    messages = [{'role': 'system', 'content': 'a'}, {'role': 'system', 'content': 'b'}]
    with router_config(raw) as base:
        response = httpx.post(base + '/v1/chat/completions', json={
            'model': 'm-chat', 'messages': messages,
        }, headers={'Authorization': 'Bearer ' + session}, timeout=30)
        assert response.status_code == 200, response.text
        request = upstream.last_request('/v1/chat/completions')
        assert request['body']['messages'] == messages
        assert 'x-opencode-session' not in request['headers']
