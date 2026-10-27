"""Model router package with lazy, compatible public embedding imports.

Use the owning modules for implementation details. Importing configuration does
not initialize LiteLLM; CLI/server startup still checks the pinned adapter.
"""
from importlib import import_module

_EXPORTS = {
    'App': 'app',
    'CatalogConfig': 'config',
    'CatalogRule': 'config',
    'CatalogSnapshot': 'discovery',
    'Config': 'config',
    'DEFAULT_ANTHROPIC_VERSION': 'constants',
    'DEFAULT_BODY_READ_TIMEOUT': 'constants',
    'DEFAULT_HEADER_READ_TIMEOUT': 'constants',
    'DEFAULT_LLAMA_CONTEXT': 'constants',
    'DEFAULT_LLAMA_PROBE_TIMEOUT': 'constants',
    'DEFAULT_LLAMA_REFRESH': 'constants',
    'DEFAULT_LLAMA_URL': 'constants',
    'DEFAULT_MAX_BODY': 'constants',
    'DEFAULT_MAX_CONNECTIONS': 'constants',
    'DEFAULT_MAX_GENERATIONS': 'constants',
    'DEFAULT_TIMEOUT': 'constants',
    'DEFAULT_WRITE_TIMEOUT': 'constants',
    'DIALECTS': 'constants',
    'DIALECT_ANTHROPIC': 'constants',
    'DIALECT_GEMINI': 'constants',
    'DIALECT_OPENAI_CHAT': 'constants',
    'DIALECT_OPENAI_RESPONSES': 'constants',
    'DISCOVERY_CACHE_MAX_BYTES': 'constants',
    'DISCOVERY_CACHE_VERSION': 'constants',
    'DiscoveryCacheConfig': 'config',
    'ENDPOINTS': 'constants',
    'ErrorSanitizer': 'common',
    'Handler': 'http',
    'LLAMA_LIMIT_FIELDS': 'constants',
    'LOG': 'constants',
    'LlamaCppConfig': 'config',
    'LlamaCppServer': 'config',
    'LlamaSnapshot': 'discovery',
    'MAX_CHUNK_LINE': 'constants',
    'MAX_CHUNK_METADATA': 'constants',
    'MAX_CHUNK_TRAILERS': 'constants',
    'MAX_ERROR_MESSAGE': 'constants',
    'Plan': 'streams',
    'Provider': 'config',
    'REFRESH_BACKOFF_BASE': 'constants',
    'REFRESH_BACKOFF_MAX': 'constants',
    'RefreshState': 'discovery',
    'RegistrySnapshot': 'discovery',
    'RetryPolicy': 'config',
    'Route': 'config',
    'RouterError': 'common',
    'RouterHTTPServer': 'http',
    'StreamState': 'streams',
    'TESTED_LITELLM_VERSION': 'constants',
    'UNAUTHENTICATED_API_KEY': 'constants',
    '__version__': 'constants',
    'build_config': 'config',
    'default_env_files': 'config',
    'load_config': 'config',
    'load_default_credentials': 'config',
    'load_env_file': 'config',
    'main': 'cli',
    'serve': 'http',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
