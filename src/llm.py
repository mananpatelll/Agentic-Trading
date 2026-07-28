"""Central LLM factory.

Agents ask for a model by ROLE; this module decides which model that is and how
it is built. Nothing else in the codebase constructs a chat model directly.

Built for concurrent use: instances are shared, and every agent draws from one
process-wide rate limit, so fanning nodes out in parallel needs no changes here.
"""

import threading
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, InternalServerError, RateLimitError

from load_config import load_config

# Failures worth waiting out and retrying, for callers that retry a whole
# candidate. Deliberately NOT openai.APIError: that also covers permanent
# failures (BadRequestError, AuthenticationError) where retrying only burns
# the backoff and fails anyway. APITimeoutError subclasses APIConnectionError.
TRANSIENT_API_ERRORS = (RateLimitError, InternalServerError, APIConnectionError)

# Once, before any client is constructed — not per agent module.
load_dotenv()

_CFG: dict = load_config().get("models", {})
_DEFAULTS: dict = _CFG.get("defaults", {})
_ROLES: dict = _CFG.get("roles", {})

_cache: dict[tuple, ChatOpenAI] = {}
_lock = threading.Lock()


def _build_rate_limiter() -> Optional[InMemoryRateLimiter]:
    rl = _CFG.get("rate_limit")
    if not rl:
        return None
    rps = rl["requests_per_second"]
    return InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=rl.get("check_every_n_seconds", 0.1),
        max_bucket_size=rl.get("max_bucket_size", rps),
    )


# One limiter for the whole process. Because every model shares this object,
# the cap is on TOTAL requests, not per-agent — which is what a provider rate
# limit actually measures once nodes run concurrently.
_RATE_LIMITER = _build_rate_limiter()


def get_model(role: str, **overrides: Any) -> ChatOpenAI:
    """Return the chat model for `role`, resolved from config.

    Settings layer: defaults <- role <- overrides.

    Instances are cached and shared. A ChatOpenAI holds no per-call state, so
    reusing one across threads is safe, and the same object serves both
    `.invoke` and `.ainvoke`. Sharing also means one HTTP connection pool and
    one rate-limit budget across parallel nodes instead of one per agent.
    """
    settings = {**_DEFAULTS, **_ROLES.get(role, {}), **overrides}
    key = (role, tuple(sorted(settings.items())))

    cached = _cache.get(key)
    if cached is not None:
        return cached

    with _lock:
        # Re-check: another thread may have built it while we waited.
        if key not in _cache:
            _cache[key] = ChatOpenAI(rate_limiter=_RATE_LIMITER, **settings)
        return _cache[key]
