from agent_common.secrets.akv import load_akv_secrets
load_akv_secrets()

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from agent_common.logging import configure_logging
from agent_common.auth import KeycloakJWTMiddleware
from agent_common.server.app_factory import create_agent_app
from config import AGENT_URLS
from bff_router import router as bff_router
from router.registry import AgentRegistry
from router.router_agent import EmbeddingRouter
from router.a2a_caller import AgentCaller
from router.proxy import create_proxy_router

load_dotenv()
configure_logging("marketplace")
logger = logging.getLogger("marketplace.api")

_INTERNAL_HEADERS = {"X-Internal-API-Key": os.getenv("INTERNAL_API_KEY")} if os.getenv("INTERNAL_API_KEY") else {}

registry = AgentRegistry(AGENT_URLS, internal_headers=_INTERNAL_HEADERS)
embedding_router = EmbeddingRouter()
caller = AgentCaller()

_proxy_client: httpx.AsyncClient | None = None
_REGISTRY_REFRESH_INTERVAL = int(os.getenv("REGISTRY_REFRESH_INTERVAL", "60"))


def _get_rate_limit_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agent_common.observability import init_sentry
    init_sentry("marketplace")
    global _proxy_client
    _proxy_client = httpx.AsyncClient(headers=_INTERNAL_HEADERS, timeout=120.0)
    await registry.refresh()
    await embedding_router.build_index(registry.get_cards())
    await embedding_router.init()
    logger.info("Marketplace started — %d agent(s) registered", len(registry.get_cards()))

    async def _refresh_loop():
        while True:
            await asyncio.sleep(_REGISTRY_REFRESH_INTERVAL)
            try:
                changed = await registry.refresh()
                if changed:
                    await embedding_router.build_index(registry.get_cards())
                    await embedding_router.clear_routing_cache()
                    logger.info("Periodic refresh: registry changed — rebuilt routing index (%d agents)", len(registry.get_cards()))
            except Exception:
                logger.exception("Periodic registry refresh failed — will retry in %ds", _REGISTRY_REFRESH_INTERVAL)

    refresh_task = asyncio.create_task(_refresh_loop())
    yield
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    await embedding_router.close()
    await caller.close()
    await _proxy_client.aclose()
    logger.info("Marketplace shutdown")


app, limiter = create_agent_app(
    title="Agent Marketplace",
    lifespan=lifespan,
    key_func=_get_rate_limit_key,
    require_internal_key=False,
)

_REDIS_URL = os.getenv("REDIS_URL", "")
if _REDIS_URL:
    # Re-create limiter with Redis storage; reuse the same key_func.
    limiter = Limiter(key_func=_get_rate_limit_key, storage_uri=_REDIS_URL)
    app.state.limiter = limiter

if os.getenv("HTTPS_REDIRECT", "").lower() == "true":
    app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(KeycloakJWTMiddleware)

app.include_router(bff_router)
app.include_router(create_proxy_router(registry, lambda: _proxy_client))

# Import routers after limiter, registry, embedding_router, caller are defined
from routers.routing import router as routing_router
from routers.agents import router as agents_router
from routers.admin import router as admin_router

app.include_router(routing_router)
app.include_router(agents_router)
app.include_router(admin_router)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
