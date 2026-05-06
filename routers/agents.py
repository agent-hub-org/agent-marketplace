import asyncio
import logging

import httpx
from fastapi import APIRouter

from agent_sdk.llm_services.model_registry import list_models as _sdk_list_models
from config import AGENT_URLS

from app import registry, embedding_router, _INTERNAL_HEADERS

router = APIRouter(tags=["agents"])
logger = logging.getLogger("marketplace.api")


@router.get("/agents/status")
async def agents_status():
    """Fan out /health checks to all registered agents and return their status."""
    async def _check(agent_id: str, base_url: str):
        t0 = asyncio.get_running_loop().time()
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base_url}/health", headers=_INTERNAL_HEADERS)
            latency_ms = round((asyncio.get_running_loop().time() - t0) * 1000)
            return agent_id, {"status": "ok" if r.is_success else "error", "latencyMs": latency_ms}
        except Exception:
            return agent_id, {"status": "error", "latencyMs": None}

    results = await asyncio.gather(*[_check(aid, url) for aid, url in AGENT_URLS.items()])
    return dict(results)


@router.get("/agents")
async def list_agents():
    """List all registered agents with their Agent Cards."""
    return {"agents": registry.get_cards()}


@router.post("/agents/refresh")
async def refresh_agents():
    """Re-fetch Agent Cards and rebuild the embedding index."""
    changed = await registry.refresh()
    cards = registry.get_cards()
    if changed:
        await embedding_router.build_index(cards)
        await embedding_router.clear_routing_cache()
        logger.info("Agent registry changed — routing and embedding caches cleared")
    return {
        "status": "refreshed" if changed else "no_change",
        "agents_available": len(cards),
        "agent_ids": list(cards.keys()),
    }


@router.get("/models")
async def get_models():
    """List available LLM models that can be selected from the frontend."""
    return {"models": _sdk_list_models()}
