import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from router.router_agent import LowConfidenceError
from router.streaming import build_marketplace_sse_stream
from models.requests import (
    QueryRequest, QueryResponse,
    DirectQueryRequest, DirectQueryResponse,
)

# app.py creates limiter, registry, router, caller before importing routers
from app import limiter, registry, embedding_router, caller

router = APIRouter(tags=["routing"])
logger = logging.getLogger("marketplace.api")


def _resolve_mode(agent_card: dict | None, agent_name: str, body_mode: str | None = None) -> str | None:
    """Resolve execution mode from body, agent card metadata, or known agent defaults."""
    if body_mode:
        return body_mode
    if agent_card:
        mode = agent_card.get("metadata", {}).get("mode")
        if mode:
            return mode
    if agent_name == "financial-agent":
        return "financial_analyst"
    if agent_name == "research-agent":
        return "researcher"
    return None


@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, body: QueryRequest):
    """Route a query to the best agent via embedding-based routing."""
    logger.info("POST /query — query='%s'", body.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    try:
        decision = await embedding_router.route_with_cache(body.query)
    except LowConfidenceError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Your query doesn't clearly match any available agent (best match: '{e.best_agent}', "
                f"confidence: {e.best_score:.2f}). Try rephrasing or be more specific."
            ),
        )

    from config import AGENT_URLS
    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    user_id = request.state.user_id
    agent_card = registry.get_card(decision.agent_name)
    mode = _resolve_mode(agent_card, decision.agent_name)

    response_text = await caller.call_agent(
        agent_url, body.query, body.session_id, user_id=user_id, mode=mode,
        request_id=request.state.request_id,
        watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
    )

    from agent_sdk.config import settings as sdk_settings
    is_low_confidence = decision.confidence < sdk_settings.min_routing_confidence
    return QueryResponse(
        query=body.query,
        routed_to=decision.agent_name,
        reasoning=decision.reasoning,
        response=response_text,
        routing_confidence=decision.confidence,
        low_confidence=is_low_confidence,
    )


@router.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(body: QueryRequest, request: Request):
    """Route a query to the best agent and stream the response as SSE."""
    logger.info("POST /query/stream — query='%s'", body.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    try:
        decision = await embedding_router.route_with_cache(body.query)
    except LowConfidenceError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Your query doesn't clearly match any available agent (best match: '{e.best_agent}', "
                f"confidence: {e.best_score:.2f}). Try rephrasing or be more specific."
            ),
        )

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(registry.get_cards().keys())}",
        )

    user_id = request.state.user_id
    agent_card = registry.get_card(decision.agent_name)
    supports_streaming = agent_card and agent_card.get("capabilities", {}).get("streaming", False)
    mode = _resolve_mode(agent_card, decision.agent_name)
    _request_id = request.state.request_id

    from agent_sdk.config import settings as sdk_settings
    _is_low_confidence = decision.confidence < sdk_settings.min_routing_confidence
    preamble = f"data: {json.dumps({'routed_to': decision.agent_name, 'reasoning': decision.reasoning, 'routing_confidence': decision.confidence, 'low_confidence': _is_low_confidence})}\n\n"

    async def _source():
        if supports_streaming:
            async for chunk in caller.stream_agent(
                agent_url, body.query, body.session_id,
                response_format=body.response_format, model_id=body.model_id,
                mode=mode, user_id=user_id, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            ):
                yield chunk
        else:
            response_text = await caller.call_agent(
                agent_url, body.query, body.session_id, user_id=user_id,
                mode=mode, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            )
            yield response_text

    return StreamingResponse(
        build_marketplace_sse_stream(_source(), preamble=preamble),
        media_type="text/event-stream",
    )


@router.post("/agents/{agent_id}/query", response_model=DirectQueryResponse)
@limiter.limit("30/minute")
async def direct_query(agent_id: str, request: Request, body: DirectQueryRequest):
    """Call a specific agent directly via A2A, bypassing the router."""
    logger.info("POST /agents/%s/query — query='%s'", agent_id, body.query[:100])

    from config import AGENT_URLS
    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    user_id = request.state.user_id
    agent_card = registry.get_card(agent_id)
    mode = _resolve_mode(agent_card, agent_id, body.mode)

    response_text = await caller.call_agent(
        agent_url, body.query, body.session_id, user_id=user_id, mode=mode,
        request_id=request.state.request_id,
        watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
    )

    return DirectQueryResponse(agent_id=agent_id, query=body.query, response=response_text)


@router.post("/agents/{agent_id}/query/stream")
@limiter.limit("30/minute")
async def direct_query_stream(agent_id: str, body: DirectQueryRequest, request: Request):
    """Stream a response from a specific agent, bypassing the router."""
    logger.info("POST /agents/%s/query/stream — query='%s'", agent_id, body.query[:100])

    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(registry.get_cards().keys())}",
        )

    user_id = request.state.user_id
    agent_card = registry.get_card(agent_id)
    supports_streaming = agent_card and agent_card.get("capabilities", {}).get("streaming", False)
    mode = _resolve_mode(agent_card, agent_id, body.mode)
    _request_id = request.state.request_id

    async def _source():
        if supports_streaming:
            async for chunk in caller.stream_agent(
                agent_url, body.query, body.session_id,
                response_format=body.response_format, model_id=body.model_id,
                mode=mode, user_id=user_id, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            ):
                yield chunk
        else:
            response_text = await caller.call_agent(
                agent_url, body.query, body.session_id, user_id=user_id,
                mode=mode, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            )
            yield response_text

    return StreamingResponse(
        build_marketplace_sse_stream(_source()),
        media_type="text/event-stream",
    )
