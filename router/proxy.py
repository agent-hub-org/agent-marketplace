"""Proxy route handlers — collapses ~25 near-identical agent-forwarding endpoints."""
import re
from collections.abc import Callable

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from router.registry import AgentRegistry

_SAFE_SESSION_RE = re.compile(r'^[a-zA-Z0-9\-]{1,64}$')


class _SessionsBody(BaseModel):
    session_ids: list[str]


def create_proxy_router(registry: AgentRegistry, get_client: Callable[[], httpx.AsyncClient]) -> APIRouter:
    """Build and return an APIRouter with all agent proxy routes."""

    router = APIRouter()

    def _agent_url(agent_id: str) -> str:
        url = registry.get_url(agent_id)
        if not url:
            raise HTTPException(404, f"Agent '{agent_id}' not found.")
        return url

    async def _json(method: str, agent_id: str, path: str, *,
                    body=None, headers: dict | None = None, params: dict | None = None,
                    timeout: float = 30.0):
        resp = await get_client().request(
            method, f"{_agent_url(agent_id)}{path}",
            json=body, headers=headers or {}, params=params, timeout=timeout,
        )
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()

    async def _file(agent_id: str, path: str, *, timeout: float = 60.0) -> Response:
        resp = await get_client().get(f"{_agent_url(agent_id)}{path}", timeout=timeout)
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        return Response(
            content=resp.content,
            media_type=resp.headers.get("content-type", "application/octet-stream"),
            headers={"Content-Disposition": resp.headers.get("content-disposition", "")},
        )

    def _uid(request: Request) -> str | None:
        return getattr(request.state, "user_id", None)

    def _uid_header(request: Request) -> dict:
        uid = _uid(request)
        return {"X-User-Id": uid} if uid else {}

    def _require_uid(request: Request) -> str:
        uid = _uid(request)
        if not uid:
            raise HTTPException(401, "Authentication required")
        return uid

    # ── Upload / Download ──

    @router.post("/agents/{agent_id}/upload/{upload_type}")
    async def proxy_upload(agent_id: str, upload_type: str,
                           file: UploadFile = File(...), session_id: str = Form(...)):
        agent_url = _agent_url(agent_id)
        file_bytes = await file.read()
        resp = await get_client().post(
            f"{agent_url}/upload/{upload_type}",
            files={"file": (file.filename, file_bytes, file.content_type)},
            data={"session_id": session_id}, timeout=120.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()

    @router.get("/agents/{agent_id}/download/{file_id}")
    async def proxy_download(agent_id: str, file_id: str):
        return await _file(agent_id, f"/download/{file_id}")

    @router.get("/agents/{agent_id}/files/{session_id}")
    async def proxy_list_files(agent_id: str, session_id: str):
        return await _json("GET", agent_id, f"/files/{session_id}", timeout=30.0)

    # ── Charts / Quotes ──

    @router.get("/agents/{agent_id}/charts/{ticker}")
    async def proxy_charts(agent_id: str, ticker: str, request: Request):
        return await _json("GET", agent_id, f"/charts/{ticker}", params=dict(request.query_params))

    @router.get("/agents/{agent_id}/quotes")
    async def proxy_quotes(agent_id: str, request: Request):
        return await _json("GET", agent_id, "/quotes", params=dict(request.query_params), timeout=15.0)

    # ── Watchlists ──

    @router.post("/agents/{agent_id}/watchlists")
    async def proxy_create_watchlist(agent_id: str, request: Request):
        return await _json("POST", agent_id, "/watchlists",
                           body=await request.json(), headers=_uid_header(request))

    @router.get("/agents/{agent_id}/watchlists")
    async def proxy_list_watchlists(agent_id: str, request: Request):
        return await _json("GET", agent_id, "/watchlists", headers=_uid_header(request))

    @router.get("/agents/{agent_id}/watchlists/{watchlist_id}/performance")
    async def proxy_watchlist_performance(agent_id: str, watchlist_id: str, request: Request):
        return await _json("GET", agent_id, f"/watchlists/{watchlist_id}/performance",
                           headers=_uid_header(request))

    @router.get("/agents/{agent_id}/watchlists/{watchlist_id}")
    async def proxy_get_watchlist(agent_id: str, watchlist_id: str, request: Request):
        return await _json("GET", agent_id, f"/watchlists/{watchlist_id}",
                           headers=_uid_header(request))

    @router.put("/agents/{agent_id}/watchlists/{watchlist_id}")
    async def proxy_update_watchlist(agent_id: str, watchlist_id: str, request: Request):
        return await _json("PUT", agent_id, f"/watchlists/{watchlist_id}",
                           body=await request.json(), headers=_uid_header(request))

    @router.delete("/agents/{agent_id}/watchlists/{watchlist_id}")
    async def proxy_delete_watchlist(agent_id: str, watchlist_id: str, request: Request):
        await _json("DELETE", agent_id, f"/watchlists/{watchlist_id}",
                    headers=_uid_header(request))
        return {"success": True}

    # ── Holdings ──

    @router.post("/agents/{agent_id}/holdings")
    async def proxy_create_holding(agent_id: str, request: Request):
        return await _json("POST", agent_id, "/holdings",
                           body=await request.json(), headers=_uid_header(request))

    @router.get("/agents/{agent_id}/holdings")
    async def proxy_list_holdings(agent_id: str, request: Request):
        return await _json("GET", agent_id, "/holdings", headers=_uid_header(request))

    @router.get("/agents/{agent_id}/holdings/performance")
    async def proxy_portfolio_performance(agent_id: str, request: Request):
        return await _json("GET", agent_id, "/holdings/performance", headers=_uid_header(request))

    @router.put("/agents/{agent_id}/holdings/{holding_id}")
    async def proxy_update_holding(agent_id: str, holding_id: str, request: Request):
        return await _json("PUT", agent_id, f"/holdings/{holding_id}",
                           body=await request.json(), headers=_uid_header(request))

    @router.delete("/agents/{agent_id}/holdings/{holding_id}")
    async def proxy_delete_holding(agent_id: str, holding_id: str, request: Request):
        await _json("DELETE", agent_id, f"/holdings/{holding_id}", headers=_uid_header(request))
        return {"success": True}

    # ── Investor Profile ──

    @router.get("/agents/{agent_id}/profile/onboard/start")
    async def proxy_onboard_start(agent_id: str):
        return await _json("GET", agent_id, "/profile/onboard/start", timeout=10.0)

    @router.get("/agents/{agent_id}/profile")
    async def proxy_get_profile(agent_id: str, request: Request):
        return await _json("GET", agent_id, "/profile",
                           headers=_uid_header(request), timeout=10.0)

    @router.put("/agents/{agent_id}/profile")
    async def proxy_upsert_profile(agent_id: str, request: Request):
        return await _json("PUT", agent_id, "/profile",
                           body=await request.json(), headers=_uid_header(request), timeout=10.0)

    # ── History ──

    @router.get("/agents/{agent_id}/history/me")
    async def proxy_history(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("GET", agent_id, "/history/user/me", headers={"X-User-Id": uid})

    @router.post("/agents/{agent_id}/history/sessions")
    async def proxy_history_sessions(agent_id: str, body: _SessionsBody, request: Request):
        safe_ids = [s for s in body.session_ids[:20] if isinstance(s, str) and _SAFE_SESSION_RE.match(s)]
        return await _json("POST", agent_id, "/history/sessions", body={"session_ids": safe_ids})

    # ── Health agent: Progress & Nutrition ──

    @router.get("/agents/{agent_id}/progress")
    async def proxy_get_progress(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("GET", agent_id, "/progress",
                           params=dict(request.query_params), headers={"X-User-Id": uid})

    @router.post("/agents/{agent_id}/progress")
    async def proxy_log_progress(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("POST", agent_id, "/progress",
                           body=await request.json(), headers={"X-User-Id": uid})

    @router.get("/agents/{agent_id}/nutrition")
    async def proxy_get_nutrition(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("GET", agent_id, "/nutrition",
                           params=dict(request.query_params), headers={"X-User-Id": uid})

    @router.post("/agents/{agent_id}/nutrition")
    async def proxy_log_nutrition(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("POST", agent_id, "/nutrition",
                           body=await request.json(), headers={"X-User-Id": uid})

    # ── Interview prep: Scores & Notes ──

    @router.get("/agents/{agent_id}/scores/user/me")
    async def proxy_get_user_scores(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("GET", agent_id, "/scores/user/me", headers={"X-User-Id": uid})

    @router.post("/agents/{agent_id}/scores")
    async def proxy_create_score(agent_id: str, request: Request):
        return await _json("POST", agent_id, "/scores",
                           body=await request.json(), headers=_uid_header(request))

    @router.post("/agents/{agent_id}/notes/share")
    async def proxy_share_note(agent_id: str, request: Request):
        return await _json("POST", agent_id, "/notes/share", body=await request.json())

    @router.get("/agents/{agent_id}/notes/shared/{token}")
    async def proxy_shared_note(agent_id: str, token: str):
        return await _file(agent_id, f"/notes/shared/{token}")

    # ── News agent: Preferences ──

    @router.get("/agents/{agent_id}/preferences")
    async def proxy_get_preferences(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("GET", agent_id, "/preferences", headers={"X-User-Id": uid})

    @router.post("/agents/{agent_id}/preferences")
    async def proxy_save_preferences(agent_id: str, request: Request):
        uid = _require_uid(request)
        return await _json("POST", agent_id, "/preferences",
                           body=await request.json(), headers={"X-User-Id": uid})

    return router
