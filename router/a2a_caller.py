# NOTE (AR-1 resolved): Streaming now uses A2A message/stream instead of
# direct /ask/stream SSE. The marketplace checks agent capabilities and falls
# back to message/send (non-streaming) if the agent doesn't support streaming.

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator

import httpx

from agent_sdk.config import settings

logger = logging.getLogger("marketplace.a2a_caller")


class AgentCaller:
    """Sends tasks to agents via the A2A protocol and supports SSE streaming."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        self._internal_key = os.getenv("INTERNAL_API_KEY")

    async def close(self):
        await self._client.aclose()

    def _build_a2a_payload(
        self,
        method: str,
        query: str,
        session_id: str,
        *,
        user_id: str | None = None,
        mode: str | None = None,
        model_id: str | None = None,
        response_format: str | None = None,
        watchlist_id: str | None = None,
        as_of_date: str | None = None,
    ) -> dict:
        task_id = uuid.uuid4().hex
        return {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": method,
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "messageId": uuid.uuid4().hex,
                    "role": "user",
                    "parts": [{"type": "text", "text": query}],
                },
                "metadata": {
                    "user_id": user_id,
                    "mode": mode,
                    "model_id": model_id,
                    "response_format": response_format,
                    "watchlist_id": watchlist_id,
                    "as_of_date": as_of_date,
                },
                "acceptedOutputModes": ["text"],
            },
        }

    def _build_headers(self, request_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if self._internal_key:
            headers["X-Internal-API-Key"] = self._internal_key
        return headers

    async def call_agent(self, agent_url: str, query: str, session_id: str | None = None,
                         user_id: str | None = None, mode: str | None = None,
                         request_id: str | None = None, watchlist_id: str | None = None,
                         as_of_date: str | None = None) -> str:
        """Send a message/send request to an A2A agent and return the response text."""
        session_id = session_id or uuid.uuid4().hex
        payload = self._build_a2a_payload(
            "message/send", query, session_id,
            user_id=user_id, mode=mode,
            watchlist_id=watchlist_id, as_of_date=as_of_date,
        )
        a2a_endpoint = f"{agent_url}/a2a/"
        headers = self._build_headers(request_id)
        logger.info("Calling A2A agent at %s — task_id='%s'", a2a_endpoint, payload["id"])

        _MAX_RETRIES = settings.a2a_max_retries
        data = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(a2a_endpoint, json=payload, headers=headers)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    backoff = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                    logger.warning(
                        "A2A call rate limited (attempt %d/%d) — retrying in %ds",
                        attempt + 1, _MAX_RETRIES, backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                backoff = 2 ** attempt
                logger.warning("A2A call failed (attempt %d/%d): %s — retrying in %ds",
                                attempt + 1, _MAX_RETRIES, e, backoff)
                await asyncio.sleep(backoff)
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                if attempt == _MAX_RETRIES - 1:
                    raise
                backoff = 2 ** attempt
                logger.warning("A2A call HTTP %d (attempt %d/%d) — retrying in %ds",
                                e.response.status_code, attempt + 1, _MAX_RETRIES, backoff)
                await asyncio.sleep(backoff)

        result = data.get("result", {})
        artifacts = result.get("artifacts", [])

        texts = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part["text"])

        if not texts:
            history = result.get("history", [])
            for msg in reversed(history):
                if msg.get("role") == "agent":
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            texts.append(part["text"])
                    break

        response_text = "\n".join(texts) if texts else "No response from agent."
        logger.info("A2A call complete — response length: %d chars", len(response_text))
        return response_text

    async def stream_agent(self, agent_url: str, query: str, session_id: str | None = None,
                           response_format: str | None = None, model_id: str | None = None,
                           mode: str | None = None, user_id: str | None = None,
                           request_id: str | None = None, watchlist_id: str | None = None,
                           as_of_date: str | None = None) -> AsyncIterator[str]:
        """Call an agent via the A2A protocol's message/stream method."""
        session_id = session_id or uuid.uuid4().hex
        payload = self._build_a2a_payload(
            "message/stream", query, session_id,
            user_id=user_id, mode=mode, model_id=model_id, response_format=response_format,
            watchlist_id=watchlist_id, as_of_date=as_of_date,
        )
        a2a_endpoint = f"{agent_url}/a2a/"
        headers = self._build_headers(request_id)
        logger.info("Streaming via A2A from %s — session='%s'", a2a_endpoint, session_id)

        _MAX_RETRIES = settings.a2a_max_retries
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._client.stream("POST", a2a_endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            parsed = json.loads(data)
                            if not isinstance(parsed, dict):
                                continue
                            result_obj = parsed.get("result", {})
                            if not isinstance(result_obj, dict):
                                continue
                            if result_obj.get("kind") == "artifact-update":
                                for part in result_obj.get("artifact", {}).get("parts", []):
                                    if part.get("kind") == "text" and part.get("text"):
                                        yield part["text"]
                            elif result_obj.get("kind") == "status-update":
                                status = result_obj.get("status", {})
                                state = status.get("state") if isinstance(status, dict) else None
                                if state == "failed":
                                    generic_error = "The agent failed to process your request. Please try again."
                                    msg_obj = status.get("message", {})
                                    if isinstance(msg_obj, dict) and msg_obj.get("text"):
                                        logger.warning("Agent task failed — raw error (suppressed): %s", msg_obj["text"])
                                    yield f"__ERROR__:{generic_error}"
                                    return
                                elif result_obj.get("final"):
                                    return
                        except json.JSONDecodeError:
                            pass
                return
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                backoff = 2 ** attempt
                logger.warning("A2A stream failed (attempt %d/%d): %s — retrying in %ds",
                                attempt + 1, _MAX_RETRIES, e, backoff)
                await asyncio.sleep(backoff)
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                if attempt == _MAX_RETRIES - 1:
                    raise
                backoff = 2 ** attempt
                logger.warning("A2A stream HTTP %d (attempt %d/%d) — retrying in %ds",
                                e.response.status_code, attempt + 1, _MAX_RETRIES, backoff)
                await asyncio.sleep(backoff)
