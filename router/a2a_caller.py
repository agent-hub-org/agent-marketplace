import json
import logging
import uuid
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger("marketplace.a2a_caller")


class AgentCaller:
    """Sends tasks to agents via the A2A protocol and supports SSE streaming."""

    async def call_agent(self, agent_url: str, query: str, session_id: str | None = None) -> str:
        """
        Send a message/send request to an A2A agent and return the response text.

        Args:
            agent_url: Base URL of the A2A agent (e.g., http://localhost:9001)
            query: The user's query text
            session_id: Optional session ID for conversation continuity
        """
        task_id = uuid.uuid4().hex
        session_id = session_id or uuid.uuid4().hex

        # A2A message/send JSON-RPC request
        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "message/send",
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "messageId": uuid.uuid4().hex,
                    "role": "user",
                    "parts": [{"type": "text", "text": query}],
                },
                "acceptedOutputModes": ["text"],
            },
        }

        a2a_endpoint = f"{agent_url}/a2a/"
        logger.info("Calling A2A agent at %s — task_id='%s'", a2a_endpoint, task_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            response = await client.post(a2a_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        # Extract the response text from A2A result
        result = data.get("result", {})
        artifacts = result.get("artifacts", [])

        texts = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part["text"])

        # Fallback: check history
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

    async def stream_agent(self, agent_url: str, query: str, session_id: str | None = None) -> AsyncIterator[str]:
        """
        Call an agent's /ask/stream SSE endpoint and yield text chunks.

        This bypasses A2A for streaming — the agents expose /ask/stream which
        returns SSE events with {"text": "..."} payloads.
        """
        session_id = session_id or uuid.uuid4().hex
        stream_url = f"{agent_url}/ask/stream"
        payload = {"query": query, "session_id": session_id}

        logger.info("Streaming from %s — session='%s'", stream_url, session_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream("POST", stream_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]  # strip "data: " prefix
                    if data == "[DONE]":
                        return
                    try:
                        parsed = json.loads(data)
                        if "text" in parsed:
                            yield parsed["text"]
                        elif "session_id" in parsed:
                            # Final event with session_id — yield as metadata
                            yield json.dumps({"session_id": parsed["session_id"]})
                    except json.JSONDecodeError:
                        continue
