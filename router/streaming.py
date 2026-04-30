"""Shared SSE streaming logic for marketplace query endpoints."""
import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator

logger = logging.getLogger("marketplace.streaming")

_HEARTBEAT_INTERVAL = 15.0
_PROGRESS_PREFIX = "__PROGRESS__:"
_ERROR_PREFIX = "__ERROR__:"


async def build_marketplace_sse_stream(
    source: AsyncIterator[str],
    *,
    preamble: str | None = None,
) -> AsyncGenerator[str, None]:
    """Wrap an async text iterator as an SSE event stream.

    Args:
        source: Async iterator of text chunks from the agent.
        preamble: Optional SSE data line to emit before the heartbeat/queue loop
                  (e.g. routing metadata). Must be a complete ``data: ...\\n\\n`` line.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)

    async def _heartbeat():
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                await queue.put(f": heartbeat {int(asyncio.get_running_loop().time())}\n\n")
        except asyncio.CancelledError:
            pass

    async def _producer():
        try:
            async for chunk in source:
                try:
                    await asyncio.wait_for(queue.put(chunk), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("SSE queue full — client likely disconnected")
                    return
        except Exception as exc:
            logger.error("Marketplace stream producer failed: %s", exc, exc_info=True)
            await queue.put(f"{_ERROR_PREFIX}An error occurred while communicating with the agent. Please try again.")
        finally:
            # Retry until the sentinel lands — the consumer must receive it to exit.
            while True:
                try:
                    queue.put_nowait(None)
                    break
                except asyncio.QueueFull:
                    await asyncio.sleep(0.05)

    if preamble:
        yield preamble

    heartbeat_task = asyncio.create_task(_heartbeat())
    producer_task = asyncio.create_task(_producer())

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break

            if chunk.startswith(": heartbeat"):
                yield chunk
            elif chunk.startswith(_PROGRESS_PREFIX):
                phase = chunk[len(_PROGRESS_PREFIX):]
                yield f"event: progress\ndata: {json.dumps({'phase': phase})}\n\n"
            elif chunk.startswith(_ERROR_PREFIX):
                error_msg = chunk[len(_ERROR_PREFIX):]
                yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"
                yield f"data: {json.dumps({'text': error_msg})}\n\n"
            else:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
    finally:
        heartbeat_task.cancel()
        producer_task.cancel()
        for t in (heartbeat_task, producer_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        yield "data: [DONE]\n\n"
