"""WebSocket client for sending eval questions and capturing responses."""

import asyncio
import json
import logging
import uuid

import websockets

from config import Config

logger = logging.getLogger(__name__)


async def ask_question(
    ws_url: str,
    cookies: dict,
    question: str,
    session_id: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Send a question via WebSocket and wait for the assistant_message response.

    Returns the full assistant_message data dict including payload with
    intent, sql, sql_results, answer_text, etc.
    """
    # Build cookie header from auth cookies
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
    # Origin header is required — AllowedHostsOriginValidator rejects without it
    origin = ws_url.replace("ws://", "http://").replace("wss://", "https://").split("/wss/")[0]
    headers = {"Cookie": cookie_header, "Origin": origin}

    logger.debug("Connecting to %s", ws_url)
    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        client_message_id = str(uuid.uuid4())
        message = {
            "action": "send_message",
            "message": question,
            "session_id": session_id,
            "client_message_id": client_message_id,
        }
        await ws.send(json.dumps(message))
        logger.debug("Sent message %s: %s", client_message_id, question)

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.error("Timeout after %ss for: %s", timeout, question)
                raise TimeoutError(f"No response within {timeout}s for: {question}")

            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            response = json.loads(raw)

            if response.get("status") == "error":
                error_msg = response.get("message", "Unknown error")
                logger.error("Server error for '%s': %s", question, error_msg)
                return {
                    "error": error_msg,
                    "event_type": "error",
                }

            data = response.get("data", {})
            event_type = data.get("event_type")

            if event_type == "progress":
                logger.debug("Progress: %s", data.get("label"))
            elif event_type == "assistant_message":
                intent = data.get("payload", {}).get("intent", "unknown")
                latency = data.get("response_latency_ms")
                logger.info(
                    "Got response for '%s' — intent=%s, latency=%sms",
                    question, intent, latency,
                )
                return data


def run_single_question(cfg: Config, cookies: dict, question: str) -> dict:
    """Synchronous wrapper to ask a single question."""
    ws_url = f"{cfg.ws_url}?orgslug={cfg.ORG_SLUG}"
    return asyncio.run(ask_question(ws_url, cookies, question))
