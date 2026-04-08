"""Authenticate against a Dalgo environment and obtain auth cookies."""

import logging

import httpx

from config import Config

logger = logging.getLogger(__name__)


def login(cfg: Config) -> dict:
    """POST /api/v2/login/ and return the cookies (access_token, refresh_token).

    Returns a dict with 'access_token' and 'refresh_token' cookie values.
    """
    logger.info("Logging in to %s as %s", cfg.login_url, cfg.USERNAME)
    response = httpx.post(
        cfg.login_url,
        json={"username": cfg.USERNAME, "password": cfg.PASSWORD},
    )
    response.raise_for_status()

    cookies = {c.name: c.value for c in response.cookies.jar}
    if "access_token" not in cookies:
        raise ValueError(
            f"Login response missing 'access_token' cookie. "
            f"Response cookies: {list(cookies.keys())}"
        )
    logger.info("Login successful, got cookies: %s", list(cookies.keys()))
    return cookies
