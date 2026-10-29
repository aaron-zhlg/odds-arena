"""Signed client for the Polymarket US API (https://docs.polymarket.us/api-reference/authentication).

Each request is signed with Ed25519 over `{timestamp_ms}{METHOD}{path}` and sent with
X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature headers. Timestamps must be within
30 seconds of server time, so keep the machine clock in sync.
"""
from __future__ import annotations

import base64
import os
import time
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

BASE_URL = os.getenv("POLYMARKET_US_API", "https://api.polymarket.us")


def load_private_key(secret_key_b64: str) -> Ed25519PrivateKey:
    # The decoded secret's first 32 bytes are the Ed25519 private key seed.
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(secret_key_b64)[:32])


def sign_headers(key_id: str, private_key: Ed25519PrivateKey, method: str, path: str, timestamp_ms: int | None = None) -> dict:
    ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
    # Sign the path only; query parameters are sent separately (matches the official example).
    path = urlsplit(path).path
    signature = private_key.sign(f"{ts}{method.upper()}{path}".encode())
    return {
        "X-PM-Access-Key": key_id,
        "X-PM-Timestamp": ts,
        "X-PM-Signature": base64.b64encode(signature).decode(),
    }


class PolymarketUSClient:
    def __init__(self, key_id: str | None = None, secret_key: str | None = None, base_url: str = BASE_URL):
        key_id = key_id or os.getenv("POLYMARKET_KEY_ID")
        secret_key = secret_key or os.getenv("POLYMARKET_SECRET_KEY")
        if not key_id or not secret_key:
            raise RuntimeError("Set POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY in .env")
        self.key_id = key_id
        self._private_key = load_private_key(secret_key)
        self._http = httpx.Client(base_url=base_url, timeout=30)

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = sign_headers(self.key_id, self._private_key, method, path)
        headers["Content-Type"] = "application/json"
        r = self._http.request(method, path, headers=headers | kwargs.pop("headers", {}), **kwargs)
        r.raise_for_status()
        return r

    def get(self, path: str, params: dict | None = None):
        return self.request("GET", path, params=params).json()

    def positions(self):
        return self.get("/v1/portfolio/positions")
