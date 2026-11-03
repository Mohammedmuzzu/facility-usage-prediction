"""Hashed/scoped service keys, byte limits and process-safe request quotas."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
from fastapi import HTTPException
from starlette.responses import JSONResponse


@dataclass(frozen=True)
class ServiceSettings:
    environment: str = "development"
    auth_file: str | None = None
    rate_db: str = ".cache/api_rate.sqlite3"
    requests_per_minute: int = 120
    max_body_bytes: int = 16_384
    max_concurrency: int = 4
    max_data_age_hours: float = 24

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            environment=os.getenv("FACILITY_ENV", "development"),
            auth_file=os.getenv("FACILITY_AUTH_FILE"),
            rate_db=os.getenv("FACILITY_RATE_DB", ".cache/api_rate.sqlite3"),
            requests_per_minute=int(os.getenv("FACILITY_RATE_LIMIT", "120")),
            max_body_bytes=int(os.getenv("FACILITY_MAX_BODY_BYTES", "16384")),
            max_concurrency=int(os.getenv("FACILITY_MAX_CONCURRENCY", "4")),
            max_data_age_hours=float(os.getenv("FACILITY_MAX_DATA_AGE_HOURS", "24")),
        )

    def __post_init__(self) -> None:
        import math

        if self.environment not in ["development", "production"]:
            raise ValueError("FACILITY_ENV must be development or production")
        if (
            min(self.requests_per_minute, self.max_body_bytes, self.max_concurrency)
            <= 0
            or not math.isfinite(self.max_data_age_hours)
            or self.max_data_age_hours <= 0
        ):
            raise ValueError("Service limits must be positive and finite")
        if self.environment == "production" and not self.auth_file:
            raise ValueError("Production requires FACILITY_AUTH_FILE")
        if self.auth_file:
            read_principals(self.auth_file)


def read_principals(location: str) -> list[dict]:
    path = Path(location)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Credential file is too large")
    principals = json.loads(path.read_text())["principals"]
    if not isinstance(principals, list) or not 1 <= len(principals) <= 1000:
        raise ValueError("Require 1–1000 service principals")
    seen = set()
    for principal in principals:
        digest = principal.get("key_sha256", "")
        if (
            len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or digest in seen
        ):
            raise ValueError("Credential hashes must be unique SHA-256 hex digests")
        seen.add(digest)
        if (
            not isinstance(principal.get("id"), str)
            or not 1 <= len(principal["id"]) <= 100
        ):
            raise ValueError("Require a bounded principal ID")
        if not isinstance(principal.get("residents"), list) or any(
            not isinstance(value, str) for value in principal["residents"]
        ):
            raise ValueError("Require explicit resident scopes")
        if type(principal.get("admin", False)) is not bool:
            raise ValueError("admin must be boolean")
    return principals


def authenticate(settings: ServiceSettings, key: str | None) -> dict:
    if not settings.auth_file:
        return {
            "id": "local-development",
            "admin": True,
            "residents": [],
            "key_sha256": "development",
        }
    if not key or len(key) > 256:
        raise HTTPException(401, "Valid API key required")
    digest = hashlib.sha256(key.encode()).hexdigest()
    try:
        principals = read_principals(settings.auth_file)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Authentication configuration unavailable") from exc
    match = None
    for principal in principals:
        if secrets.compare_digest(digest, principal["key_sha256"]):
            match = principal
    if match is None:
        raise HTTPException(401, "Valid API key required")
    return match


def authorize(principal: dict, resident: str | None = None) -> None:
    if principal.get("admin"):
        return
    if resident is None or resident not in principal["residents"]:
        raise HTTPException(403, "Access denied")


class RateLimiter:
    """Atomic SQLite quota shared by workers on one host/volume, not across hosts."""

    def __init__(self, path: str, limit: int):
        self.path, self.limit = path, limit
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                "CREATE TABLE IF NOT EXISTS quota (principal TEXT, window INTEGER, count INTEGER, PRIMARY KEY(principal, window))"
            )

    def check(self, principal: str, now: float | None = None) -> None:
        window = int(time.time() if now is None else now) // 60
        try:
            with sqlite3.connect(self.path, timeout=2) as database:
                database.execute("BEGIN IMMEDIATE")
                row = database.execute(
                    "SELECT count FROM quota WHERE principal=? AND window=?",
                    (principal, window),
                ).fetchone()
                if row and row[0] >= self.limit:
                    raise HTTPException(
                        429, "Request quota exceeded", headers={"Retry-After": "60"}
                    )
                database.execute(
                    "INSERT INTO quota VALUES (?,?,1) ON CONFLICT(principal,window) DO UPDATE SET count=count+1",
                    (principal, window),
                )
                database.execute("DELETE FROM quota WHERE window < ?", (window - 2,))
        except sqlite3.Error as exc:
            raise HTTPException(503, "Request quota service unavailable") from exc


class BodyLimitMiddleware:
    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] not in ["POST", "PUT", "PATCH"]:
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.maximum:
                await JSONResponse(
                    {"detail": "Request body too large"}, status_code=413
                )(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
