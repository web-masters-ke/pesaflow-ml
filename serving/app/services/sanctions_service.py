"""Sanctions & Watchlist Screening Service."""

from difflib import SequenceMatcher
from typing import Any

import redis.asyncio as redis
from loguru import logger

from serving.app.schemas.aml import SanctionsScreenResult


class SanctionsScreeningService:
    """Screen entities against OFAC, UN, and local sanctions/watchlists."""

    def __init__(self, redis_client: redis.Redis, db_pool: Any = None, fuzzy_threshold: float = 0.85):
        self._redis = redis_client
        self._db = db_pool
        self._fuzzy_threshold = fuzzy_threshold
        # In-memory cache of sanctions entities (loaded at startup)
        self._sanctions_list: list[dict] = []
        self._blacklisted_users: set[str] = set()
        self._blacklisted_devices: set[str] = set()
        self._blacklisted_ips: set[str] = set()

    async def load_sanctions_data(self) -> None:
        """Load sanctions list from DB into memory for fast matching."""
        if self._db:
            try:
                async with self._db.acquire() as conn:
                    rows = await conn.fetch("SELECT * FROM aml_sanctions_entities")
                    self._sanctions_list = [dict(r) for r in rows]
                    logger.info(f"Loaded {len(self._sanctions_list)} sanctions entities")
            except Exception as e:
                logger.error(f"Failed to load sanctions data: {e}")

        # Load blacklists from Redis
        try:
            self._blacklisted_users = await self._load_set("blacklist:users")
            self._blacklisted_devices = await self._load_set("blacklist:devices")
            self._blacklisted_ips = await self._load_set("blacklist:ips")
        except Exception as e:
            logger.error(f"Failed to load blacklists: {e}")

    async def _load_set(self, key: str) -> set[str]:
        members = await self._redis.smembers(key)  # type: ignore[misc]
        return {m.decode() if isinstance(m, bytes) else m for m in members}

    async def screen_user(self, user_name: str, user_id: str, country: str = "") -> SanctionsScreenResult:
        """Screen a user against sanctions and watchlists."""
        # Check internal blacklist first
        if user_id in self._blacklisted_users:
            return SanctionsScreenResult(
                matched=True, match_type="BLACKLIST", matched_entity=user_id, confidence=1.0, source="INTERNAL"
            )

        # Exact name match
        for entity in self._sanctions_list:
            if user_name.lower() == entity.get("name", "").lower():
                return SanctionsScreenResult(
                    matched=True,
                    match_type="EXACT",
                    matched_entity=entity.get("name"),
                    confidence=1.0,
                    source=entity.get("source", "UNKNOWN"),
                )

            # Check aliases
            aliases = entity.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    if user_name.lower() == alias.lower():
                        return SanctionsScreenResult(
                            matched=True,
                            match_type="ALIAS",
                            matched_entity=entity.get("name"),
                            confidence=1.0,
                            source=entity.get("source", "UNKNOWN"),
                        )

        # Fuzzy name match
        for entity in self._sanctions_list:
            ratio = SequenceMatcher(None, user_name.lower(), entity.get("name", "").lower()).ratio()
            if ratio >= self._fuzzy_threshold:
                return SanctionsScreenResult(
                    matched=True,
                    match_type="FUZZY",
                    matched_entity=entity.get("name"),
                    confidence=ratio,
                    source=entity.get("source", "UNKNOWN"),
                )

        # Country-based match
        if country:
            for entity in self._sanctions_list:
                if entity.get("country", "").upper() == country.upper():
                    return SanctionsScreenResult(
                        matched=True,
                        match_type="COUNTRY",
                        matched_entity=entity.get("name"),
                        confidence=0.7,
                        source=entity.get("source", "UNKNOWN"),
                    )

        return SanctionsScreenResult(matched=False)

    def is_blacklisted_device(self, device_id: str | None) -> bool:
        if not device_id:
            return False
        return device_id in self._blacklisted_devices

    def is_blacklisted_ip(self, ip_address: str | None) -> bool:
        if not ip_address:
            return False
        return ip_address in self._blacklisted_ips

    def is_blacklisted_user(self, user_id: str) -> bool:
        return user_id in self._blacklisted_users
