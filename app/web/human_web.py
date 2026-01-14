from __future__ import annotations

import asyncio
from typing import Any

from app.web.session import Session


class WebHumanIO:
    def __init__(self, session: Session, *, request_timeout_s: int = 3600):
        self.session = session
        self.request_timeout_s = request_timeout_s

    async def emit(self, event: dict) -> None:
        await self.session.emit(event)

    async def confirm(self, prompt: str, default: str = "y") -> bool:
        request_id, fut = await self.session.create_human_request(
            {"kind": "confirm", "prompt": prompt, "default": default}
        )
        try:
            val = await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        except asyncio.TimeoutError:
            await self.session.emit(
                {
                    "type": "human_timeout",
                    "request_id": request_id,
                    "prompt": prompt,
                    "default": default,
                }
            )
            return default.lower() == "y"
        if isinstance(val, bool):
            return val
        if val is None:
            return default.lower() == "y"
        s = str(val).strip().lower()
        if s in ("y", "yes", "true", "1"):
            return True
        if s in ("n", "no", "false", "0"):
            return False
        return default.lower() == "y"

    async def feedback(self, prompt: str, allow_empty: bool = True) -> str:
        request_id, fut = await self.session.create_human_request(
            {"kind": "text", "prompt": prompt, "allow_empty": allow_empty}
        )
        try:
            val = await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        except asyncio.TimeoutError:
            await self.session.emit(
                {
                    "type": "human_timeout",
                    "request_id": request_id,
                    "prompt": prompt,
                }
            )
            return ""
        text = "" if val is None else str(val)
        if (not text) and (not allow_empty):
            # Minimal validation: if empty not allowed, keep waiting by re-asking.
            return await self.feedback(prompt, allow_empty=allow_empty)
        return text

    async def choose(self, prompt: str, *, choices: list[str], default: str) -> str:
        request_id, fut = await self.session.create_human_request(
            {"kind": "choice", "prompt": prompt, "choices": choices, "default": default}
        )
        try:
            val = await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        except asyncio.TimeoutError:
            await self.session.emit(
                {"type": "human_timeout", "request_id": request_id, "prompt": prompt}
            )
            return default
        val = default if val is None else str(val)
        if val not in set(choices):
            return default
        return val

    async def coerce_response(self, kind: str, raw: Any, default: Any) -> Any:
        if kind == "confirm":
            if isinstance(raw, bool):
                return raw
            if raw is None:
                return default
            s = str(raw).strip().lower()
            if s in ("y", "yes", "true", "1"):
                return True
            if s in ("n", "no", "false", "0"):
                return False
            return default
        return raw

