from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class Session:
    id: str = field(default_factory=lambda: new_id("sess"))
    created_at: float = field(default_factory=time.time)
    events: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    pending_human: dict[str, "asyncio.Future[Any]"] = field(default_factory=dict)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: Optional[str] = None

    async def emit(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("ts", time.time())
        await self.events.put(event)

    async def create_human_request(self, payload: dict) -> Any:
        request_id = new_id("hr")
        fut: "asyncio.Future[Any]" = asyncio.get_running_loop().create_future()
        self.pending_human[request_id] = fut
        await self.emit({"type": "human_request", "request_id": request_id, **payload})
        return request_id, fut

    async def resolve_human(self, request_id: str, value: Any) -> None:
        fut = self.pending_human.get(request_id)
        if not fut or fut.done():
            return
        fut.set_result(value)

    async def fail(self, message: str) -> None:
        self.error = message
        await self.emit({"type": "error", "message": message})
        self.done.set()

