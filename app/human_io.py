"""
Pluggable human interaction layer.

CLI mode uses stdin/stdout; Web mode can provide an async handler that waits for
frontend responses without blocking the event loop.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Protocol


class HumanIO(Protocol):
    async def confirm(self, prompt: str, default: str = "y") -> bool: ...

    async def feedback(self, prompt: str, allow_empty: bool = True) -> str: ...

    async def choose(self, prompt: str, *, choices: list[str], default: str) -> str: ...

    async def emit(self, event: dict) -> None: ...


@dataclass
class _NoopEmitter:
    async def emit(self, event: dict) -> None:
        return


class CLIHumanIO:
    """Async wrapper around blocking input() for CLI usage."""

    def __init__(self):
        self._emitter = _NoopEmitter()

    async def emit(self, event: dict) -> None:
        await self._emitter.emit(event)

    async def confirm(self, prompt: str, default: str = "y") -> bool:
        valid_inputs = {
            "y": True,
            "yes": True,
            "n": False,
            "no": False,
            "": default.lower() == "y",
        }

        while True:
            try:
                user_input = await asyncio.to_thread(
                    input, f"{prompt} (y/n, default={default}): "
                )
                user_input = (user_input or "").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return default.lower() == "y"

            if user_input in valid_inputs:
                return valid_inputs[user_input]

            print("Please enter 'y' or 'n' (or press Enter for default)")

    async def feedback(self, prompt: str, allow_empty: bool = True) -> str:
        while True:
            try:
                user_input = await asyncio.to_thread(input, f"{prompt}: ")
                user_input = (user_input or "").strip()
            except (EOFError, KeyboardInterrupt):
                return "" if allow_empty else ""

            if user_input or allow_empty:
                return user_input

            print("Please provide feedback (or press Enter if empty feedback is allowed)")

    async def choose(self, prompt: str, *, choices: list[str], default: str) -> str:
        choices_set = set(choices)
        while True:
            try:
                user_input = await asyncio.to_thread(
                    input, f"{prompt} (choices={choices}, default={default}): "
                )
                user_input = (user_input or "").strip()
            except (EOFError, KeyboardInterrupt):
                return default

            if not user_input:
                return default
            if user_input in choices_set:
                return user_input
            print(f"Please enter one of: {choices}")


_HUMAN_IO: ContextVar[Optional[HumanIO]] = ContextVar("HUMAN_IO", default=None)


def get_human_io() -> HumanIO:
    io = _HUMAN_IO.get()
    return io if io is not None else CLIHumanIO()


def set_human_io(io: HumanIO) -> None:
    _HUMAN_IO.set(io)

