"""Human interaction utilities for flow execution.

This module previously used blocking input() which breaks web-server usage.
It now delegates to a pluggable async HumanIO handler.
"""

from __future__ import annotations

import asyncio

from app.human_io import get_human_io


async def ask_human_confirmation(prompt: str, default: str = "y") -> bool:
    return await get_human_io().confirm(prompt, default=default)


async def ask_human_feedback(prompt: str, allow_empty: bool = True) -> str:
    return await get_human_io().feedback(prompt, allow_empty=allow_empty)


async def display_text_with_pagination(text: str, page_size: int = 20) -> None:
    """
    CLI-friendly pagination. Web mode should display text directly.
    """
    # In web mode we still emit the text; UI can paginate.
    await get_human_io().emit({"type": "text", "text": text, "page_size": page_size})

    # Keep CLI behavior.
    lines = text.split("\n")
    total_lines = len(lines)
    if total_lines <= page_size:
        print(text)
        return

    current = 0
    while current < total_lines:
        end = min(current + page_size, total_lines)
        print("\n".join(lines[current:end]))
        if end < total_lines:
            user_input = await get_human_io().feedback(
                f"\n--- Showing {end}/{total_lines} lines. Press Enter for more, 'q' to quit",
                allow_empty=True,
            )
            if (user_input or "").strip().lower() == "q":
                break
        current = end
