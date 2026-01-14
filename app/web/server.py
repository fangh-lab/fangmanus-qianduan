from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.data_analysis import DataAnalysis
from app.agent.manus import Manus
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.human_io import set_human_io
from app.logger import logger
from app.web.human_web import WebHumanIO
from app.web.session import Session


def create_app(*, web_dir: str = "web") -> FastAPI:
    app = FastAPI(title="Manus Web UI", version="0.1.0")

    sessions: dict[str, Session] = {}

    async def run_session(
        session: Session,
        *,
        prompt: str,
        planning_context: str,
        business_files: dict[str, str],
    ) -> None:
        try:
            set_human_io(WebHumanIO(session))
            await session.emit(
                {
                    "type": "session_start",
                    "session_id": session.id,
                    "workspace_root": str(config.workspace_root),
                }
            )

            agents = {"manus": Manus()}
            if config.run_flow_config.use_data_analysis_agent:
                agents["data_analysis"] = DataAnalysis()

            flow = FlowFactory.create_flow(flow_type=FlowType.PLANNING, agents=agents)
            await session.emit({"type": "run_start"})
            result = await flow.execute(
                prompt, planning_context=planning_context, business_files=business_files
            )
            await session.emit({"type": "run_end", "result": result})
            session.done.set()
        except Exception as e:
            logger.exception(f"Session {session.id} failed: {e}")
            await session.fail(str(e))

    def get_session(session_id: str) -> Session:
        s = sessions.get(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="session not found")
        return s

    @app.get("/")
    async def index():
        index_path = Path(web_dir) / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="web UI not built")
        return FileResponse(str(index_path))

    if Path(web_dir).exists():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.post("/api/run")
    async def api_run(
        prompt: Annotated[str, Form(...)],
        planning_template: Annotated[Optional[UploadFile], File()] = None,
        business_files: Annotated[Optional[list[UploadFile]], File()] = None,
    ):
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is empty")

        planning_context = ""
        if planning_template is not None:
            content = await planning_template.read()
            planning_context = content.decode("utf-8", errors="replace")

        bf: dict[str, str] = {}
        if business_files:
            for f in business_files:
                data = await f.read()
                bf[f.filename or "unnamed"] = data.decode("utf-8", errors="replace")

        session = Session()
        sessions[session.id] = session
        asyncio.create_task(
            run_session(
                session,
                prompt=prompt,
                planning_context=planning_context,
                business_files=bf,
            )
        )
        return {"session_id": session.id}

    @app.get("/api/sessions/{session_id}/events")
    async def api_events(session_id: str):
        session = get_session(session_id)

        async def gen():
            # Initial ping so EventSource opens cleanly
            yield "event: ping\ndata: {}\n\n"
            while True:
                if session.done.is_set() and session.events.empty():
                    break
                try:
                    evt = await asyncio.wait_for(session.events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                etype = evt.get("type", "message")
                yield f"event: {etype}\n"
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/sessions/{session_id}/human/{request_id}")
    async def api_human_response(
        session_id: str,
        request_id: str,
        value: Annotated[str, Form(...)],
    ):
        session = get_session(session_id)
        await session.resolve_human(request_id, value)
        await session.emit({"type": "human_response", "request_id": request_id})
        return {"ok": True}

    return app

