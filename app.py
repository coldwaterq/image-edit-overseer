#!/usr/bin/env python3
"""
Web UI for image-edit-overseer.

    python app.py            # then open http://127.0.0.1:7860

Drop in an image, say what you want changed, and watch each attempt arrive with
the prompt that produced it and the judge's verdict. The loop itself lives in
overseer.iterate(); this only drives it and streams the events out.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image

from overseer import Settings, iterate

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
WEB = ROOT / "web"

app = FastAPI(title="image-edit-overseer")

# run_id -> {"queue": Queue, "dir": Path, "done": bool}
RUN_STATE: dict[str, dict] = {}


def _worker(run_id: str, source_path: Path, cfg: Settings) -> None:
    """Drive the loop on a thread, pushing events onto the run's queue.

    The loop is heavy and fully synchronous (CUDA, blocking HTTP), so it gets
    its own thread and the event loop only ever touches the queue.
    """
    q: queue.Queue = RUN_STATE[run_id]["queue"]
    try:
        source = Image.open(source_path).convert("RGB")
        for event in iterate(source, cfg):
            q.put(event)
    except Exception as exc:  # surface failures in the UI, not just the console
        q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        q.put(None)  # sentinel: stream complete


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.post("/api/run")
async def start_run(
    image: UploadFile = File(...),
    request: str = Form(...),
    max_iters: int = Form(5),
    editor: str = Form("flux"),
    flux_size: str = Form("9B"),
    judge_model: str = Form("qwen3.6:27b"),
    judge: str = Form("local"),
    max_side: int = Form(1024),
    num_ctx: int = Form(32768),
) -> dict:
    if not request.strip():
        raise HTTPException(400, "describe the edit you want")

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    outdir = RUNS / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    source_path = outdir / "source.png"
    Image.open(image.file).convert("RGB").save(source_path)

    cfg = Settings(
        request=request.strip(),
        out=str(outdir),
        max_iters=max_iters,
        editor=editor,
        flux_size=flux_size,
        judge=judge,
        judge_model=judge_model,
        max_side=max_side,
        num_ctx=num_ctx,
    )

    RUN_STATE[run_id] = {"queue": queue.Queue(), "dir": outdir}
    threading.Thread(
        target=_worker, args=(run_id, source_path, cfg), daemon=True
    ).start()
    return {"run_id": run_id, "source": f"/api/image/{run_id}/source.png"}


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    if run_id not in RUN_STATE:
        raise HTTPException(404, "no such run")
    q: queue.Queue = RUN_STATE[run_id]["queue"]

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            # Blocking get on the executor keeps the event loop responsive.
            event = await loop.run_in_executor(None, q.get)
            if event is None:
                yield "data: " + json.dumps({"type": "close"}) + "\n\n"
                break
            if event.get("type") in {"render", "done"} and event.get("image", "final.png"):
                name = event.get("image") or "final.png"
                event["url"] = f"/api/image/{run_id}/{name}"
            yield "data: " + json.dumps(event) + "\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/image/{run_id}/{name}")
def image(run_id: str, name: str) -> FileResponse:
    if run_id not in RUN_STATE:
        raise HTTPException(404, "no such run")
    path = (RUN_STATE[run_id]["dir"] / name).resolve()
    if not path.is_file() or RUN_STATE[run_id]["dir"].resolve() not in path.parents:
        raise HTTPException(404, "no such image")
    return FileResponse(path)


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    print("image-edit-overseer  ->  http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
