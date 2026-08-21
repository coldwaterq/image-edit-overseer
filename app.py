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
import sys
import threading
import time
import traceback
import uuid
from dataclasses import replace
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

# run_id -> {"dir": Path, "cancel": Event, "running": bool, "cfg": Settings, ...}
RUN_STATE: dict[str, dict] = {}


def _run_dir(run_id: str) -> Path:
    state = RUN_STATE.get(run_id)
    if state:
        return state["dir"]
    d = (RUNS / run_id).resolve()
    if not d.is_dir() or RUNS.resolve() not in d.parents:
        raise HTTPException(404, "no such run")
    return d


def _worker(run_id: str, source_path: Path, cfg: Settings) -> None:
    """Drive the loop on a thread, appending each event to events.jsonl.

    The loop is heavy and fully synchronous (CUDA, blocking HTTP), so it gets
    its own thread; the event loop only ever reads the file.
    """
    state = RUN_STATE[run_id]
    cancel: threading.Event = state["cancel"]
    gen = None

    # events.jsonl is the record of the run, and the only thing the stream
    # reads. A crash must survive nobody watching: without this a failed run
    # looks identical to one still working.
    events_path = state["dir"] / "events.jsonl"

    def emit(event: dict) -> None:
        try:
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + chr(10))
        except Exception:
            pass
        if event.get("type") == "error":
            print(f"[{run_id}] {event['message']}", file=sys.stderr, flush=True)

    # Thinking arrives as hundreds of few-character deltas. Writing each one as
    # its own event would bloat the log and swamp the browser, so they are
    # coalesced into at most a few updates a second.
    pending: dict = {"phase": None, "text": [], "at": 0.0}

    def flush_thinking(force: bool = False) -> None:
        if not pending["text"]:
            return
        now = time.monotonic()
        if not force and now - pending["at"] < 0.4:
            return
        emit({
            "type": "thinking",
            "phase": pending["phase"],
            "delta": "".join(pending["text"]),
        })
        pending["text"] = []
        pending["at"] = now

    def side_event(event: dict) -> None:
        if event["type"] == "thinking":
            if event.get("phase") != pending["phase"]:
                flush_thinking(force=True)
                pending["phase"] = event.get("phase")
            pending["text"].append(event.get("delta", ""))
            flush_thinking()
        else:
            flush_thinking(force=True)
            emit(event)

    try:
        source = Image.open(source_path).convert("RGB")
        gen = iterate(source, cfg, on_event=side_event)
        for event in gen:
            flush_thinking(force=True)
            if event["type"] in ("criteria", "critique") and event.get("criteria"):
                state["criteria"] = event["criteria"]
            if event["type"] == "render":
                state["last_image"] = event["image"]
                state["last_prompt"] = event["prompt"]
                state["last_iter"] = event["iteration"]
            emit(event)
            if cancel.is_set():
                # A render or a critique cannot be interrupted mid-flight, so
                # the earliest safe exit is the next event boundary. Closing
                # the generator runs its finally: weights released, log saved.
                gen.close()
                emit({"type": "stopped"})
                break
    except Exception as exc:  # surface failures in the UI, not just the console
        emit({
            "type": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    finally:
        if gen is not None:
            gen.close()
        state["running"] = False


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
    max_side: int = Form(0),
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
        max_side=max_side or None,
        num_ctx=num_ctx,
    )

    RUN_STATE[run_id] = {
        "dir": outdir,
        "cancel": threading.Event(),
        "running": True,
        "cfg": cfg,
        "source": source_path,
        "criteria": [],
        "last_image": None,
        "last_prompt": None,
        "last_iter": 0,
    }
    threading.Thread(
        target=_worker, args=(run_id, source_path, cfg), daemon=True
    ).start()
    return {"run_id": run_id, "source": f"/api/image/{run_id}/source.png"}


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    """Replay everything that has happened, then follow along live.

    This tails the run's events.jsonl rather than draining an in-memory queue.
    A queue can only be read once, so a browser that reloaded -- or a reader
    that was abandoned mid-request -- would consume events nobody ever saw. A
    file can be read from the start by any number of viewers, so refreshing the
    page or attaching to a run someone else started both just work.
    """
    outdir = _run_dir(run_id)
    events_path = outdir / "events.jsonl"

    async def gen():
        pos = 0
        idle = 0.0
        while True:
            chunk = ""
            if events_path.exists():
                with events_path.open("r", encoding="utf-8") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
            if chunk:
                idle = 0.0
                for line in chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    name = event.get("image") or (
                        "final.png" if event.get("type") == "done" else None
                    )
                    if name:
                        event["url"] = f"/api/image/{run_id}/{name}"
                    yield "data: " + json.dumps(event) + chr(10) * 2
            else:
                state = RUN_STATE.get(run_id)
                live = bool(state and state["running"])
                if not live:
                    # Give the writer a moment to flush its final events, then
                    # close rather than holding the connection open forever.
                    idle += 0.4
                    if idle > 1.6:
                        yield "data: " + json.dumps({"type": "close"}) + chr(10) * 2
                        return
                await asyncio.sleep(0.4)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs")
def list_runs(limit: int = 25) -> dict:
    """Every run on disk, newest first, so the UI can offer them."""
    out = []
    for d in sorted(RUNS.glob("*/"), key=lambda p: p.name, reverse=True)[:limit]:
        events = d / "events.jsonl"
        if not events.exists():
            continue
        request, attempts, satisfied = "", 0, False
        try:
            for line in events.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["type"] == "start":
                    request = e.get("request", "")
                elif e["type"] == "render":
                    attempts = max(attempts, e.get("iteration", 0))
                elif e["type"] == "critique" and e.get("satisfied"):
                    satisfied = True
        except Exception:
            continue
        state = RUN_STATE.get(d.name)
        out.append(
            {
                "run_id": d.name,
                "request": request,
                "attempts": attempts,
                "satisfied": satisfied,
                "running": bool(state and state["running"]),
                "resumable": bool(state and not state["running"] and state["last_image"]),
            }
        )
    return {"runs": out}


@app.post("/api/continue/{run_id}")
async def continue_run(
    run_id: str,
    criteria: str = Form(...),
    max_iters: int = Form(3),
) -> dict:
    """Carry on from the last result against hand-edited criteria.

    The criteria are the specification, so editing them is how you say what you
    actually wanted. Same directory, same settings, numbering continues, so the
    attempts already made are kept rather than redone.
    """
    state = RUN_STATE.get(run_id)
    if state is None:
        raise HTTPException(404, "no such run")
    if state["running"]:
        raise HTTPException(409, "that run is still going")
    if not state["last_image"]:
        raise HTTPException(400, "nothing to continue from yet")

    edited = [c.strip() for c in json.loads(criteria) if c and c.strip()]
    if not edited:
        raise HTTPException(400, "keep at least one criterion")

    cfg = replace(
        state["cfg"],
        max_iters=max_iters,
        criteria=edited,
        prior_prompt=state["last_prompt"],
        resume_from=state["last_image"],
        start_index=state["last_iter"] + 1,
        prompt=None,
    )
    state.update(cancel=threading.Event(), running=True, cfg=cfg)
    threading.Thread(
        target=_worker, args=(run_id, state["source"], cfg), daemon=True
    ).start()
    return {"run_id": run_id, "from_attempt": state["last_iter"], "criteria": edited}


@app.post("/api/stop/{run_id}")
def stop(run_id: str) -> dict:
    """Ask a run to stop at the next event boundary."""
    if run_id not in RUN_STATE:
        raise HTTPException(404, "no such run")
    RUN_STATE[run_id]["cancel"].set()
    return {"stopping": True, "running": RUN_STATE[run_id]["running"]}


@app.get("/api/image/{run_id}/{name}")
def image(run_id: str, name: str) -> FileResponse:
    outdir = _run_dir(run_id)
    path = (outdir / name).resolve()
    if not path.is_file() or outdir.resolve() not in path.parents:
        raise HTTPException(404, "no such image")
    return FileResponse(path)


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    print("image-edit-overseer  ->  http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
