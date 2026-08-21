#!/usr/bin/env python3
"""
image-edit-overseer

Say what you want changed about an image. An LLM writes the edit prompt, looks
at what came back, and rewrites the prompt until the result actually matches
what you asked for.

Everything runs locally by default: a vision model served by Ollama acts as the
prompt writer and critic, and a diffusion editor runs on the GPU through
diffusers. `--judge claude` swaps the critic for Claude Opus 5 when a local
model keeps missing the point.

  python overseer.py photo.png "make the sky a stormy purple, keep the dog sharp"
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434").rstrip("/")


# --------------------------------------------------------------------------
# The judge's two jobs, as JSON schemas. Ollama takes these as `format`;
# Claude takes them as output_config.format.schema. One definition, two
# backends, so a verdict means the same thing whoever produced it.
# --------------------------------------------------------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["prompt", "reasoning"],
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied": {"type": "boolean"},
        "score": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "revised_prompt": {"type": "string"},
    },
    "required": ["satisfied", "score", "issues", "revised_prompt"],
}


PLAN_SYSTEM = """\
You write prompts for {editor_name}, an instruction-following image editor.

You will see a source image and the change the user wants. Write the single
prompt that will produce that change.

What this editor responds to:
{editor_style}

Rules:
- Describe the desired FINAL image, not the act of editing it.
- Name what must stay unchanged. Untouched regions drift otherwise.
- Be concrete about colour, material, lighting and position. Drop adjectives
  that do not constrain the picture.
- One prompt. No alternatives, no commentary outside the JSON.
"""

CRITIQUE_SYSTEM = """\
You are grading an image edit for {editor_name}.

{layout}

Judge only whether the edited image satisfies the user's request while keeping
everything the user did not ask to change.

The editor regenerates every pixel, so a faithful edit still shifts colours a
little everywhere. That is reconstruction noise, not a defect. Judge what a
person would notice with the two pictures in front of them, at a glance:

- IGNORE slight shifts in shade, brightness or saturation — "light beige to
  very light beige", "gray to slightly lighter gray", mild warmth or contrast
  changes. These are never issues and must never be listed.
- REPORT changes a person would actually see: an object added, removed, moved
  or reshaped; a colour changed enough to be called a different colour (green
  hills turning grey); text altered; identity or style visibly changed.

Score against what was asked, in this order:
- 1-2: the requested change did not happen at all.
- 3-5: it happened, but is wrong or incomplete, or something else visibly
  drifted.
- 6-7: the request is met and drift is minor but noticeable.
- 8-10: the request is met and nothing else visibly changed.

- "satisfied" is true at 8 or above — the request is met and a person would
  not point at anything else and ask what happened to it. Do not withhold it
  over shade differences; if the edit is right and nothing visibly drifted,
  say so and stop.
- "issues" lists only what you would actually report at those thresholds.
  Empty is the correct answer for a good edit.
- "revised_prompt" is a full replacement prompt that fixes the issues. Change
  the wording that failed rather than restating it louder, and keep the parts
  that worked. If satisfied is true, repeat the prompt that worked.

Respond with JSON only.
"""

# Ollama surfaces only one image per request, so the pair is composited into a
# single labelled canvas. That introduces its own trap: the model starts
# reporting the right-hand panel as "shifted right" and its hills as "moved".
# The caveat below is load-bearing, not decoration.
LAYOUT_PANELS = """\
You get ONE picture containing two panels side by side. The LEFT panel is the
original. The RIGHT panel is the edit that was produced. You also get the
user's request and the prompt that was used.

The panels are two separate images shown on one canvas. Their placement on that
canvas means nothing: the right panel is further right, and lower or higher, only
because of how it was pasted. Never report a subject as moved, shifted, resized
or repositioned on that basis. Compare each panel against its own frame — a
house centred in the left panel and centred in the right panel has NOT moved."""

LAYOUT_PAIR = """\
You get two images. The FIRST is the original. The SECOND is the edit that was
produced. You also get the user's request and the prompt that was used."""

EDITOR_STYLE = {
    "flux": (
        "- Plain, direct English. Short declarative clauses beat long prose.\n"
        "- It follows explicit spatial language well: in the upper left,\n"
        "  behind the subject, covering the lower third.\n"
        "- It renders text reliably when the exact string is quoted.\n"
        "- Negations are weak. Say what should be there, not what should not."
    ),
    "qwen": (
        "- Responds well to instruction phrasing: Change X to Y, Remove Z,\n"
        "  Replace the A with a B.\n"
        "- Strong at localised, surgical edits when the target is named precisely.\n"
        "- Preserves identity and layout well; say so explicitly to lock it in.\n"
        "- Handles a short ordered list of changes in one prompt."
    ),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Structured output should make this trivial, but a local model at low
    quantisation will occasionally wrap the object in prose or a fence.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no JSON object in model response:\n{text[:800]}")


def composite_pair(before: Image.Image, after: Image.Image, max_side: int = 768) -> Image.Image:
    """Lay the pair out on one canvas, labelled, for judges that see one image.

    Both panels are scaled to the same height and sit at the same vertical
    offset, so anything that differs between them is a real difference rather
    than an artefact of the layout.
    """
    from PIL import ImageDraw

    def shrink(im: Image.Image) -> Image.Image:
        scale = min(max_side / max(im.size), 1.0)
        if scale >= 1.0:
            return im.convert("RGB")
        return im.convert("RGB").resize(
            (int(im.width * scale), int(im.height * scale)), Image.LANCZOS
        )

    a, b = shrink(before), shrink(after)
    h = max(a.height, b.height)
    pad, bar = 14, 30
    canvas = Image.new("RGB", (a.width + b.width + pad * 3, h + bar + pad * 2), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(a, (pad, bar + pad))
    canvas.paste(b, (pad * 2 + a.width, bar + pad))
    draw.text((pad + 4, 9), "LEFT PANEL = ORIGINAL", fill=(20, 20, 20))
    draw.text((pad * 2 + a.width + 4, 9), "RIGHT PANEL = EDITED", fill=(20, 20, 20))
    return canvas


def fit_dimensions(img: Image.Image, max_side: int, multiple: int = 32) -> tuple[int, int]:
    """Largest size within max_side that keeps aspect and snaps to `multiple`."""
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    w, h = max(int(w * scale), multiple), max(int(h * scale), multiple)
    return (w // multiple) * multiple, (h // multiple) * multiple


def ollama_free_all() -> list[str]:
    """Unload every model Ollama currently holds, not just ours.

    A diffusion pipeline needs the whole card. Ollama cannot see the VRAM
    diffusers is about to take, so anything left resident gets oversubscribed
    and spills into shared system memory over PCIe -- which is roughly two
    orders of magnitude slower than local VRAM and turns a 13s render into
    minutes. Freeing everything first is the difference.
    """
    import requests

    try:
        loaded = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=10).json()
    except Exception as exc:
        print(f"  ! could not query Ollama: {exc}", file=sys.stderr)
        return []

    freed = []
    for entry in loaded.get("models", []):
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        try:
            requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": name, "messages": [], "keep_alive": 0},
                timeout=60,
            )
            freed.append(name)
        except Exception as exc:
            print(f"  ! could not unload {name}: {exc}", file=sys.stderr)
    return freed


# --------------------------------------------------------------------------
# judges
# --------------------------------------------------------------------------

@dataclass
class Plan:
    prompt: str
    reasoning: str


@dataclass
class Verdict:
    satisfied: bool
    score: int
    issues: list[str]
    revised_prompt: str


class Judge:
    """Writes the edit prompt, then grades the result."""

    name = "judge"

    def plan(self, source: Image.Image, request: str, editor: "Editor") -> Plan:
        raise NotImplementedError

    def critique(
        self,
        source: Image.Image,
        candidate: Image.Image,
        request: str,
        prompt: str,
        editor: "Editor",
    ) -> Verdict:
        raise NotImplementedError

    def release(self) -> None:
        """Give up any VRAM. No-op for hosted judges."""


class OllamaJudge(Judge):
    """Local VLM through Ollama.

    Ollama owns the loading and unloading, which is the whole reason it is here:
    the judge and the diffusion pipeline cannot both sit in 32GB, so we hand the
    VRAM back after every call via keep_alive=0.
    """

    def __init__(self, model: str, num_ctx: int = 8192, temperature: float = 0.3):
        import requests  # local import so --judge claude works without it

        self._requests = requests
        self.model = model
        self.name = f"ollama:{model}"
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._verify()

    def _verify(self) -> None:
        try:
            tags = self._requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10).json()
        except Exception as exc:
            raise SystemExit(
                f"cannot reach Ollama at {OLLAMA_BASE} ({exc}).\n"
                "Start it with `ollama serve`, or point OLLAMA_API_BASE elsewhere."
            ) from exc
        have = {m["name"] for m in tags.get("models", [])}
        if self.model not in have and f"{self.model}:latest" not in have:
            raise SystemExit(
                f"Ollama does not have '{self.model}'.\n"
                f"Pull it first:  ollama pull {self.model}\n"
                f"Installed: {', '.join(sorted(have)) or '(none)'}"
            )

    def _chat(self, system: str, user: str, images: list[Image.Image], schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [b64_png(i) for i in images]},
            ],
            "stream": False,
            "format": schema,
            "keep_alive": "5m",
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        last: Exception | None = None
        for attempt in range(2):
            resp = self._requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=900)
            resp.raise_for_status()
            message = resp.json().get("message", {})
            # qwen3-vl reasons before answering. Normally the JSON lands in
            # `content`, but on some turns the whole answer ends up inside
            # `thinking` and `content` comes back empty. Do not "fix" this by
            # sending think=False: the model reasons anyway, still returns no
            # content, and the image stops being processed properly
            # (prompt_eval collapses from ~4600 to ~1400 tokens).
            for field in ("content", "thinking"):
                text = message.get(field) or ""
                if not text.strip():
                    continue
                try:
                    return extract_json(text)
                except ValueError as exc:
                    last = exc
            payload["options"] = dict(payload["options"], temperature=0.0)
            if attempt == 0:
                print("  ! judge returned nothing parseable, retrying at temp 0", file=sys.stderr)
        raise last or ValueError("judge returned an empty response twice")

    def plan(self, source, request, editor):
        data = self._chat(
            PLAN_SYSTEM.format(editor_name=editor.display, editor_style=editor.style),
            f"The user wants this change:\n\n{request}\n\n"
            "Write the prompt that produces it.",
            [source],
            PLAN_SCHEMA,
        )
        return Plan(prompt=data["prompt"].strip(), reasoning=data.get("reasoning", "").strip())

    def critique(self, source, candidate, request, prompt, editor):
        data = self._chat(
            CRITIQUE_SYSTEM.format(editor_name=editor.display, layout=LAYOUT_PANELS),
            f"The user asked for:\n\n{request}\n\n"
            f"The prompt used was:\n\n{prompt}\n\n"
            "Grade the RIGHT panel against the LEFT panel.",
            [composite_pair(source, candidate)],
            CRITIQUE_SCHEMA,
        )
        return Verdict(
            satisfied=bool(data["satisfied"]),
            score=int(data["score"]),
            issues=[str(i) for i in data.get("issues", [])],
            revised_prompt=str(data["revised_prompt"]).strip(),
        )

    def release(self) -> None:
        """Unload from VRAM immediately so the editor can have the card."""
        try:
            self._requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": self.model, "messages": [], "keep_alive": 0},
                timeout=60,
            )
        except Exception as exc:  # unloading is best-effort
            print(f"  ! could not unload {self.model}: {exc}", file=sys.stderr)


class ClaudeJudge(Judge):
    """Claude Opus 5. The escalation path when the local judge keeps missing."""

    MODEL = "claude-opus-5"

    def __init__(self, model: str = MODEL):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def _msg(self, system: str, user: str, images: list[Image.Image], schema: dict) -> dict:
        content: list[dict] = []
        for img in images:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64_png(img)},
                }
            )
        content.append({"type": "text", "text": user})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
        )

        try:
            resp = self.client.messages.create(
                **kwargs,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except TypeError:
            # Older SDK without output_config/fallbacks: ask for JSON in words.
            kwargs["system"] = system + "\n\nRespond with a single JSON object and nothing else."
            resp = self.client.messages.create(**kwargs)

        if getattr(resp, "stop_reason", None) == "refusal":
            detail = getattr(resp, "stop_details", None)
            raise SystemExit(f"Claude declined this request ({detail}). Try --judge local.")

        text = next(b.text for b in resp.content if b.type == "text")
        return extract_json(text)

    def plan(self, source, request, editor):
        data = self._msg(
            PLAN_SYSTEM.format(editor_name=editor.display, editor_style=editor.style),
            f"The user wants this change:\n\n{request}\n\nWrite the prompt that produces it.",
            [source],
            PLAN_SCHEMA,
        )
        return Plan(prompt=data["prompt"].strip(), reasoning=data.get("reasoning", "").strip())

    def critique(self, source, candidate, request, prompt, editor):
        data = self._msg(
            CRITIQUE_SYSTEM.format(editor_name=editor.display, layout=LAYOUT_PAIR),
            f"The user asked for:\n\n{request}\n\n"
            f"The prompt used was:\n\n{prompt}\n\n"
            "Image 1 is the original. Image 2 is the edit. Grade it.",
            [source, candidate],
            CRITIQUE_SCHEMA,
        )
        return Verdict(
            satisfied=bool(data["satisfied"]),
            score=int(data["score"]),
            issues=[str(i) for i in data.get("issues", [])],
            revised_prompt=str(data["revised_prompt"]).strip(),
        )


# --------------------------------------------------------------------------
# editors
# --------------------------------------------------------------------------

def free_vram_gb() -> float:
    """Free VRAM, not total.

    This box runs other Ollama work, and diffusers allocations are invisible to
    Ollama's scheduler (and vice versa). Sizing against the card's total
    capacity assumes we own it; sizing against what is actually free lets the
    loop coexist with whatever else is resident.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


class Editor:
    """A diffusion image editor that can hand its VRAM back between turns."""

    key: str
    display: str
    repo: str
    default_steps: int
    weights_gb: float  # bf16 size of every component, for the offload decision

    def __init__(self, steps: int | None = None, max_side: int = 1024, offload: str = "auto"):
        self.steps = steps or self.default_steps
        self.max_side = max_side
        self.offload = self._decide_offload(offload)
        self.pipe = None
        self._on_gpu = False

    def _decide_offload(self, mode: str) -> bool:
        """Resident weights need headroom for activations, so 0.85 not 1.0.

        FLUX.2 ships a language-model-sized text encoder, which pushes the 9B
        past a 32GB card even though the transformer alone would fit. Offload
        swaps components in one at a time, and each one fits on its own.
        """
        if mode != "auto":
            return mode == "on"
        vram = free_vram_gb()
        if vram == 0.0:
            return False
        need_offload = self.weights_gb > 0.85 * vram
        if need_offload:
            print(
                f"  {self.display} is ~{self.weights_gb:.0f}GB of weights and only "
                f"{vram:.0f}GB is free -> offloading (--offload off to force resident)"
            )
        return need_offload

    @property
    def style(self) -> str:
        return EDITOR_STYLE[self.key]

    def _build(self):
        raise NotImplementedError

    def acquire(self) -> None:
        import torch

        if self.pipe is None:
            print(f"  loading {self.display} ({self.repo}) ...", flush=True)
            t0 = time.time()
            self.pipe = self._build()
            if self.offload:
                self.pipe.enable_model_cpu_offload()
            print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
        if not self.offload and not self._on_gpu:
            self.pipe.to("cuda")
            self._on_gpu = True
        torch.cuda.synchronize()

    def release(self) -> None:
        """Park the weights in system RAM. Far cheaper than reloading from disk."""
        import torch

        if self.pipe is not None and not self.offload and self._on_gpu:
            self.pipe.to("cpu")
            self._on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()

    def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        raise NotImplementedError


class FluxKleinEditor(Editor):
    key = "flux"
    display = "FLUX.2 [klein]"
    default_steps = 4

    # Measured on disk: 9B is 16.4GB text encoder + 18.2GB transformer + VAE.
    # The text encoder dominates and is shared by both sizes.
    _WEIGHTS = {"9B": 34.8, "4B": 24.0}

    def __init__(self, size: str = "9B", **kw):
        self.repo = f"black-forest-labs/FLUX.2-klein-{size}"
        self.weights_gb = self._WEIGHTS[size]
        self.display = f"FLUX.2 [klein] {size}"
        super().__init__(**kw)

    def _build(self):
        import torch
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16)

    def edit(self, image, prompt, seed):
        import torch

        w, h = fit_dimensions(image, self.max_side)
        return self.pipe(
            image=image.convert("RGB").resize((w, h), Image.LANCZOS),
            prompt=prompt,
            height=h,
            width=w,
            guidance_scale=1.0,
            num_inference_steps=self.steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]


class QwenEditor(Editor):
    key = "qwen"
    display = "Qwen-Image-Edit-2511"
    repo = "Qwen/Qwen-Image-Edit-2511"
    default_steps = 40
    weights_gb = 40.0  # 20B in bf16

    def _build(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        return QwenImageEditPlusPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16)

    def edit(self, image, prompt, seed):
        import torch

        w, h = fit_dimensions(image, self.max_side)
        return self.pipe(
            image=[image.convert("RGB").resize((w, h), Image.LANCZOS)],
            prompt=prompt,
            negative_prompt=" ",
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_inference_steps=self.steps,
            num_images_per_prompt=1,
            generator=torch.manual_seed(seed),
        ).images[0]


def build_editor(cfg) -> Editor:
    if cfg.editor == "flux":
        return FluxKleinEditor(
            size=cfg.flux_size, steps=cfg.steps, max_side=cfg.max_side, offload=cfg.offload
        )
    return QwenEditor(steps=cfg.steps, max_side=cfg.max_side, offload=cfg.offload)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

@dataclass
class Settings:
    """Everything a run needs, independent of where it was configured."""

    request: str
    out: str
    max_iters: int = 5
    seed: int = 1234
    editor: str = "flux"
    flux_size: str = "9B"
    steps: int | None = None
    max_side: int = 1024
    offload: str = "auto"
    judge: str = "local"
    judge_model: str = "qwen3.6:27b"
    claude_model: str = ClaudeJudge.MODEL
    num_ctx: int = 8192
    free_ollama: str = "all"
    prompt: str | None = None


def iterate(source: Image.Image, cfg: Settings) -> "Iterator[dict[str, Any]]":
    """Run the refine loop, yielding one event per step.

    A generator rather than a function with prints, so the CLI and the web UI
    drive the same loop instead of keeping two copies of it in step.
    """
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)

    editor = build_editor(cfg)
    judge: Judge = (
        ClaudeJudge(cfg.claude_model)
        if cfg.judge == "claude"
        else OllamaJudge(cfg.judge_model, num_ctx=cfg.num_ctx)
    )

    yield {
        "type": "start",
        "request": cfg.request,
        "editor": editor.display,
        "steps": editor.steps,
        "judge": judge.name,
        "outdir": str(outdir),
        "max_iters": cfg.max_iters,
    }

    log: list[dict[str, Any]] = []
    best: tuple[int, Path] | None = None
    satisfied_at: int | None = None

    prompt = cfg.prompt
    needs_plan = prompt is None

    try:
        for i in range(1, cfg.max_iters + 1):
            if needs_plan:
                editor.release()
                plan = judge.plan(source, cfg.request, editor)
                judge.release()
                prompt = plan.prompt
                yield {
                    "type": "plan",
                    "iteration": i,
                    "prompt": prompt,
                    "reasoning": plan.reasoning,
                }
            needs_plan = False

            if cfg.free_ollama == "all":
                freed = ollama_free_all()
                if freed:
                    yield {"type": "freed", "models": freed}

            editor.acquire()
            t0 = time.time()
            candidate = editor.edit(source, prompt, cfg.seed + i - 1)
            elapsed = time.time() - t0
            path = outdir / f"iter{i:02d}.png"
            candidate.save(path)
            yield {
                "type": "render",
                "iteration": i,
                "image": path.name,
                "prompt": prompt,
                "seconds": round(elapsed, 1),
            }

            editor.release()
            verdict = judge.critique(source, candidate, cfg.request, prompt, editor)
            judge.release()

            yield {
                "type": "critique",
                "iteration": i,
                "score": verdict.score,
                "satisfied": verdict.satisfied,
                "issues": verdict.issues,
            }

            log.append(
                {
                    "iteration": i,
                    "prompt": prompt,
                    "image": path.name,
                    "seconds": round(elapsed, 1),
                    "score": verdict.score,
                    "satisfied": verdict.satisfied,
                    "issues": verdict.issues,
                }
            )
            if best is None or verdict.score > best[0]:
                best = (verdict.score, path)

            if verdict.satisfied:
                satisfied_at = i
                candidate.save(outdir / "final.png")
                break

            prompt = verdict.revised_prompt
            yield {"type": "revised", "iteration": i, "prompt": prompt}
        else:
            if best is not None:
                Image.open(best[1]).save(outdir / "final.png")

        yield {
            "type": "done",
            "satisfied": satisfied_at is not None,
            "iterations": len(log),
            "best_score": best[0] if best else 0,
            "final": "final.png",
        }
    finally:
        # Runs on GeneratorExit too, so a stopped run still leaves its images,
        # a best-so-far final.png, and a log of what was tried.
        editor.release()
        if log:
            if best is not None and not (outdir / "final.png").exists():
                Image.open(best[1]).save(outdir / "final.png")
            (outdir / "log.json").write_text(
                json.dumps(
                    {
                        "request": cfg.request,
                        "editor": editor.repo,
                        "judge": judge.name,
                        "iterations": log,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


def run(args) -> int:
    cfg = Settings(
        request=args.request,
        out=args.out,
        max_iters=args.max_iters,
        seed=args.seed,
        editor=args.editor,
        flux_size=args.flux_size,
        steps=args.steps,
        max_side=args.max_side,
        offload=args.offload,
        judge=args.judge,
        judge_model=args.judge_model,
        claude_model=args.claude_model,
        num_ctx=args.num_ctx,
        free_ollama=args.free_ollama,
        prompt=args.prompt,
    )
    source = Image.open(args.image).convert("RGB")

    for ev in iterate(source, cfg):
        kind = ev["type"]
        if kind == "start":
            print(f"request : {ev['request']}")
            print(f"editor  : {ev['editor']}  ({ev['steps']} steps)")
            print(f"judge   : {ev['judge']}")
            print(f"output  : {ev['outdir']}\n")
        elif kind == "plan":
            print(f"iteration {ev['iteration']} prompt:\n  {ev['prompt']}")
            if ev["reasoning"]:
                print(f"  why: {ev['reasoning']}")
            print()
        elif kind == "freed":
            print(f"  freed from VRAM: {', '.join(ev['models'])}")
        elif kind == "render":
            print(f"  rendered in {ev['seconds']}s -> {ev['image']}")
        elif kind == "critique":
            print(f"  score {ev['score']}/10  satisfied={ev['satisfied']}")
            for issue in ev["issues"]:
                print(f"    - {issue}")
        elif kind == "revised":
            print(f"\n  revised prompt:\n  {ev['prompt']}\n")
        elif kind == "done":
            if ev["satisfied"]:
                print(f"\ndone in {ev['iterations']} iteration(s) -> final.png")
            else:
                print(
                    f"\nstopped after {ev['iterations']} iterations without a pass.\n"
                    f"best scored {ev['best_score']}/10 -> final.png"
                )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Iteratively edit an image until an LLM agrees it matches your request."
    )
    p.add_argument("image", help="source image to edit")
    p.add_argument("request", help="what you want changed, in plain language")
    p.add_argument("-o", "--out", default="runs/latest", help="output directory")
    p.add_argument("-n", "--max-iters", type=int, default=5, help="give up after this many tries")
    p.add_argument("--seed", type=int, default=1234)

    g = p.add_argument_group("editor")
    g.add_argument("--editor", choices=["flux", "qwen"], default="flux")
    g.add_argument(
        "--flux-size",
        choices=["4B", "9B"],
        default="9B",
        help="9B is stronger; 4B is Apache-2.0 and leaves room for the judge",
    )
    g.add_argument("--steps", type=int, default=None, help="override sampler steps")
    g.add_argument("--max-side", type=int, default=1024, help="longest edge of the output")
    g.add_argument(
        "--offload",
        choices=["auto", "on", "off"],
        default="auto",
        help="stream weights from RAM. auto turns it on when the model exceeds the card",
    )

    j = p.add_argument_group("judge")
    j.add_argument("--judge", choices=["local", "claude"], default="local")
    j.add_argument(
        "--judge-model",
        default="qwen3.6:27b",
        help="Ollama vision model for --judge local. Prefer one you already keep "
        "loaded: a second model evicts the first on a single card.",
    )
    j.add_argument("--claude-model", default=ClaudeJudge.MODEL)
    j.add_argument("--num-ctx", type=int, default=8192)
    j.add_argument(
        "--free-ollama",
        choices=["all", "own"],
        default="all",
        help="before each render, unload every model Ollama holds (all, the "
        "default) or only our judge (own, polite on a shared box)",
    )

    p.add_argument(
        "--prompt", default=None, help="skip the first planning call and start from this prompt"
    )

    args = p.parse_args()
    if not Path(args.image).exists():
        p.error(f"no such image: {args.image}")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
