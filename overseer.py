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
from dataclasses import dataclass, replace
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

# The judge used to be asked "what is wrong with this?" and would answer with
# whatever it noticed first -- reliably colour and style, rarely geometry. A
# request like "run the gate from that post to the near side" was graded 10/10
# with the gate not touching the post at all: it verified that a post existed
# and a gate existed, never that they met. Deciding what must be true BEFORE
# seeing the result, then forcing a verdict on each item, is what stops the
# judge from grading only the axis it finds easy.
CRITERIA_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criteria"],
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "met": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["criterion", "met", "note"],
            },
        },
        "drift": {"type": "array", "items": {"type": "string"}},
        "new_criteria": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "integer"},
        "satisfied": {"type": "boolean"},
        "revised_prompt": {"type": "string"},
    },
    "required": [
        "checks",
        "drift",
        "new_criteria",
        "score",
        "satisfied",
        "revised_prompt",
    ],
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

CRITIQUE_SYSTEM = """You are grading an image edit for {editor_name}.

{layout}

You are given acceptance criteria that were written BEFORE this image was
generated. Rule on each one: does the edited image satisfy it, yes or no?

Answer the criterion you were given, not one you find easier to check, and
never skip one because it is hard to see. If a criterion is about two things
touching, connecting or spanning, look at the exact place they should meet and
say in the note what is actually there -- a join, or a gap, and roughly how
big. "There is a post and there is a gate" does not answer whether they meet.

Then, separately, list anything that visibly drifted which was not asked for.

For each thing that drifted, also write a criterion that would have caught it,
so it cannot come back. Phrase it as a fact about the finished picture, not an
instruction: "the gate is a solid wooden panel, not horizontal rails with gaps"
rather than "do not change the gate style". These are added to the criteria
permanently and every later attempt is graded against them too, which is what
stops the picture wandering away from the original one fix at a time, or
flipping back and forth between two wrong versions.

Only add a criterion for something that actually drifted in THIS image. Do not
restate a criterion you already have, and do not invent guards for things that
are still fine.

The editor regenerates every pixel, so a faithful edit still shifts colours a
little everywhere. That is reconstruction noise, not a defect:
- IGNORE slight shifts in shade, brightness or saturation. Never list these.
- REPORT what a person would see: something added, removed, moved or reshaped;
  a colour changed enough to have a different name; text altered; style
  visibly changed.

Score against the criteria:
- 1-2: no criterion met; the requested change did not happen.
- 3-5: some criteria met, some not.
- 6-7: every criterion met, but something visibly drifted.
- 8-10: every criterion met and nothing else visibly changed.

- "checks" has one entry per criterion, in the order given: the criterion
  repeated, met true or false, and a note saying what you actually see there.
- "drift" lists unrequested visible changes. Empty is correct for a good edit.
- "new_criteria" holds one new criterion per drifted item, phrased as a fact
  about the finished picture. Empty when nothing drifted.
- "satisfied" is true only when every criterion is met and nothing visibly
  drifted. An image that looks good but fails a criterion is not satisfied.
- "revised_prompt" is a full replacement prompt that fixes what failed. Target
  the criteria that came back false, change the wording that did not work
  rather than restating it louder, and keep what did. If satisfied, repeat the
  prompt that worked.

Respond with JSON only.
"""

CRITERIA_SYSTEM = """You are about to supervise an image edit. Before anything is
generated, write down what must be true of the finished picture for the
request to count as done.

You get the original image and the request.

Write 2 to 5 criteria. Each one must be a single fact a person could confirm
or refute by looking at the result, with no judgement call.

Cover what the request actually asks for, especially:
- WHERE things are, relative to other things ("the post stands between the
  wall and the driveway, not beside the wall").
- Whether things TOUCH, connect, span or attach ("the left end of the gate
  meets the post, with no gap between them"). Requests about moving or
  rearranging structures usually hinge on this, and it is the easiest thing
  to overlook because a picture can contain both objects and still be wrong.
- COUNT and EXTENT ("there is exactly one gate", "the gate reaches all the
  way across the opening").
- Anything the request says to leave alone.

Do not write criteria about colour, sharpness or style unless the request
asks for them. Do not restate the request as one vague criterion. Split it
into the separate things that must independently hold.

Respond with JSON only.
"""


REPLAN_SYSTEM = """You are correcting a prompt for {editor_name}, an instruction-following
image editor.

An earlier prompt produced the current result. A person has since reviewed the
acceptance criteria and edited them -- adding what was missing, removing what
was wrong, rewording what did not say what they meant. Write the next prompt.

{layout}

The criteria you are given are the specification. They are what the person
actually wants, and they override the original request, the earlier prompt,
and your own reading of the image. Do not argue with them or soften them.

Rules:
- Aim the prompt at the criteria that the current result fails, and keep the
  wording that already satisfies the others. You are steering, not restarting.
- Be concrete. If a criterion says two things must meet with no gap, say in
  the prompt where they meet.
- Still name what must stay unchanged, including whatever is already right.
- Describe the desired FINAL image, not the act of editing it.
- One prompt. No alternatives, no commentary outside the JSON.
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


# Rendering above this costs VRAM and time steeply, and the editors were not
# trained for it. Photos come in far larger; there is no point rendering above
# the source either, so the derived size is the smaller of the two.
MAX_SIDE_CAP = 2048


def auto_max_side(img: Image.Image, cap: int = MAX_SIDE_CAP) -> int:
    """Longest edge to render at, derived from the source.

    A fixed default silently downscales a phone photo by 4x before the editor
    ever sees it, which loses exactly the fine structure the judge is asked to
    check.
    """
    return max(min(max(img.size), cap), 256)


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
class Check:
    criterion: str
    met: bool
    note: str


@dataclass
class Verdict:
    satisfied: bool
    score: int
    checks: list[Check]
    drift: list[str]
    new_criteria: list[str]
    revised_prompt: str

    @property
    def issues(self) -> list[str]:
        """Everything wrong, for display: failed criteria first, then drift."""
        failed = [f"{c.criterion} - {c.note}" for c in self.checks if not c.met]
        return failed + list(self.drift)


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
        criteria: list[str],
    ) -> Verdict:
        raise NotImplementedError

    def criteria(self, source: Image.Image, request: str) -> list[str]:
        raise NotImplementedError

    def _verdict(self, data: dict) -> Verdict:
        """Build a Verdict, enforcing the rules in code rather than trusting.

        A model that reports a criterion unmet and then sets satisfied anyway
        is common enough that the conjunction has to be recomputed here.
        """
        checks = [
            Check(
                criterion=str(c.get("criterion", "")),
                met=bool(c.get("met")),
                note=str(c.get("note", "")),
            )
            for c in data.get("checks", [])
        ]
        drift = [str(d) for d in data.get("drift", [])]
        satisfied = bool(data.get("satisfied")) and all(c.met for c in checks) and not drift
        return Verdict(
            satisfied=satisfied,
            score=int(data.get("score", 0)),
            checks=checks,
            drift=drift,
            new_criteria=[str(c) for c in data.get("new_criteria", [])],
            revised_prompt=str(data.get("revised_prompt", "")).strip(),
        )

    def replan(
        self,
        source: Image.Image,
        current: Image.Image,
        request: str,
        criteria: list[str],
        prior_prompt: str,
        editor: "Editor",
    ) -> Plan:
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

    def critique(self, source, candidate, request, prompt, editor, criteria):
        listed = "\n".join(f"{n}. {c}" for n, c in enumerate(criteria, 1))
        data = self._chat(
            CRITIQUE_SYSTEM.format(editor_name=editor.display, layout=LAYOUT_PANELS),
            f"The user asked for:\n\n{request}\n\n"
            f"The prompt used was:\n\n{prompt}\n\n"
            f"Acceptance criteria:\n{listed}\n\n"
            "Grade the RIGHT panel against the LEFT panel, ruling on each "
            "criterion in order.",
            [composite_pair(source, candidate)],
            CRITIQUE_SCHEMA,
        )
        return self._verdict(data)

    def criteria(self, source, request, editor=None):
        data = self._chat(
            CRITERIA_SYSTEM,
            f"The request is:\n\n{request}\n\n"
            "This is the image it will be applied to. Write the criteria.",
            [source],
            CRITERIA_SCHEMA,
        )
        return [str(c).strip() for c in data.get("criteria", []) if str(c).strip()]

    def replan(self, source, current, request, criteria, prior_prompt, editor):
        data = self._chat(
            REPLAN_SYSTEM.format(editor_name=editor.display, layout=LAYOUT_PANELS),
            f"The original request was:\n\n{request}\n\n"
            f"The prompt that produced the RIGHT panel was:\n\n{prior_prompt}\n\n"
            "The criteria, as edited by the person:\n"
            + "\n".join(f"{n}. {c}" for n, c in enumerate(criteria, 1))
            + "\n\nWrite the corrected prompt.",
            [composite_pair(source, current)],
            PLAN_SCHEMA,
        )
        return Plan(prompt=data["prompt"].strip(), reasoning=data.get("reasoning", "").strip())

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

    def criteria(self, source, request, editor=None):
        data = self._msg(
            CRITERIA_SYSTEM,
            f"The request is:\n\n{request}\n\n"
            "This is the image it will be applied to. Write the criteria.",
            [source],
            CRITERIA_SCHEMA,
        )
        return [str(c).strip() for c in data.get("criteria", []) if str(c).strip()]

    def replan(self, source, current, request, criteria, prior_prompt, editor):
        listed = "\n".join(f"{n}. {c}" for n, c in enumerate(criteria, 1))
        data = self._msg(
            REPLAN_SYSTEM.format(editor_name=editor.display, layout=LAYOUT_PAIR),
            f"The original request was:\n\n{request}\n\n"
            f"The prompt that produced image 2 was:\n\n{prior_prompt}\n\n"
            f"The criteria, as edited by the person:\n{listed}\n\n"
            "Image 1 is the original. Image 2 is the current result. "
            "Write the corrected prompt.",
            [source, current],
            PLAN_SCHEMA,
        )
        return Plan(prompt=data["prompt"].strip(), reasoning=data.get("reasoning", "").strip())

    def critique(self, source, candidate, request, prompt, editor, criteria):
        listed = "\n".join(f"{n}. {c}" for n, c in enumerate(criteria, 1))
        data = self._msg(
            CRITIQUE_SYSTEM.format(editor_name=editor.display, layout=LAYOUT_PAIR),
            f"The user asked for:\n\n{request}\n\n"
            f"The prompt used was:\n\n{prompt}\n\n"
            f"Acceptance criteria:\n{listed}\n\n"
            "Image 1 is the original. Image 2 is the edit. Rule on each "
            "criterion in order.",
            [source, candidate],
            CRITIQUE_SCHEMA,
        )
        return self._verdict(data)


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
    max_side: int | None = None  # None = derive from the source image
    offload: str = "auto"
    judge: str = "local"
    judge_model: str = "qwen3.6:27b"
    claude_model: str = ClaudeJudge.MODEL
    num_ctx: int = 8192
    free_ollama: str = "all"
    prompt: str | None = None
    # Criteria are the specification. Supplied here (hand-edited in the UI, or
    # carried over from a previous run) they replace the judge's own draft.
    criteria: list[str] | None = None
    # Continuation: carry on from an earlier result instead of starting over.
    prior_prompt: str | None = None  # the prompt that produced resume_from
    resume_from: str | None = None   # image in `out` to continue from
    start_index: int = 1             # first attempt number, so files keep counting


def iterate(source: Image.Image, cfg: Settings) -> "Iterator[dict[str, Any]]":
    """Run the refine loop, yielding one event per step.

    A generator rather than a function with prints, so the CLI and the web UI
    drive the same loop instead of keeping two copies of it in step.
    """
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.max_side is None:
        cfg = replace(cfg, max_side=auto_max_side(source))

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
        "max_side": cfg.max_side,
        "source_size": list(source.size),
        "continuing": bool(cfg.resume_from),
    }

    # Criteria are the specification. Hand-edited ones arrive via cfg and are
    # taken as given; otherwise the judge drafts them from the request before
    # anything is generated, so it cannot quietly grade only what is easy.
    criteria: list[str] = list(cfg.criteria) if cfg.criteria else []
    if not criteria:
        judge.release()
        criteria = judge.criteria(source, cfg.request)
        judge.release()
    yield {"type": "criteria", "criteria": list(criteria), "edited": bool(cfg.criteria)}

    log: list[dict[str, Any]] = []
    best: tuple[int, Path] | None = None
    satisfied_at: int | None = None

    prompt = cfg.prompt
    needs_plan = prompt is None

    try:
        if cfg.resume_from and cfg.prior_prompt:
            # Carrying on from an earlier result against the edited criteria,
            # rather than throwing away the attempts already made.
            current = Image.open(outdir / cfg.resume_from).convert("RGB")
            editor.release()
            plan = judge.replan(
                source, current, cfg.request, criteria, cfg.prior_prompt, editor
            )
            judge.release()
            prompt = plan.prompt
            needs_plan = False
            yield {
                "type": "replan",
                "iteration": cfg.start_index,
                "prompt": prompt,
                "reasoning": plan.reasoning,
            }

        first = cfg.start_index
        for i in range(first, first + cfg.max_iters):
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
            verdict = judge.critique(
                source, candidate, cfg.request, prompt, editor, criteria
            )
            judge.release()

            # Every unrequested change becomes a permanent criterion, so the
            # picture cannot drift away one fix at a time, and cannot flip back
            # and forth between two wrong versions.
            added = [
                c for c in verdict.new_criteria
                if c and c.lower() not in {x.lower() for x in criteria}
            ]
            criteria.extend(added)

            yield {
                "type": "critique",
                "iteration": i,
                "score": verdict.score,
                "satisfied": verdict.satisfied,
                "checks": [
                    {"criterion": c.criterion, "met": c.met, "note": c.note}
                    for c in verdict.checks
                ],
                "drift": verdict.drift,
                "issues": verdict.issues,
                "added_criteria": added,
                "criteria": list(criteria),
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
                    "checks": [
                        {"criterion": c.criterion, "met": c.met, "note": c.note}
                        for c in verdict.checks
                    ],
                    "drift": verdict.drift,
                    "added_criteria": added,
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
            "criteria": list(criteria),
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
                        "criteria": criteria,
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
        criteria=args.criteria,
    )
    source = Image.open(args.image).convert("RGB")

    for ev in iterate(source, cfg):
        kind = ev["type"]
        if kind == "start":
            print(f"request : {ev['request']}")
            print(f"editor  : {ev['editor']}  ({ev['steps']} steps)")
            print(f"judge   : {ev['judge']}")
            print(f"output  : {ev['outdir']}\n")
        elif kind == "criteria":
            label = "criteria (edited)" if ev["edited"] else "criteria"
            print(f"{label}:")
            for n, c in enumerate(ev["criteria"], 1):
                print(f"  {n}. {c}")
            print()
        elif kind == "replan":
            print(f"attempt {ev['iteration']} prompt (steered by your criteria):")
            print(f"  {ev['prompt']}")
            if ev["reasoning"]:
                print(f"  why: {ev['reasoning']}")
            print()
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
            for c in ev["checks"]:
                mark = "PASS" if c["met"] else "FAIL"
                print(f"    [{mark}] {c['criterion']}")
                if c["note"]:
                    print(f"           {c['note']}")
            for d in ev["drift"]:
                print(f"    [drift] {d}")
            for a in ev["added_criteria"]:
                print(f"    [+criterion] {a}")
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
    g.add_argument(
        "--max-side",
        type=lambda v: None if v.lower() == "auto" else int(v),
        default=None,
        help="longest edge of the output, or 'auto' (default) to derive it "
        f"from the source, capped at {MAX_SIDE_CAP}",
    )
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
    p.add_argument(
        "--criterion",
        action="append",
        dest="criteria",
        metavar="TEXT",
        help="acceptance criterion the result must meet; repeat for several. "
        "Given any, the judge uses yours instead of drafting its own.",
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
