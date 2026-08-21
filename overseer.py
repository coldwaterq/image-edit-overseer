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
from typing import Any

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

You get two images. The FIRST is the original. The SECOND is the edit that was
produced. You also get the user's request and the prompt that was used.

Judge only whether the SECOND image satisfies the user's request while keeping
everything the user did not ask to change.

Be specific and be hard to please:
- "satisfied" is true only if a careful person who asked for this would accept
  it without further comment.
- "issues" lists what is actually wrong, each one concrete and visible. If the
  edit did not happen at all, say so. If something drifted that should not
  have, say what.
- "score" is 1-10 for how well the request was met.
- "revised_prompt" is a full replacement prompt that fixes the issues. Change
  the wording that failed rather than restating it louder, and keep the parts
  that worked. If satisfied is true, repeat the prompt that worked.

Respond with JSON only.
"""

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


def fit_dimensions(img: Image.Image, max_side: int, multiple: int = 32) -> tuple[int, int]:
    """Largest size within max_side that keeps aspect and snaps to `multiple`."""
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    w, h = max(int(w * scale), multiple), max(int(h * scale), multiple)
    return (w // multiple) * multiple, (h // multiple) * multiple


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
        resp = self._requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=900)
        resp.raise_for_status()
        return extract_json(resp.json()["message"]["content"])

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
            CRITIQUE_SYSTEM.format(editor_name=editor.display),
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
            CRITIQUE_SYSTEM.format(editor_name=editor.display),
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

class Editor:
    """A diffusion image editor that can hand its VRAM back between turns."""

    key: str
    display: str
    repo: str
    default_steps: int

    def __init__(self, steps: int | None = None, max_side: int = 1024, offload: bool = False):
        self.steps = steps or self.default_steps
        self.max_side = max_side
        self.offload = offload
        self.pipe = None
        self._on_gpu = False

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

    def __init__(self, size: str = "9B", **kw):
        self.repo = f"black-forest-labs/FLUX.2-klein-{size}"
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


def build_editor(args) -> Editor:
    if args.editor == "flux":
        return FluxKleinEditor(
            size=args.flux_size, steps=args.steps, max_side=args.max_side, offload=args.offload
        )
    return QwenEditor(steps=args.steps, max_side=args.max_side, offload=args.offload)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def run(args) -> int:
    source = Image.open(args.image).convert("RGB")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    editor = build_editor(args)
    judge: Judge = (
        ClaudeJudge(args.claude_model)
        if args.judge == "claude"
        else OllamaJudge(args.judge_model, num_ctx=args.num_ctx)
    )

    print(f"request : {args.request}")
    print(f"editor  : {editor.display}  ({editor.steps} steps)")
    print(f"judge   : {judge.name}")
    print(f"output  : {outdir}\n")

    log: list[dict[str, Any]] = []
    best: tuple[int, Path] | None = None

    # The judge holds VRAM while it thinks, so it must let go before the
    # pipeline loads, and vice versa. Neither fits alongside the other.
    prompt = args.prompt
    needs_plan = prompt is None
    if prompt:
        print(f"iteration 1 prompt (yours):\n  {prompt}\n")

    for i in range(1, args.max_iters + 1):
        if needs_plan:
            editor.release()
            plan = judge.plan(source, args.request, editor)
            judge.release()
            prompt = plan.prompt
            print(f"iteration {i} prompt:\n  {prompt}")
            if plan.reasoning:
                print(f"  why: {plan.reasoning}")
            print()
        needs_plan = False

        editor.acquire()
        t0 = time.time()
        candidate = editor.edit(source, prompt, args.seed + i - 1)
        elapsed = time.time() - t0
        path = outdir / f"iter{i:02d}.png"
        candidate.save(path)
        print(f"  rendered in {elapsed:.1f}s -> {path}")

        editor.release()
        verdict = judge.critique(source, candidate, args.request, prompt, editor)
        judge.release()

        print(f"  score {verdict.score}/10  satisfied={verdict.satisfied}")
        for issue in verdict.issues:
            print(f"    - {issue}")

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
            final = outdir / "final.png"
            candidate.save(final)
            print(f"\ndone in {i} iteration(s) -> {final}")
            break

        prompt = verdict.revised_prompt
        print(f"\n  revised prompt:\n  {prompt}\n")
    else:
        score, path = best  # type: ignore[misc]
        final = outdir / "final.png"
        Image.open(path).save(final)
        print(
            f"\nstopped after {args.max_iters} iterations without a pass.\n"
            f"best was {path.name} at {score}/10 -> {final}"
        )

    (outdir / "log.json").write_text(
        json.dumps(
            {
                "request": args.request,
                "source": str(Path(args.image).resolve()),
                "editor": editor.repo,
                "judge": judge.name,
                "iterations": log,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    editor.release()
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
        action="store_true",
        help="stream weights from RAM (slower, much lower peak VRAM)",
    )

    j = p.add_argument_group("judge")
    j.add_argument("--judge", choices=["local", "claude"], default="local")
    j.add_argument("--judge-model", default="qwen3-vl:32b", help="Ollama model for --judge local")
    j.add_argument("--claude-model", default=ClaudeJudge.MODEL)
    j.add_argument("--num-ctx", type=int, default=8192)

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
