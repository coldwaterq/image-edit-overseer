#!/usr/bin/env python3
"""Turn finished runs into training examples for a smaller judge.

The idea: a big judge (Claude, or qwen3.6:27b) does a job well; every call it
made is a supervised example of the exact task a small judge keeps failing.
Collect enough of them and the small model can learn the shape of the answer.

That is the whole reason qwen3-vl:8b is unusable today -- it reasons in circles
and never emits the verdict. It is not being asked to be clever, it is being
asked to look, decide, and fill in seven fields. That is learnable.

    python distill.py export runs/20260821-155823-4af391
    python distill.py export --all --judge claude
    python distill.py stats

Examples land in `training/dataset.jsonl`, one JSON object per line:

    {"images": [...], "system": "...", "user": "...", "assistant": "{...}"}

`assistant` is exactly what the model should have produced -- the JSON, with no
reasoning in front of it. That is deliberate: the failure being trained out is
reasoning that never terminates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

import overseer as o

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "training"
DATASET = TRAIN_DIR / "dataset.jsonl"
IMAGE_DIR = TRAIN_DIR / "images"


def _events(run: Path) -> list[dict[str, Any]]:
    path = run / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _save(img: Image.Image, name: str) -> str:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / name
    if not path.exists():
        img.save(path)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def examples_from(run: Path) -> Iterator[dict[str, Any]]:
    """Rebuild each judge call the run made, paired with the answer it gave.

    The prompts are not stored in the log, but they are deterministic: the same
    templates, the same request, the same images. So they can be reconstructed
    exactly rather than approximated.
    """
    events = _events(run)
    if not events:
        return
    start = next((e for e in events if e["type"] == "start"), None)
    if start is None:
        return
    request = start.get("request", "")
    editor_name = start.get("editor", "FLUX.2 [klein] 9B")
    source_path = run / "source.png"
    if not source_path.exists():
        return
    source = Image.open(source_path).convert("RGB")
    tag = run.name

    # 1. deciding what "done" means, from the request and the original image
    crit = next((e for e in events if e["type"] == "criteria"), None)
    if crit and not crit.get("edited"):
        yield {
            "kind": "criteria",
            "images": [_save(source, f"{tag}-source.png")],
            "system": o.CRITERIA_SYSTEM,
            "user": f"The request is:\n\n{request}\n\n"
                    "This is the image it will be applied to. Write the criteria.",
            "assistant": json.dumps({"criteria": crit["criteria"]}, indent=2),
        }

    # 2. each verdict, against the pair the judge actually saw
    criteria = list(crit["criteria"]) if crit else []
    prompts: dict[int, str] = {}
    for e in events:
        if e["type"] in ("plan", "replan", "revised"):
            prompts[e["iteration"] + (1 if e["type"] == "revised" else 0)] = e["prompt"]
        elif e["type"] == "render":
            prompts.setdefault(e["iteration"], e.get("prompt", ""))
        elif e["type"] == "critique":
            i = e["iteration"]
            cand_path = run / f"iter{i:02d}.png"
            if not cand_path.exists():
                continue
            cand = Image.open(cand_path).convert("RGB")
            listed = "\n".join(f"{n}. {c}" for n, c in enumerate(criteria, 1))
            answer = {
                "differences": e.get("differences", []),
                "checks": e.get("checks", []),
                "drift": e.get("drift", []),
                "new_criteria": e.get("added_criteria", []),
                "score": e["score"],
                "satisfied": e["satisfied"],
                "revised_prompt": prompts.get(i + 1, ""),
            }
            yield {
                "kind": "critique",
                "images": [_save(o.composite_pair(source, cand), f"{tag}-pair{i:02d}.png")],
                "system": o.CRITIQUE_SYSTEM.format(
                    editor_name=editor_name, layout=o.LAYOUT_PANELS
                ),
                "user": f"The user asked for:\n\n{request}\n\n"
                        f"The prompt used was:\n\n{prompts.get(i, '')}\n\n"
                        f"Acceptance criteria:\n{listed}\n\n"
                        "Grade the RIGHT panel against the LEFT panel, ruling on each "
                        "criterion in order.",
                "assistant": json.dumps(answer, indent=2),
            }
            criteria = e.get("criteria", criteria)


def export(runs: list[Path], only_judge: str | None = None) -> int:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    if DATASET.exists():
        for line in DATASET.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(hash(line.strip()))

    written = 0
    with DATASET.open("a", encoding="utf-8") as fh:
        for run in runs:
            start = next((e for e in _events(run) if e["type"] == "start"), None)
            if start is None:
                continue
            judge = start.get("judge", "")
            if only_judge and only_judge not in judge:
                continue
            for ex in examples_from(run):
                ex["source_run"] = run.name
                ex["source_judge"] = judge
                line = json.dumps(ex)
                if hash(line) in seen:
                    continue
                fh.write(line + "\n")
                seen.add(hash(line))
                written += 1
            print(f"  {run.name}  {judge}")
    return written


def stats() -> None:
    if not DATASET.exists():
        print("no dataset yet")
        return
    kinds: dict[str, int] = {}
    judges: dict[str, int] = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        j = d.get("source_judge", "?")
        judges[j] = judges.get(j, 0) + 1
    total = sum(kinds.values())
    print(f"{DATASET}: {total} examples")
    for k, v in sorted(kinds.items()):
        print(f"  {k:10s} {v}")
    print("by teacher:")
    for k, v in sorted(judges.items(), key=lambda x: -x[1]):
        print(f"  {k:26s} {v}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="turn runs into training examples")
    e.add_argument("runs", nargs="*", help="run directories")
    e.add_argument("--all", action="store_true", help="every run under runs/")
    e.add_argument(
        "--judge",
        default="anthropic",
        help="only export runs judged by this teacher (default: anthropic). "
        "The student learns whatever taught it, so mixing in a weaker judge "
        "trains toward the weaker judge.",
    )
    sub.add_parser("stats", help="what is in the dataset")

    args = p.parse_args()
    if args.cmd == "stats":
        stats()
        return 0

    runs = [Path(r) for r in args.runs]
    if args.all:
        runs = sorted(d for d in (ROOT / "runs").iterdir() if d.is_dir())
    if not runs:
        p.error("give run directories or --all")
    n = export(runs, args.judge)
    print(f"\n{n} new example(s) -> {DATASET}")
    stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
