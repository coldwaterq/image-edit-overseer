#!/usr/bin/env python3
"""Fast checks that exercise the plumbing without touching the GPU.

`python -m py_compile` proves the file parses, which is not the same as the
names existing: a slice-based edit once deleted `composite_pair` outright and
the failure only surfaced after a six-minute render. Everything here runs in
under a second, so run it before starting anything long.

    python smoke_test.py
"""

from __future__ import annotations

import sys

from PIL import Image

import overseer as o

FAILS: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ok    {name}")
    except Exception as exc:  # noqa: BLE001 - a smoke test wants every failure
        FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")


def main() -> int:
    a = Image.new("RGB", (640, 480), (30, 60, 90))
    b = Image.new("RGB", (640, 480), (90, 60, 30))

    print("sizing")
    check("fit_to_area caps at one megapixel", lambda: _eq(
        o.fit_to_area((4080, 3072))[0] * o.fit_to_area((4080, 3072))[1] <= o.MAX_RENDER_AREA, True))
    check("fit_to_area never upscales", lambda: _eq(o.fit_to_area((320, 240)), (320, 224)))
    check("fit_dimensions snaps to a multiple", lambda: _eq(o.fit_dimensions(a, 512)[0] % 32, 0))
    check("auto_steps beats the 4-step default", lambda: _eq(
        o.FluxKleinEditor.auto_steps(1024) >= 16, True))

    print("judge plumbing")
    check("composite_pair builds a labelled canvas", lambda: _eq(
        o.composite_pair(a, b).size[0] > a.size[0], True))
    check("b64_png encodes", lambda: _eq(len(o.b64_png(a)) > 100, True))
    check("extract_json survives a fenced blob", lambda: _eq(
        o.extract_json('here you go\n```json\n{"a": 1}\n```'), {"a": 1}))

    print("verdict rules (the ones that must not be left to the model)")
    judge = o.Judge()
    claimed = {
        "differences": [],
        "checks": [{"criterion": "post moved", "met": True, "note": "clearly moved"}],
        "drift": [], "new_criteria": [], "score": 10,
        "satisfied": True, "revised_prompt": "x",
    }
    check("an unchanged image cannot be satisfied", lambda: _eq(
        judge._verdict(claimed).satisfied, False))
    check("an unchanged image scores at most 1", lambda: _eq(
        judge._verdict(claimed).score <= 1, True))
    check("a real difference can pass", lambda: _eq(
        judge._verdict(dict(claimed, differences=["the post moved left"])).satisfied, True))
    check("a failed criterion blocks satisfaction", lambda: _eq(
        judge._verdict({
            "differences": ["something moved"],
            "checks": [{"criterion": "c", "met": False, "note": "n"}],
            "drift": [], "new_criteria": [], "score": 9,
            "satisfied": True, "revised_prompt": "x",
        }).satisfied, False))

    print("config")
    check("Settings defaults are sane", lambda: _eq(
        (o.Settings(request="r", out="o").max_side, o.Settings(request="r", out="o").free_ollama),
        (None, "all")))
    check("build_editor accepts Settings", lambda: _eq(
        o.build_editor(o.Settings(request="r", out="o", max_side=1024)).key, "flux"))
    check("prompt templates format", lambda: _eq(
        "{" not in o.CRITIQUE_SYSTEM.format(editor_name="E", layout=o.LAYOUT_PANELS)[:200], True))

    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        return 1
    print("all good")
    return 0


def _eq(got, want):
    if got != want:
        raise AssertionError(f"got {got!r}, wanted {want!r}")


if __name__ == "__main__":
    raise SystemExit(main())
