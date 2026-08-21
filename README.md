# image-edit-overseer

This is not cost effective, but I was tired of trying to convince chatgpt to make the image I wanted.

Tell it what you want changed. An LLM writes the edit prompt, looks at what came
back, and rewrites the prompt until the result actually matches what you asked
for — instead of you doing that by hand, one disappointing render at a time.

```
python overseer.py photo.png "make the sky a stormy purple, keep the dog sharp"
```

Everything runs on your own GPU by default. The only thing that leaves the
machine is the optional `--judge claude` escalation.

## How it works

```
  your request ──▶ judge writes a prompt ──▶ editor renders ──▶ judge grades it
                            ▲                                          │
                            └─────────── revised prompt ◀──────────────┘
```

The judge is a vision model. It sees the original and the candidate side by
side, decides whether a careful person who asked for this would accept it, and
if not, says what is wrong and rewrites the prompt to fix it. The loop stops
when the judge is satisfied or `--max-iters` runs out, and the best-scoring
render is saved as `final.png`.

Each run writes every intermediate image plus a `log.json` with the prompt,
score, and issues for each iteration, so you can see where it went wrong.

## Setup

**Judge** — a vision model in Ollama. Ollama is used here specifically because
it loads and unloads models on demand, which is what lets the judge and the
diffusion pipeline share one GPU:

```
ollama pull qwen3-vl:32b     # ~21GB, the strongest option that fits 32GB
ollama pull qwen3-vl:8b      # ~6GB, use with --judge-model qwen3-vl:8b
```

**Editor** — a diffusion pipeline through diffusers:

```
pip install -r requirements.txt
```

Weights download from Hugging Face on first use. Set `HF_HOME` first if you
don't want ~20-40GB landing on your system drive.

**Optional** — for `--judge claude`, set `ANTHROPIC_API_KEY`.

## VRAM

The judge and the editor cannot both sit in 32GB, so the script hands the card
back and forth: Ollama unloads the judge (`keep_alive: 0`) before a render, and
the pipeline parks its weights in system RAM before a critique. Parking beats
reloading from disk by a wide margin, so the swap costs seconds, not minutes.

| Combination | Peak VRAM | Fits 32GB |
|---|---|---|
| FLUX.2 [klein] 9B, swapped with any judge | ~29GB | yes |
| FLUX.2 [klein] 4B, swapped with any judge | ~10GB | comfortably |
| Qwen-Image-Edit-2511 (20B bf16) | ~40GB | only with `--offload` |

`--offload` streams weights from system RAM instead. Much slower per render,
but it drops peak VRAM enough for the 20B editor.

## Options

| Flag | Default | Notes |
|---|---|---|
| `--editor` | `flux` | `flux` or `qwen` |
| `--flux-size` | `9B` | `4B` is Apache-2.0; **9B is non-commercial** |
| `--judge` | `local` | `claude` escalates the critic to Claude Opus 5 |
| `--judge-model` | `qwen3-vl:32b` | any vision model in your Ollama library |
| `--max-iters` | `5` | |
| `--steps` | editor default | 4 for FLUX klein, 40 for Qwen |
| `--max-side` | `1024` | longest edge of the output |
| `--offload` | off | lower peak VRAM, slower |
| `--prompt` | — | skip the first planning call, start from your own prompt |

## Choosing an editor

**FLUX.2 [klein] 9B** is the stronger default — highest-scoring open-weight
editor that fits a 32GB card. Note the 9B weights are released under a
non-commercial license; the 4B is Apache-2.0.

**Qwen-Image-Edit-2511** is often better at surgical, localised edits ("remove
the sign", "change only her jacket") and at holding a subject's identity
steady across iterations. It needs `--offload` on a 32GB card and is
substantially slower.

If one editor keeps failing on a particular edit, switching to the other is
usually more productive than another five iterations of prompt rewording.

## Choosing a judge

The judge decides whether the loop converges, so it matters more than it looks.
It has to spot the specific way a render missed and then rewrite the prompt in
the idiom the editor responds to.

`qwen3-vl:32b` is the strongest local option and costs nothing per iteration.
`--judge claude` uses Claude Opus 5, which is a meaningfully sharper critic —
worth it for an image you actually care about, and roughly a few cents for a
full run.
