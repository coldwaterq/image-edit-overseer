# image-edit-overseer

This is not cost effective, but I was tired of trying to convince chatgpt to make the image I wanted.

Tell it what you want changed. An LLM writes the edit prompt, looks at what came
back, and rewrites the prompt until the result actually matches what you asked
for — instead of you doing that by hand, one disappointing render at a time.

```
python app.py        # web UI at http://127.0.0.1:7860
```

or from the command line:

```
python overseer.py photo.png "make the sky a stormy purple, keep the dog sharp"
```

Everything runs on your own GPU by default. The only thing that leaves the
machine is the optional `--judge claude` escalation.

## The UI

`python app.py` serves a page at `http://127.0.0.1:7860`. Drop in an image,
describe the edit, and each attempt appears as it finishes — the picture, the
prompt that produced it, the score, and exactly what the judge objected to.
Watching the prompt change between attempts is the useful part: you can see
which wording the editor ignored.

Runs are written to `runs/<timestamp>-<id>/` exactly as the CLI writes them,
so anything you start in the browser leaves the same images and `log.json`
behind.

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

**Judge** — any Ollama model with the `vision` capability (check with
`ollama show <model>`). Ollama is used here specifically because it loads and
unloads models on demand, which is what lets the judge and the diffusion
pipeline share one GPU:

```
ollama pull qwen3.6:27b      # ~18GB, the default
ollama pull qwen3-vl:32b     # ~24GB, vision-specialised
ollama pull qwen3-vl:8b      # ~6GB, for smaller cards
```

**If you already run a local model for something else, judge with that one.**
Two models do not fit on a 32GB card, so a second one evicts the first, and
every alternation pays a full reload. Reusing the model that is already warm
removes that cost entirely — and a general model with vision judges these
comparisons as well as a vision-specialised one. Vision training also costs
some coding ability, so a general model is the better shared choice.

**Editor** — a diffusion pipeline through diffusers:

```
pip install -r requirements.txt
```

Weights download from Hugging Face on first use — around 35GB for FLUX.2
klein 9B and 40GB for Qwen-Image-Edit. Set `HF_HOME` to a drive with room
before the first run.

**Optional** — for `--judge claude`, set `ANTHROPIC_API_KEY`.

## VRAM

The judge and the editor cannot both sit in 32GB, so the script hands the card
back and forth: before every render it tells Ollama to unload its models
(`keep_alive: 0`), and the pipeline parks its weights in system RAM before a
critique. Parking beats reloading from disk by a wide margin, so the swap costs
seconds, not minutes.

By default this frees **every** model Ollama holds, not just the judge —
`--free-ollama own` restricts it to ours. The default is aggressive on purpose.
Ollama cannot see the VRAM diffusers is about to claim, so any model left
resident pushes the render into oversubscription and Windows spills the
overflow into shared system memory across PCIe. Measured here: ~11GB spilled,
and renders that take 13 seconds on a free card had not finished one in ten
minutes.

| Editor | Weights (bf16) | Fits 32GB resident |
|---|---|---|
| FLUX.2 [klein] 9B | ~34.8GB | no — offloads |
| FLUX.2 [klein] 4B | ~24GB | yes |
| Qwen-Image-Edit-2511 | ~40GB | no — offloads |

FLUX.2 is heavier than its parameter count suggests: the 9B is 18.2GB of
transformer plus a **16.4GB language-model text encoder**. The transformer
alone would fit a 32GB card; the pair does not.

`--offload` handles that by keeping one component on the GPU at a time, which
works because each component fits individually. It defaults to `auto` and turns
itself on when the weights exceed 85% of your card — `--offload off` forces
them resident, `--offload on` forces streaming.

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
| `--offload` | `auto` | `auto`/`on`/`off`; auto trips when weights exceed the card |
| `--prompt` | — | skip the first planning call, start from your own prompt |

## Choosing an editor

**FLUX.2 [klein] 9B** is the stronger default — the highest-scoring open-weight
editor you can run on a 32GB card, though only with offloading (see VRAM above).
Note the 9B weights are released under a non-commercial license; the 4B is
Apache-2.0 and is the one that runs resident.

**Qwen-Image-Edit-2511** is often better at surgical, localised edits ("remove
the sign", "change only her jacket") and at holding a subject's identity
steady across iterations. It offloads on a 32GB card and is substantially
slower — 40 sampler steps against klein's 4.

If one editor keeps failing on a particular edit, switching to the other is
usually more productive than another five iterations of prompt rewording.

## Choosing a judge

The judge decides whether the loop converges, so it matters more than it looks.
It has to spot the specific way a render missed and then rewrite the prompt in
the idiom the editor responds to.

Any vision-capable model works and costs nothing per iteration. Pick the one
you already keep loaded — see the eviction note in Setup.

`--judge claude` uses Claude Opus 5, a meaningfully sharper critic, for roughly
a few cents per run. Worth it for an image you actually care about, or when a
local judge keeps missing the same flaw.

### Calibrating a judge

Before trusting a judge, check it against three images: one where the edit did
not happen, one that is visibly wrong, and one that is genuinely good. A judge
that scores all three the same is useless even if each verdict reads sensibly —
and that failure is easy to miss, because a harsh judge rejecting a bad edit
looks exactly like a working one. The loop can only converge if a good edit
actually passes.
