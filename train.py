#!/usr/bin/env python3
"""QLoRA fine-tune of the small judge on a stronger judge's answers.

Qwen3-VL-8B-Instruct already produces well-formed verdicts -- it just agrees
with everything. On the calibration ladder it returned 10/10 "satisfied" for an
unchanged image, a partial edit and a good edit alike, and invented differences
to justify the unchanged one. Format is not the problem; discrimination is.

Discrimination is what examples teach. Each training example is one call a
teacher made, paired with the answer it gave, so the student learns what "no
change" and "criterion failed" actually look like.

    python train.py                       # train on training/dataset.jsonl
    python train.py --epochs 4 --out training/adapter-v2

Then use it:

    python overseer.py photo.png "..." --judge local8b --adapter training/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATASET = ROOT / "training" / "dataset.jsonl"
DEFAULT_OUT = ROOT / "training" / "adapter"
BASE = "Qwen/Qwen3-VL-8B-Instruct"


def load_examples(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"no dataset at {path}.\n"
            "Run a judged edit with Claude, then: python distill.py export --all"
        )
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("dataset is empty")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=1024 * 768,
                    help="cap on image tokens; activation memory is driven by this, "
                         "not by parameter count")
    args = ap.parse_args()

    rows = load_examples(args.dataset)
    print(f"{len(rows)} examples from {args.dataset}")
    teachers = {}
    for r in rows:
        teachers[r.get("source_judge", "?")] = teachers.get(r.get("source_judge", "?"), 0) + 1
    for k, v in teachers.items():
        print(f"  teacher {k}: {v}")
    if len(rows) < 20:
        print("\n  WARNING: under 20 examples. LoRA will memorise these rather than\n"
              "  generalise. Collect more runs before trusting the result.\n")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration, Trainer, TrainingArguments)

    proc = AutoProcessor.from_pretrained(BASE, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # language side only: the vision tower already sees the picture fine,
        # what it lacks is the judgement about what it sees.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    def build(row):
        imgs = [Image.open(ROOT / p).convert("RGB") for p in row["images"]]
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": row["user"]})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": row["assistant"]}]},
        ]
        enc = proc.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
        )
        # Loss on the answer only. Training on the prompt teaches it to recite
        # the criteria back, which is not the skill being taught.
        prompt_only = proc.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        cut = prompt_only["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :cut] = -100
        out = {k: v[0] for k, v in enc.items()}
        out["labels"] = labels[0]
        return out

    ds = Dataset.from_list([build(r) for r in rows])

    def collate(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) if batch[0][k].dim() == 0
                else torch.nn.utils.rnn.pad_sequence(
                    [b[k] for b in batch], batch_first=True,
                    padding_value=-100 if k == "labels" else 0)
                for k in keys}

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,     # image tokens are the memory cost
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            optim="paged_adamw_8bit",
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.out))
    proc.save_pretrained(str(args.out))
    print(f"\nadapter -> {args.out}")
    print(f"try it:  python overseer.py IMAGE 'REQUEST' --judge local8b --adapter {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
