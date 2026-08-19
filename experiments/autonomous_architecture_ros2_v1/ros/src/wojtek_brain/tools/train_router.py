#!/usr/bin/env python3
"""Fine-tune the intent-router encoder on the synthetic set.

Parameterized tool (env-declared, no experiment values hard-coded).
Compare Polish-capable encoders on the SAME dataset before committing —
the design doc names HerBERT (default), mmBERT/EuroBERT and ModernBERT as
the candidates; ModernBERT is English-centric so it must earn its place on
the Polish eval split, not by reputation.

  ROUTER_DATA=router_dataset.jsonl ROUTER_OUT=router_model \
    [ROUTER_BASE=allegro/herbert-base-cased] [ROUTER_EPOCHS=4] \
    python3 train_router.py

Writes the model + id2label into ROUTER_OUT; point the router node's
`model_path` parameter there.  Prints the held-out classification report —
paste it into the PR that changes the default.
"""

import json
import os
import sys

DATA = os.environ.get("ROUTER_DATA") or sys.exit("ROUTER_DATA required (jsonl)")
OUT = os.environ.get("ROUTER_OUT") or sys.exit("ROUTER_OUT required (dir)")
BASE = os.environ.get("ROUTER_BASE", "allegro/herbert-base-cased")
EPOCHS = int(os.environ.get("ROUTER_EPOCHS", "4"))
SEED = int(os.environ.get("ROUTER_SEED", "0"))

INTENTS = ("chat", "nav", "visual", "cancel", "system")


def main():
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    rows = [json.loads(ln) for ln in open(DATA, encoding="utf-8") if ln.strip()]
    label2id = {name: i for i, name in enumerate(INTENTS)}
    ds = Dataset.from_list(
        [{"text": r["text"], "label": label2id[r["label"]]} for r in rows]
    ).train_test_split(test_size=0.15, seed=SEED, stratify_by_column=None)

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=128), batched=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        num_labels=len(INTENTS),
        id2label={i: n for n, i in label2id.items()},
        label2id=label2id,
    )

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = float((preds == p.label_ids).mean())
        per_class = {
            n: float((preds[p.label_ids == i] == i).mean())
            for n, i in label2id.items()
            if (p.label_ids == i).any()
        }
        return {"accuracy": acc, **{f"recall_{k}": v for k, v in per_class.items()}}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUT,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=32,
            learning_rate=3e-5,
            eval_strategy="epoch",
            save_strategy="no",
            seed=SEED,
            report_to=[],
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=metrics,
    )
    trainer.train()
    print(json.dumps(trainer.evaluate(), indent=2))
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)


if __name__ == "__main__":
    main()
