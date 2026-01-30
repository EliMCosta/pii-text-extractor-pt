"""
Fine-tune a BERT-like encoder for PII extraction via token classification.

Dataset (JSONL) schema per line:
  {"text": <str>, "entities": [{"type": <str>, "value": <str>}, ...]}

The model is trained as NER with labels:
  O, B-<TYPE>, I-<TYPE> for each allowed PII type.

Default base model: neuralmind/bert-base-portuguese-cased
Default dataset path: data/generated/finetune_multilabel.jsonl
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _add_repo_root_to_syspath() -> Path:
    """
    Ensure the repository root is importable even under multi-process launchers.

    `accelerate launch ... training/script.py` may run workers with a working directory
    that is not the repo root, so relying on CWD for `import ner_labels` is fragile.
    """

    cur = Path(__file__).resolve().parent
    for _ in range(20):
        if (cur / "ner_labels.py").is_file():
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(
        "Could not locate repo root containing ner_labels.py (needed for stable label space)."
    )


_REPO_ROOT = _add_repo_root_to_syspath()

# Keep label space consistent across the repo (generation/preprocessing/inference).
from ner_labels import ALLOWED_ENTITY_TYPES


def _parse_semver_triplet(v: str) -> tuple[int, int, int]:
    # Best-effort semver parsing without extra deps.
    # Examples: "1.2.3", "1.2.3.post1", "1.2.3rc1"
    core = v.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    out: list[int] = []
    for p in parts[:3]:
        n = ""
        for ch in p:
            if ch.isdigit():
                n += ch
            else:
                break
        out.append(int(n or "0"))
    while len(out) < 3:
        out.append(0)
    return (out[0], out[1], out[2])


def _require_accelerate() -> None:
    try:
        import accelerate  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'accelerate' required by transformers.Trainer. "
            "Install the project dependencies (e.g. `uv pip install -r requirements.txt`)."
        ) from e

    try:
        from importlib.metadata import version

        v = version("accelerate")
    except Exception:
        return

    if _parse_semver_triplet(v) < (0, 26, 0):
        raise ImportError(
            f"'accelerate>={0}.{26}.{0}' is required by transformers.Trainer, but found accelerate=={v}. "
            "Upgrade dependencies (e.g. `uv pip install -r requirements.txt --upgrade`)."
        )


from utils.cuda_env import maybe_reexec_without_system_cuda, maybe_init_distributed_with_device_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a BERT-like model for PII extraction (token classification)."
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="neuralmind/bert-base-portuguese-cased",
        help="Base model checkpoint (HF name or local path).",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default="data/generated/finetune_multilabel.jsonl",
        help='Path to JSONL. Must contain "text" and "entities".',
    )
    p.add_argument(
        "--text_column",
        type=str,
        default="text",
        help='Text column name (default: "text").',
    )
    p.add_argument(
        "--entities_column",
        type=str,
        default="entities",
        help='Entities column name (default: "entities").',
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/finetune-multilabel",
        help="Where to write checkpoints and the final model.",
    )
    p.add_argument(
        "--best_checkpoint_dir",
        type=str,
        default="auto",
        help=(
            'Directory where the best checkpoint will be copied after training. '
            'Use "auto" to keep it at "<output_dir>/best". Use "none" to disable.'
        ),
    )

    p.add_argument("--validation_split", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=-1, help="If > 0, override num_train_epochs.")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="none",
        help='Path to a checkpoint to resume from, or "none".',
    )

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


@dataclass(frozen=True)
class PiiSpan:
    start: int  # char start (inclusive)
    end: int  # char end (exclusive)
    pii_type: str


def _iter_literal_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    # Literal occurrences (including punctuation/spaces exactly). Non-overlapping by default.
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        i = text.find(value, start)
        if i < 0:
            break
        out.append((i, i + len(value)))
        start = i + len(value)
    return out


def _build_spans(text: str, pii_candidates: list[dict[str, Any]], *, row_index: int) -> list[PiiSpan]:
    allowed = set(ALLOWED_ENTITY_TYPES)
    spans: list[PiiSpan] = []

    for j, it in enumerate(pii_candidates):
        if not isinstance(it, dict):
            raise ValueError(f"row={row_index}: entities[{j}] must be an object")
        if set(it.keys()) != {"type", "value"}:
            raise ValueError(
                f"row={row_index}: entities[{j}] invalid keys {sorted(it.keys())}; expected ['type','value']"
            )
        t = it.get("type")
        v = it.get("value")
        if not isinstance(t, str) or t not in allowed:
            raise ValueError(f"row={row_index}: entities[{j}].type invalid: {t!r}")
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"row={row_index}: entities[{j}].value invalid/empty")
        if v not in text:
            raise ValueError(f"row={row_index}: entities[{j}].value not found in text: {v!r}")

        for (a, b) in _iter_literal_occurrences(text, v):
            spans.append(PiiSpan(start=a, end=b, pii_type=t))

    # Prefer longer spans first to minimize conflicts (if any).
    spans.sort(key=lambda s: (-(s.end - s.start), s.start, s.end, s.pii_type))

    # Fail fast on overlaps with different types.
    occupied: list[tuple[int, int, str]] = []
    for s in spans:
        for (a, b, t) in occupied:
            if s.start < b and s.end > a and t != s.pii_type:
                raise ValueError(
                    f"row={row_index}: overlapping spans with different types: "
                    f"{(a, b, t)} vs {(s.start, s.end, s.pii_type)}"
                )
        occupied.append((s.start, s.end, s.pii_type))

    # Deduplicate exact duplicates
    uniq: list[PiiSpan] = []
    seen = set()
    for s in spans:
        k = (s.start, s.end, s.pii_type)
        if k not in seen:
            uniq.append(s)
            seen.add(k)
    uniq.sort(key=lambda s: (s.start, s.end))
    return uniq


def _token_labels_from_spans(
    *,
    offsets: list[tuple[int, int]],
    spans: list[PiiSpan],
    label2id: dict[str, int],
    row_index: int,
) -> list[int]:
    o_id = label2id["O"]
    labels = [o_id for _ in offsets]

    # For special tokens fast tokenizers typically return (0, 0).
    special = [a == 0 and b == 0 for (a, b) in offsets]
    for i, is_special in enumerate(special):
        if is_special:
            labels[i] = -100

    # BIO token classification cannot represent overlapping entities.
    # Enforce "each token has at most one label" by selecting a non-overlapping subset of spans
    # in TOKEN SPACE (more robust than char-only overlap checks).
    span_tokens: list[tuple[PiiSpan, list[int]]] = []
    for span in spans:
        token_idxs: list[int] = []
        for i, (a, b) in enumerate(offsets):
            if labels[i] == -100:
                continue
            if a < span.end and b > span.start:
                token_idxs.append(i)
        if not token_idxs:
            raise ValueError(
                f"row={row_index}: span {span} did not align to any token (offset mapping issue)"
            )
        span_tokens.append((span, token_idxs))

    # Prefer longer spans (more tokens), then longer char length, then stable tie-break.
    span_tokens.sort(
        key=lambda st: (
            -len(st[1]),
            -(st[0].end - st[0].start),
            st[0].start,
            st[0].end,
            st[0].pii_type,
        )
    )
    chosen: list[tuple[PiiSpan, list[int]]] = []
    used_tokens: set[int] = set()
    for span, token_idxs in span_tokens:
        if any(ti in used_tokens for ti in token_idxs):
            continue
        chosen.append((span, token_idxs))
        used_tokens.update(token_idxs)

    # Now assign labels for chosen spans only.
    for span, token_idxs in chosen:
        for k, ti in enumerate(token_idxs):
            tag = ("B-" if k == 0 else "I-") + span.pii_type
            labels[ti] = label2id[tag]

    return labels


def _maybe_copy_best_checkpoint(*, trainer, best_checkpoint_dir: str) -> None:
    """
    Sync the best checkpoint directory at the end of training.

    In distributed runs (accelerate/torchrun), only rank 0 must touch the filesystem
    destination to avoid races between processes (rmtree/copytree collisions).
    """
    if best_checkpoint_dir in ("none", "", None):
        return

    # In distributed setups, only perform filesystem operations on world process zero.
    is_wp0 = True
    if hasattr(trainer, "is_world_process_zero"):
        try:
            is_wp0 = bool(trainer.is_world_process_zero())
        except Exception:
            is_wp0 = True
    if not is_wp0:
        return

    best_ckpt = trainer.state.best_model_checkpoint
    if not best_ckpt:
        return

    dst = Path(best_checkpoint_dir)
    src = Path(best_ckpt)
    if not src.exists():
        return

    parent = str(dst.parent) if str(dst.parent) else "."
    os.makedirs(parent, exist_ok=True)

    import stat
    import time
    from uuid import uuid4

    def _rmtree_strict(path: str, attempts: int = 6, delay_s: float = 0.2) -> None:
        def _onerror(func, p, exc_info):
            # Try to make the path writable then retry the failing operation.
            try:
                mode = os.stat(p).st_mode
                if stat.S_ISDIR(mode):
                    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                else:
                    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
            func(p)

        last_exc: BaseException | None = None
        for i in range(attempts):
            try:
                shutil.rmtree(path, onerror=_onerror)
                return
            except FileNotFoundError:
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(delay_s * (i + 1))
        raise RuntimeError(f"Failed to remove directory: {path}") from last_exc

    # Copy into a unique tmp dir then atomically swap into place.
    tmp_dir = os.path.join(parent, dst.name + f".tmp.{uuid4().hex}")
    old_dir: str | None = None
    try:
        if os.path.exists(tmp_dir):
            _rmtree_strict(tmp_dir)
        shutil.copytree(str(src), tmp_dir)

        if dst.exists():
            old_dir = os.path.join(parent, dst.name + f".old.{uuid4().hex}")
            os.replace(str(dst), old_dir)
        os.replace(tmp_dir, str(dst))
        tmp_dir = ""  # renamed away
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            _rmtree_strict(tmp_dir)
        if old_dir and os.path.exists(old_dir):
            _rmtree_strict(old_dir)


def main() -> None:
    maybe_reexec_without_system_cuda()
    maybe_init_distributed_with_device_id()
    args = _parse_args()
    _require_accelerate()

    from datasets import load_dataset
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    # Fail fast: if multiple GPUs are available, require a distributed launcher so all GPUs are used.
    # Otherwise transformers/torch may fall back to DataParallel and emit "gather scalar" warnings.
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cuda_devices >= 2 and world_size == 1:
        raise RuntimeError(
            "Detected multiple GPUs, but the script is running with WORLD_SIZE=1.\n"
            "To use all GPUs, launch with one process per GPU, e.g.:\n"
            "  torchrun --standalone --nproc_per_node=2 training/finetune_pii_token_classification.py\n"
            "or:\n"
            "  accelerate launch --num_processes 2 training/finetune_pii_token_classification.py\n"
            "\n"
            "If you intentionally want to use a single GPU, set CUDA_VISIBLE_DEVICES=0 (or another single id)."
        )

    if args.validation_split < 0.0 or args.validation_split >= 1.0:
        raise ValueError("--validation_split must satisfy 0 <= v < 1")
    if args.max_length < 16:
        raise ValueError("--max_length too small")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one: --bf16 or --fp16")

    set_seed(args.seed)

    raw = load_dataset("json", data_files=args.dataset_path, split="train")
    if args.text_column not in raw.column_names:
        raise ValueError(f"Missing text column {args.text_column!r} in dataset columns={raw.column_names}")
    if args.entities_column not in raw.column_names:
        if "pii_candidate" in raw.column_names:
            raise ValueError(
                f"Missing entities column {args.entities_column!r} in dataset (found legacy "
                '"pii_candidate"; please regenerate/migrate your JSONL).'
            )
        raise ValueError(
            f"Missing entities column {args.entities_column!r} in dataset columns={raw.column_names}"
        )

    if args.validation_split > 0.0:
        split = raw.train_test_split(test_size=args.validation_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
        do_eval = True
    else:
        train_ds = raw
        eval_ds = None
        do_eval = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("This script requires a fast tokenizer (offset_mapping).")

    label_list: list[str] = ["O"]
    for t in ALLOWED_ENTITY_TYPES:
        label_list.append(f"B-{t}")
        label_list.append(f"I-{t}")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    def preprocess(ex: dict[str, Any], idx: int) -> dict[str, Any]:
        text = ex.get(args.text_column)
        pii = ex.get(args.entities_column)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row={idx}: invalid text")
        if not isinstance(pii, list):
            raise ValueError(f"row={idx}: {args.entities_column} must be a list")

        spans = _build_spans(text, pii, row_index=idx)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets_raw = enc.pop("offset_mapping")
        offsets: list[tuple[int, int]] = [(int(a), int(b)) for (a, b) in offsets_raw]
        enc["labels"] = _token_labels_from_spans(
            offsets=offsets, spans=spans, label2id=label2id, row_index=idx
        )
        return enc

    remove_cols = list(train_ds.column_names)
    train_ds = train_ds.map(
        preprocess,
        with_indices=True,
        remove_columns=remove_cols,
        desc="Tokenizing + aligning token labels",
    )
    if do_eval and eval_ds is not None:
        eval_ds = eval_ds.map(
            preprocess,
            with_indices=True,
            remove_columns=remove_cols,
            desc="Tokenizing + aligning token labels (eval)",
        )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )

    o_id = label2id["O"]

    def compute_metrics(eval_pred):
        import numpy as np

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        tp = 0
        pred_pos = 0
        true_pos = 0
        for p_seq, y_seq in zip(preds, labels):
            for p, y in zip(p_seq, y_seq):
                if int(y) == -100:
                    continue
                p = int(p)
                y = int(y)
                if p != o_id:
                    pred_pos += 1
                if y != o_id:
                    true_pos += 1
                if p != o_id and y != o_id and p == y:
                    tp += 1

        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / true_pos if true_pos else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    best_dir = args.best_checkpoint_dir
    if best_dir == "auto":
        best_dir = str(Path(args.output_dir) / "best")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=do_eval,
        eval_strategy="steps" if do_eval else "no",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        ddp_find_unused_parameters=False,
        report_to=[],
        load_best_model_at_end=do_eval,
        metric_for_best_model="f1" if do_eval else None,
        greater_is_better=True if do_eval else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if do_eval else None,
    )

    resume = None if args.resume_from_checkpoint == "none" else args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if do_eval:
        _maybe_copy_best_checkpoint(trainer=trainer, best_checkpoint_dir=best_dir)

    # Clean shutdown for distributed runs (avoids NCCL resource leak warnings).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

