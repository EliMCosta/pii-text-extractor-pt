"""
Inference / evaluation script for the PII token-classification model.

This project fine-tunes a DistilBERT-like encoder as NER with IOB labels:
  O, B-<TYPE>, I-<TYPE>

This script supports:
  - infer: run inference on a raw text (with inference-aligned chunking)
  - eval:  run token-level precision/recall/F1 on a JSONL dataset

Model default:
  outputs/finetune-multilabel-multi/best
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _add_repo_root_to_syspath() -> Path:
    """
    Ensure the repository root is importable even when launched from elsewhere.
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
    raise RuntimeError("Could not locate repo root containing ner_labels.py.")


_REPO_ROOT = _add_repo_root_to_syspath()

from utils.cuda_env import maybe_reexec_without_system_cuda

maybe_reexec_without_system_cuda()

from ner_labels import ALLOWED_ENTITY_TYPES, PII_TYPES

from data_preprocessing.chunking import build_chunks

from inference import (
    PiiSpan,
    PiiSpanScored,
    spans_from_token_predictions,
    spans_from_token_predictions_scored,
    filter_scored_spans,
    merge_spans,
    merge_and_resolve_scored_spans,
    build_gold_spans,
    token_labels_from_gold_spans,
    viterbi_decode_bio,
    get_label_maps_from_model,
    EvalMetrics,
    compute_binary_metrics,
    compute_token_metrics,
    compute_span_metrics,
    compute_prf,
    write_eval_report,
    metrics_to_json_dict,
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON at line {line_no + 1}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: expected object at line {line_no + 1}")
            yield obj


def _parse_type_thresholds(arg: str) -> dict[str, float]:
    """
    Parse a JSON object like: {"NOME_PESSOA": 0.75, "DOC_PESSOAL": 0.5}
    """
    s = str(arg).strip()
    if not s or s.lower() == "none":
        return {}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for per-type thresholds: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Per-type thresholds must be a JSON object (dict).")
    out: dict[str, float] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise ValueError("Per-type threshold keys must be strings.")
        if not isinstance(v, (int, float)):
            raise ValueError(f"Threshold for {k!r} must be a number.")
        out[k] = float(v)
    return out


def _parse_type_ints(arg: str) -> dict[str, int]:
    """
    Parse a JSON object like: {"NOME_PESSOA": 2}
    """
    s = str(arg).strip()
    if not s or s.lower() == "none":
        return {}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for per-type ints: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Per-type ints must be a JSON object (dict).")
    out: dict[str, int] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise ValueError("Per-type int keys must be strings.")
        if not isinstance(v, int):
            raise ValueError(f"Value for {k!r} must be an int.")
        out[k] = int(v)
    return out


def _infer_one_text(
    *,
    text: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    max_length: int,
    stride: int,
    boundary_backoff: int,
    batch_size: int,
    decode: str,
    aggregate_overlaps: str,
    span_conf_threshold: float,
    span_conf_threshold_by_type: dict[str, float],
    span_conf_agg: str,
    min_span_tokens: int,
    min_span_tokens_by_type: dict[str, int],
    resolve_overlaps: bool,
) -> list[PiiSpan]:
    import torch

    chunks = build_chunks(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        boundary_backoff=boundary_backoff,
    )

    label2id, id2label = get_label_maps_from_model(model)
    o_id = int(label2id["O"])
    decode_n = str(decode).strip().lower()
    if decode_n not in ("argmax", "bio_viterbi"):
        raise ValueError("--decode must be one of: argmax, bio_viterbi")

    aggregate_n = str(aggregate_overlaps).strip().lower()
    if aggregate_n not in ("none", "mean_logits"):
        raise ValueError("--aggregate_overlaps must be one of: none, mean_logits")

    all_spans_scored: list[PiiSpanScored] = []
    token_logits_sum: dict[tuple[int, int], np.ndarray] = {}
    token_logits_count: dict[tuple[int, int], int] = {}
    model.eval()
    with torch.no_grad():
        i = 0
        while i < len(chunks):
            batch = chunks[i : i + batch_size]
            i += batch_size
            texts = [c.text for c in batch]
            enc = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")  # (B, T, 2) on CPU
            attn = enc.get("attention_mask")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits  # (B, T, C)
            logits_np = logits.detach().to("cpu").numpy()
            preds_argmax = torch.argmax(logits, dim=-1).to("cpu").tolist()
            attn_cpu = attn.to("cpu").tolist() if attn is not None else None

            for bi, ch in enumerate(batch):
                offs_i = offsets[bi].tolist()
                offs = [(int(a), int(b)) for (a, b) in offs_i]
                em = logits_np[bi]  # (T,C)

                if aggregate_n == "mean_logits":
                    am = attn_cpu[bi] if attn_cpu is not None else None
                    for ti, (a, b) in enumerate(offs):
                        # Skip special tokens and padding.
                        if a == 0 and b == 0:
                            continue
                        if am is not None and int(am[ti]) == 0:
                            continue
                        ga = int(a) + int(ch.char_start)
                        gb = int(b) + int(ch.char_start)
                        if ga >= gb:
                            continue
                        k = (ga, gb)
                        v = em[ti].astype(np.float32, copy=False)
                        if k in token_logits_sum:
                            token_logits_sum[k] = token_logits_sum[k] + v
                            token_logits_count[k] = int(token_logits_count[k]) + 1
                        else:
                            token_logits_sum[k] = v.copy()
                            token_logits_count[k] = 1
                    continue

                if decode_n == "bio_viterbi":
                    force_o = np.array([(a == 0 and b == 0) for (a, b) in offs], dtype=bool)
                    if attn_cpu is not None:
                        # Padding tokens: attention_mask == 0
                        am = np.array([int(x) for x in attn_cpu[bi]], dtype=np.int32)
                        force_o = np.logical_or(force_o, am == 0)
                    pred_ids = viterbi_decode_bio(
                        emissions=em, id2label=id2label, o_id=o_id, force_o_mask=force_o
                    )
                else:
                    pred_ids = [int(x) for x in preds_argmax[bi]]

                spans_local_scored = spans_from_token_predictions_scored(
                    offsets=offs,
                    pred_ids=pred_ids,
                    logits=em,
                    id2label=id2label,
                    o_id=o_id,
                    conf_agg=span_conf_agg,
                )
                spans_local_scored = filter_scored_spans(
                    spans_local_scored,
                    conf_threshold=float(span_conf_threshold),
                    conf_threshold_by_type=span_conf_threshold_by_type,
                    min_span_tokens=int(min_span_tokens),
                    min_span_tokens_by_type=min_span_tokens_by_type,
                )
                for s in spans_local_scored:
                    all_spans_scored.append(
                        PiiSpanScored(
                            start=int(s.start + ch.char_start),
                            end=int(s.end + ch.char_start),
                            pii_type=s.pii_type,
                            confidence=float(s.confidence),
                            n_tokens=int(s.n_tokens),
                        )
                    )

    if aggregate_n == "mean_logits":
        if not token_logits_sum:
            return []
        # Build a single global token sequence from unique (char_start,char_end) keys.
        keys = sorted(token_logits_sum.keys(), key=lambda x: (x[0], x[1]))
        em_global = np.stack(
            [token_logits_sum[k] / float(token_logits_count[k]) for k in keys], axis=0
        ).astype(np.float32, copy=False)
        offs_global = [(int(a), int(b)) for (a, b) in keys]
        if decode_n == "bio_viterbi":
            pred_ids = viterbi_decode_bio(
                emissions=em_global,
                id2label=id2label,
                o_id=o_id,
                force_o_mask=None,
            )
        else:
            pred_ids = [int(x) for x in np.argmax(em_global, axis=-1).tolist()]
        spans_scored = spans_from_token_predictions_scored(
            offsets=offs_global,
            pred_ids=pred_ids,
            logits=em_global,
            id2label=id2label,
            o_id=o_id,
            conf_agg=span_conf_agg,
        )
        spans_scored = filter_scored_spans(
            spans_scored,
            conf_threshold=float(span_conf_threshold),
            conf_threshold_by_type=span_conf_threshold_by_type,
            min_span_tokens=int(min_span_tokens),
            min_span_tokens_by_type=min_span_tokens_by_type,
        )
        all_spans_scored = spans_scored

    merged = merge_and_resolve_scored_spans(all_spans_scored, resolve_overlaps=resolve_overlaps)
    merged_plain = [PiiSpan(start=s.start, end=s.end, pii_type=s.pii_type) for s in merged]
    return merged_plain


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference/eval for the PII token-classification model.")
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="outputs/finetune-multilabel-multi/best",
        help="HF model name or local path (default: outputs/finetune-multilabel-multi/best).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length (must match training; default: 512).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="Run inference and output spans.")
    p_infer.add_argument("--text", type=str, default="none", help='Text to analyze (or "none").')
    p_infer.add_argument(
        "--text_file",
        type=str,
        default="none",
        help='Read input text from a UTF-8 file (or "none").',
    )
    p_infer.add_argument(
        "--jsonl_in",
        type=str,
        default="none",
        help='Input JSONL with a "text" field per line (or "none").',
    )
    p_infer.add_argument(
        "--jsonl_out",
        type=str,
        default="none",
        help='Output JSONL path (or "none" to print to stdout for --text/--text_file).',
    )
    p_infer.add_argument(
        "--text_column",
        type=str,
        default="text",
        help='Text field name for --jsonl_in (default: "text").',
    )
    p_infer.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Token overlap between chunks (default: 64).",
    )
    p_infer.add_argument(
        "--boundary_backoff",
        type=int,
        default=32,
        help="Try moving chunk end backward to land on safe boundaries (default: 32).",
    )
    p_infer.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size in number of chunks (default: 8).",
    )
    p_infer.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cpu", or "cuda" (default: auto).',
    )
    p_infer.add_argument(
        "--decode",
        type=str,
        default="bio_viterbi",
        choices=["argmax", "bio_viterbi"],
        help='Decoding: "argmax" or BIO-constrained Viterbi ("bio_viterbi"). Default: bio_viterbi.',
    )
    p_infer.add_argument(
        "--aggregate_overlaps",
        type=str,
        default="mean_logits",
        choices=["none", "mean_logits"],
        help=(
            'How to handle overlapping chunks. "none" keeps per-chunk decoding + span merge. '
            '"mean_logits" aggregates token logits across overlapping chunks (by global char offsets) '
            "then decodes once. Default: mean_logits."
        ),
    )
    p_infer.add_argument(
        "--span_conf_threshold",
        type=float,
        default=0.0,
        help="Drop predicted spans with confidence < threshold. 0 disables. Default: 0.",
    )
    p_infer.add_argument(
        "--span_conf_threshold_by_type",
        type=str,
        default="none",
        help='JSON dict of per-type thresholds, e.g. {"NOME_PESSOA":0.75}. Default: none.',
    )
    p_infer.add_argument(
        "--span_conf_agg",
        type=str,
        default="mean",
        choices=["mean", "min"],
        help='How to aggregate token confidences into a span confidence. Default: "mean".',
    )
    p_infer.add_argument(
        "--min_span_tokens",
        type=int,
        default=0,
        help="Drop predicted spans shorter than this number of tokens. 0 disables. Default: 0.",
    )
    p_infer.add_argument(
        "--min_span_tokens_by_type",
        type=str,
        default="none",
        help='JSON dict of per-type min token counts, e.g. {"NOME_PESSOA":2}. Default: none.',
    )
    p_infer.add_argument(
        "--no_resolve_overlaps",
        dest="resolve_overlaps",
        action="store_false",
        default=True,
        help="Disable overlap resolution between different entity types.",
    )

    p_eval = sub.add_parser("eval", help="Evaluate token-level metrics on a JSONL dataset.")
    p_eval.add_argument(
        "--dataset_path",
        type=str,
        default="data/generated/finetune_multilabel.jsonl",
        help="JSONL with {text, entities} (default: data/generated/finetune_multilabel.jsonl).",
    )
    p_eval.add_argument("--text_column", type=str, default="text")
    p_eval.add_argument("--entities_column", type=str, default="entities")
    p_eval.add_argument("--max_rows", type=int, default=-1, help="If > 0, evaluate only first N rows.")
    p_eval.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Token overlap between chunks during eval (default: 64).",
    )
    p_eval.add_argument(
        "--boundary_backoff",
        type=int,
        default=32,
        help="Try moving chunk end backward to land on safe boundaries during eval (default: 32).",
    )
    p_eval.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size in number of chunks during eval (default: 16).",
    )
    p_eval.add_argument(
        "--report_path",
        type=str,
        default="outputs/eval_report.md",
        help="Markdown report output path (default: outputs/eval_report.md).",
    )
    p_eval.add_argument(
        "--max_chars",
        type=int,
        default=200,
        help="Max characters of text to include per sample in report (default: 200).",
    )
    p_eval.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cpu", or "cuda" (default: auto).',
    )
    p_eval.add_argument(
        "--decode",
        type=str,
        default="bio_viterbi",
        choices=["argmax", "bio_viterbi"],
        help='Decoding: "argmax" or BIO-constrained Viterbi ("bio_viterbi"). Default: bio_viterbi.',
    )

    return p.parse_args()


def _resolve_device(device_arg: str):
    import torch

    d = str(device_arg).strip().lower()
    if d == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError('Requested --device=cuda but torch.cuda.is_available() is False.')
        return torch.device("cuda")
    raise ValueError(f"Invalid --device: {device_arg!r}")


def _load_model_and_tokenizer(model_name_or_path: str):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    p = Path(model_name_or_path)
    if str(model_name_or_path).startswith("outputs/") and not p.exists():
        raise FileNotFoundError(f"Model path not found: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("This script requires a fast tokenizer (offset_mapping).")
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    return model, tokenizer


def _cmd_infer(args: argparse.Namespace) -> int:
    import torch

    model, tokenizer = _load_model_and_tokenizer(args.model_name_or_path)
    device = _resolve_device(args.device)
    model.to(device)

    max_length = int(args.max_length)
    if max_length < 16:
        raise ValueError("--max_length too small")
    stride = int(args.stride)
    boundary_backoff = int(args.boundary_backoff)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    span_conf_threshold_by_type = _parse_type_thresholds(args.span_conf_threshold_by_type)
    min_span_tokens_by_type = _parse_type_ints(args.min_span_tokens_by_type)

    text_sources = sum(
        1
        for v in (args.text, args.text_file, args.jsonl_in)
        if isinstance(v, str) and v.strip().lower() != "none"
    )
    if text_sources != 1:
        raise ValueError("Provide exactly one input source: --text or --text_file or --jsonl_in")

    def _format_out(*, text: str, spans: list[PiiSpan]) -> dict[str, Any]:
        pii_types = set(PII_TYPES)
        should_be_public = not any(s.pii_type in pii_types for s in spans)
        return {
            "text": text,
            "spans": [
                {
                    "type": s.pii_type,
                    "start": s.start,
                    "end": s.end,
                    "value": text[s.start : s.end],
                }
                for s in spans
            ],
            "should_be_public": should_be_public,
        }

    if str(args.text).strip().lower() != "none":
        text = args.text
        spans = _infer_one_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            stride=stride,
            boundary_backoff=boundary_backoff,
            batch_size=batch_size,
            decode=str(args.decode),
            aggregate_overlaps=str(args.aggregate_overlaps),
            span_conf_threshold=float(args.span_conf_threshold),
            span_conf_threshold_by_type=span_conf_threshold_by_type,
            span_conf_agg=str(args.span_conf_agg),
            min_span_tokens=int(args.min_span_tokens),
            min_span_tokens_by_type=min_span_tokens_by_type,
            resolve_overlaps=bool(args.resolve_overlaps),
        )
        rec = _format_out(text=text, spans=spans)
        out = json.dumps(rec, ensure_ascii=False)
        if str(args.jsonl_out).strip().lower() != "none":
            Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.jsonl_out).write_text(out + "\n", encoding="utf-8")
        else:
            print(out)
        return 0

    if str(args.text_file).strip().lower() != "none":
        p = Path(args.text_file)
        if not p.is_file():
            raise FileNotFoundError(f"--text_file not found: {p}")
        text = p.read_text(encoding="utf-8")
        spans = _infer_one_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            stride=stride,
            boundary_backoff=boundary_backoff,
            batch_size=batch_size,
            decode=str(args.decode),
            aggregate_overlaps=str(args.aggregate_overlaps),
            span_conf_threshold=float(args.span_conf_threshold),
            span_conf_threshold_by_type=span_conf_threshold_by_type,
            span_conf_agg=str(args.span_conf_agg),
            min_span_tokens=int(args.min_span_tokens),
            min_span_tokens_by_type=min_span_tokens_by_type,
            resolve_overlaps=bool(args.resolve_overlaps),
        )
        rec = _format_out(text=text, spans=spans)
        out = json.dumps(rec, ensure_ascii=False)
        if str(args.jsonl_out).strip().lower() != "none":
            Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.jsonl_out).write_text(out + "\n", encoding="utf-8")
        else:
            print(out)
        return 0

    # JSONL mode
    in_path = Path(args.jsonl_in)
    if not in_path.is_file():
        raise FileNotFoundError(f"--jsonl_in not found: {in_path}")
    if str(args.jsonl_out).strip().lower() == "none":
        raise ValueError("When using --jsonl_in, you must also set --jsonl_out")
    out_path = Path(args.jsonl_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pii_types = set(PII_TYPES)

    # Keep deterministic behavior for multi-GPU launches: infer is read-only, so no need for rank checks.
    n = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for obj in _read_jsonl(in_path):
            if args.text_column not in obj:
                raise ValueError(f"Missing text column {args.text_column!r} in input JSONL.")
            text = obj.get(args.text_column)
            if not isinstance(text, str):
                raise ValueError(f"Invalid text value type in input JSONL: {type(text)}")
            spans = _infer_one_text(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                stride=stride,
                boundary_backoff=boundary_backoff,
                batch_size=batch_size,
                decode=str(args.decode),
                aggregate_overlaps=str(args.aggregate_overlaps),
                span_conf_threshold=float(args.span_conf_threshold),
                span_conf_threshold_by_type=span_conf_threshold_by_type,
                span_conf_agg=str(args.span_conf_agg),
                min_span_tokens=int(args.min_span_tokens),
                min_span_tokens_by_type=min_span_tokens_by_type,
                resolve_overlaps=bool(args.resolve_overlaps),
            )
            rec = dict(obj)
            rec["spans"] = [
                {"type": s.pii_type, "start": s.start, "end": s.end, "value": text[s.start : s.end]}
                for s in spans
            ]
            rec["should_be_public"] = not any(s.pii_type in pii_types for s in spans)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import torch

    if not ALLOWED_ENTITY_TYPES:
        raise RuntimeError("ALLOWED_ENTITY_TYPES is empty. Ensure `ner_labels.py` is importable.")

    model, tokenizer = _load_model_and_tokenizer(args.model_name_or_path)
    device = _resolve_device(args.device)
    model.to(device)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("eval requires a fast tokenizer (offset_mapping).")

    max_length = int(args.max_length)
    if max_length < 16:
        raise ValueError("--max_length too small")

    in_path = Path(args.dataset_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"--dataset_path not found: {in_path}")

    label2id, _id2label = get_label_maps_from_model(model)
    o_id = int(label2id["O"])

    pii_label_ids: set[int] = set()
    for t in PII_TYPES:
        pii_label_ids.add(int(label2id[f"B-{t}"]))
        pii_label_ids.add(int(label2id[f"I-{t}"]))

    ent_label_ids: set[int] = set()
    for t in ALLOWED_ENTITY_TYPES:
        ent_label_ids.add(int(label2id[f"B-{t}"]))
        ent_label_ids.add(int(label2id[f"I-{t}"]))

    # Accumulators for token-level P/R/F1.
    # - pii_*: positive == tokens with label in PII_TYPES (excludes ORG_JURIDICA, DOC_EMPRESA)
    # - ent_*: positive == tokens with any entity label (includes ORG_JURIDICA, DOC_EMPRESA)
    pii_tp = 0
    pii_pred_pos = 0
    pii_true_pos = 0
    ent_tp = 0
    ent_pred_pos = 0
    ent_true_pos = 0

    # Span-level metrics (strict exact match: start, end, and type must all match).
    # This is more rigorous than token-level because a span with any boundary error is counted as 0.
    span_pii_tp = 0
    span_pii_pred_pos = 0
    span_pii_true_pos = 0
    span_ent_tp = 0
    span_ent_pred_pos = 0
    span_ent_true_pos = 0

    # Per-type span-level accumulators (strict exact match).
    per_type_span_tp: dict[str, int] = {t: 0 for t in ALLOWED_ENTITY_TYPES}
    per_type_span_pred_pos: dict[str, int] = {t: 0 for t in ALLOWED_ENTITY_TYPES}
    per_type_span_true_pos: dict[str, int] = {t: 0 for t in ALLOWED_ENTITY_TYPES}

    # Accumulators for binary document-level classification.
    # - pii_bin_*: positive == "has PII" (excludes ORG_JURIDICA, DOC_EMPRESA)
    # - ent_bin_*: positive == "has any entity" (includes ORG_JURIDICA, DOC_EMPRESA)
    pii_bin_tp = 0
    pii_bin_tn = 0
    pii_bin_fp = 0
    pii_bin_fn = 0
    ent_bin_tp = 0
    ent_bin_tn = 0
    ent_bin_fp = 0
    ent_bin_fn = 0

    stride = int(args.stride)
    boundary_backoff = int(args.boundary_backoff)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    max_rows = int(args.max_rows)

    row_index = 0

    max_chars = int(args.max_chars)
    if max_chars <= 0:
        raise ValueError("--max_chars must be > 0")
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _excerpt(text: str) -> str:
        t = " ".join(text.split())
        return t if len(t) <= max_chars else (t[: max_chars - 1] + "…")

    # Per-sample report buckets.
    samples_any_error: list[dict[str, Any]] = []
    samples_token_error: list[dict[str, Any]] = []
    samples_binary_wrong: list[dict[str, Any]] = []
    samples_fp: list[dict[str, Any]] = []
    samples_fn: list[dict[str, Any]] = []
    samples_tp: list[dict[str, Any]] = []
    samples_tn: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        _label2id_local, id2label = get_label_maps_from_model(model)
        num_labels = len(id2label)
        if len(_label2id_local) != num_labels:
            raise RuntimeError("label2id/id2label size mismatch")

        decode_n = str(args.decode).strip().lower()
        if decode_n not in ("argmax", "bio_viterbi"):
            raise ValueError("--decode must be one of: argmax, bio_viterbi")

        for obj in _read_jsonl(in_path):
            if max_rows > 0 and row_index >= max_rows:
                break
            if args.text_column not in obj:
                raise ValueError(f"Missing text column {args.text_column!r} in dataset.")
            if args.entities_column not in obj:
                if "pii_candidate" in obj:
                    raise ValueError(
                        f"Missing entities column {args.entities_column!r} in dataset (found legacy "
                        '"pii_candidate"; please regenerate/migrate your JSONL).'
                    )
                raise ValueError(f"Missing entities column {args.entities_column!r} in dataset.")
            text = obj.get(args.text_column)
            pii = obj.get(args.entities_column)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"row={row_index}: invalid text")
            if not isinstance(pii, list):
                raise ValueError(f"row={row_index}: entities must be a list")

            spans = build_gold_spans(text, pii, row_index=row_index)

            # Full-text tokenization (no truncation) for rigorous evaluation across the entire text.
            enc_full = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )
            offsets_raw = enc_full.get("offset_mapping")
            if offsets_raw is None:
                raise RuntimeError("Tokenizer did not return offset_mapping (requires fast tokenizer).")
            offsets_full: list[tuple[int, int]] = [(int(a), int(b)) for (a, b) in offsets_raw]
            if not offsets_full:
                raise ValueError(f"row={row_index}: empty tokenization")

            # Gold token labels aligned to the full-text tokenization.
            gold_ids = token_labels_from_gold_spans(
                offsets=offsets_full,
                spans=spans,
                label2id=label2id,
                row_index=row_index,
            )

            # Predict per-token labels for the full text using chunking + logit aggregation.
            chunks = build_chunks(
                text=text,
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
                boundary_backoff=boundary_backoff,
            )
            offset2idx: dict[tuple[int, int], int] = {}
            for i, (a, b) in enumerate(offsets_full):
                k = (int(a), int(b))
                if k in offset2idx:
                    raise ValueError(f"row={row_index}: duplicate token offsets in full tokenization: {k}")
                offset2idx[k] = i

            logits_sum = np.zeros((len(offsets_full), num_labels), dtype=np.float32)
            logits_cnt = np.zeros((len(offsets_full),), dtype=np.int32)

            for start in range(0, len(chunks), batch_size):
                chunk_batch = chunks[start : start + batch_size]
                texts_b = [c.text for c in chunk_batch]
                enc_b = tokenizer(
                    texts_b,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets_b = enc_b.pop("offset_mapping")
                attn_b = enc_b.get("attention_mask")
                enc_b = {k: v.to(device) for k, v in enc_b.items()}
                logits_b = model(**enc_b).logits.detach().to("cpu").numpy()  # (B, L, C)
                offsets_b_np = offsets_b.detach().to("cpu").numpy()  # (B, L, 2)
                attn_np = attn_b.detach().to("cpu").numpy() if attn_b is not None else None  # (B, L)

                for bi, c in enumerate(chunk_batch):
                    char_base = int(c.char_start)
                    offs = offsets_b_np[bi]
                    for ti in range(offs.shape[0]):
                        if attn_np is not None and int(attn_np[bi, ti]) == 0:
                            continue
                        a = int(offs[ti, 0])
                        b = int(offs[ti, 1])
                        if a == 0 and b == 0:
                            continue  # special/pad
                        ga = a + char_base
                        gb = b + char_base
                        gi = offset2idx.get((ga, gb))
                        if gi is None:
                            raise ValueError(
                                f"row={row_index}: chunk token offsets ({ga},{gb}) not found in full tokenization "
                                f"(chunk_index={c.chunk_index}, local=({a},{b}))"
                            )
                        logits_sum[gi] += logits_b[bi, ti]
                        logits_cnt[gi] += 1

            if np.any(logits_cnt == 0):
                missing = int(np.sum(logits_cnt == 0))
                raise ValueError(f"row={row_index}: {missing} tokens were not covered by chunk inference.")

            emissions = logits_sum / logits_cnt[:, None]
            if decode_n == "bio_viterbi":
                pred_ids = viterbi_decode_bio(emissions=emissions, id2label=id2label, o_id=o_id, force_o_mask=None)
            else:
                pred_ids = [int(x) for x in np.argmax(emissions, axis=-1).tolist()]

            if len(pred_ids) != len(gold_ids):
                raise RuntimeError(
                    f"row={row_index}: pred/gold length mismatch: {len(pred_ids)} != {len(gold_ids)}"
                )

            pred_has_pii = False
            gold_has_pii = False
            pred_has_entity = False
            gold_has_entity = False

            token_error_pii = False
            token_error_ent = False

            for p, y in zip(pred_ids, gold_ids, strict=True):
                p = int(p)
                y = int(y)
                if y == -100:
                    # There shouldn't be special tokens in full-text tokenization, but keep this for safety.
                    continue

                p_is_pii = p in pii_label_ids
                y_is_pii = y in pii_label_ids
                p_is_ent = p in ent_label_ids
                y_is_ent = y in ent_label_ids

                if p_is_pii:
                    pred_has_pii = True
                    pii_pred_pos += 1
                if y_is_pii:
                    gold_has_pii = True
                    pii_true_pos += 1
                if p_is_pii and y_is_pii and p == y:
                    pii_tp += 1

                if p_is_ent:
                    pred_has_entity = True
                    ent_pred_pos += 1
                if y_is_ent:
                    gold_has_entity = True
                    ent_true_pos += 1
                if p_is_ent and y_is_ent and p == y:
                    ent_tp += 1

                # Token error flags:
                # - PII-only: treat non-PII entity labels as O.
                p_pii = p if p_is_pii else o_id
                y_pii = y if y_is_pii else o_id
                if p_pii != y_pii:
                    token_error_pii = True

                # All-entities: treat any entity label as positive; still strict label match.
                p_ent = p if p_is_ent else o_id
                y_ent = y if y_is_ent else o_id
                if p_ent != y_ent:
                    token_error_ent = True

            # Span-level evaluation (strict exact match: start, end, type must all match).
            # Convert predicted token labels to spans using the same logic as inference.
            pred_labels_str = [id2label.get(int(p), "O") for p in pred_ids]
            pred_spans = spans_from_token_predictions(offsets=offsets_full, labels=pred_labels_str)
            pred_spans = merge_spans(pred_spans)

            # Gold spans are already computed above (variable `spans`).
            gold_spans_set: set[tuple[int, int, str]] = {(s.start, s.end, s.pii_type) for s in spans}
            pred_spans_set: set[tuple[int, int, str]] = {(s.start, s.end, s.pii_type) for s in pred_spans}

            pii_types_set = set(PII_TYPES)
            for s in spans:
                t = s.pii_type
                per_type_span_true_pos[t] = per_type_span_true_pos.get(t, 0) + 1
                span_ent_true_pos += 1
                if t in pii_types_set:
                    span_pii_true_pos += 1

            for s in pred_spans:
                t = s.pii_type
                per_type_span_pred_pos[t] = per_type_span_pred_pos.get(t, 0) + 1
                span_ent_pred_pos += 1
                if t in pii_types_set:
                    span_pii_pred_pos += 1

            # Count exact matches.
            for key in gold_spans_set & pred_spans_set:
                _start, _end, t = key
                per_type_span_tp[t] = per_type_span_tp.get(t, 0) + 1
                span_ent_tp += 1
                if t in pii_types_set:
                    span_pii_tp += 1

            # Binary confusion matrices.
            # PII-only (positive == has_pii)
            if pred_has_pii and gold_has_pii:
                pii_bin_tp += 1
                pii_bin_bucket = "tp"
            elif pred_has_pii and not gold_has_pii:
                pii_bin_fp += 1
                pii_bin_bucket = "fp"
            elif (not pred_has_pii) and gold_has_pii:
                pii_bin_fn += 1
                pii_bin_bucket = "fn"
            else:
                pii_bin_tn += 1
                pii_bin_bucket = "tn"

            # All entities (positive == has_any_entity)
            if pred_has_entity and gold_has_entity:
                ent_bin_tp += 1
                ent_bin_bucket = "tp"
            elif pred_has_entity and not gold_has_entity:
                ent_bin_fp += 1
                ent_bin_bucket = "fp"
            elif (not pred_has_entity) and gold_has_entity:
                ent_bin_fn += 1
                ent_bin_bucket = "fn"
            else:
                ent_bin_tn += 1
                ent_bin_bucket = "tn"

            gold_should_be_public = not gold_has_pii
            pred_should_be_public = not pred_has_pii
            binary_error = pii_bin_bucket in ("fp", "fn")

            sample_rec = {
                "row_index": row_index,
                "gold_should_be_public": bool(gold_should_be_public),
                "pred_should_be_public": bool(pred_should_be_public),
                "pii_binary_bucket": pii_bin_bucket,
                "all_entities_binary_bucket": ent_bin_bucket,
                "pii_token_error": bool(token_error_pii),
                "all_entities_token_error": bool(token_error_ent),
                "binary_error": bool(binary_error),
                "excerpt": _excerpt(str(text)),
                "entities": pii,
            }

            if token_error_pii or binary_error:
                samples_any_error.append(sample_rec)
            if token_error_pii:
                samples_token_error.append(sample_rec)
            if binary_error:
                samples_binary_wrong.append(sample_rec)
            if pii_bin_bucket == "fp":
                samples_fp.append(sample_rec)
            elif pii_bin_bucket == "fn":
                samples_fn.append(sample_rec)
            elif pii_bin_bucket == "tp":
                samples_tp.append(sample_rec)
            elif pii_bin_bucket == "tn":
                samples_tn.append(sample_rec)

            row_index += 1

    # Compute per-type span-level metrics.
    per_type_span_metrics: dict[str, dict[str, float]] = {}
    for t in ALLOWED_ENTITY_TYPES:
        tp = per_type_span_tp.get(t, 0)
        pred_pos = per_type_span_pred_pos.get(t, 0)
        true_pos = per_type_span_true_pos.get(t, 0)
        prec, rec, f1 = compute_prf(tp, pred_pos, true_pos)
        per_type_span_metrics[t] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "pred_pos": pred_pos,
            "true_pos": true_pos,
        }

    # Build metrics object.
    metrics = EvalMetrics(
        row_count=row_index,
        pii_token=compute_token_metrics(pii_tp, pii_pred_pos, pii_true_pos),
        pii_binary=compute_binary_metrics(pii_bin_tp, pii_bin_tn, pii_bin_fp, pii_bin_fn),
        pii_span=compute_span_metrics(span_pii_tp, span_pii_pred_pos, span_pii_true_pos),
        ent_token=compute_token_metrics(ent_tp, ent_pred_pos, ent_true_pos),
        ent_binary=compute_binary_metrics(ent_bin_tp, ent_bin_tn, ent_bin_fp, ent_bin_fn),
        ent_span=compute_span_metrics(span_ent_tp, span_ent_pred_pos, span_ent_true_pos),
        per_type_span=per_type_span_metrics,
        samples_any_error=samples_any_error,
        samples_token_error=samples_token_error,
        samples_binary_wrong=samples_binary_wrong,
        samples_fp=samples_fp,
        samples_fn=samples_fn,
    )

    # Write report and output JSON.
    write_eval_report(metrics, report_path)
    print(json.dumps(metrics_to_json_dict(metrics, report_path), ensure_ascii=False))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


def main() -> int:
    args = _parse_args()

    # Ensure relative model paths work when running from repo root.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.cmd == "infer":
        return _cmd_infer(args)
    if args.cmd == "eval":
        return _cmd_eval(args)
    raise ValueError(f"Unknown cmd: {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
