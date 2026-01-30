"""
Build a fine-tuning JSONL dataset from synthetic.jsonl with inference-aligned chunking.

Input schema (one JSON per line):
  {"text": <str>, "entities": [{"type": <str>, "value": <str>}, ...]}

Output schema (one JSON per line, per chunk):
  {
    "text": <str>,
    "entities": [{"type": <str>, "value": <str>}, ...]
  }

Important:
  If a PII value is sliced by the chunk boundary, the respective slice(s) that appear
  inside the chunk will be emitted as `entities.value` (literal substrings).

Chunking strategy:
  Sliding window over tokens with overlap (stride), designed to match inference time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm
from transformers import AutoTokenizer





def _add_repo_root_to_syspath() -> Path:
    """
    Ensure local imports work regardless of the current working directory.
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

# Inference-aligned chunking implementation.
from data_preprocessing.chunking import Chunk, build_chunks

# Single source of truth for entity taxonomy.
from ner_labels import ALLOWED_ENTITY_TYPES, validate_entity_value_format


@dataclass(frozen=True)
class EntityItem:
    type: str
    value: str



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a finetuning JSONL dataset with inference-aligned chunking."
    )
    p.add_argument(
        "--input",
        type=str,
        default="data/generated/synthetic.jsonl",
        help="Path to input JSONL (text + entities).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/generated/finetune_multilabel.jsonl",
        help="Path to output JSONL (one line per chunk).",
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="neuralmind/bert-base-portuguese-cased",
        help="HF model name or local path (used only for tokenizer).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length including special tokens.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Overlap in tokens between consecutive chunks (in token space, excluding special tokens).",
    )
    p.add_argument(
        "--boundary_backoff",
        type=int,
        default=32,
        help=(
            "Try to move chunk end backward up to this many tokens to land on a safer boundary "
            "(whitespace/punctuation). This is inference-compatible and reduces mid-word cuts."
        ),
    )
    return p.parse_args()


def _read_jsonl(path: Path) -> Iterator[dict]:
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


def _parse_entities(obj: dict, *, source_index: int) -> tuple[str, list[EntityItem]]:
    keys = set(obj.keys())
    required = {"text", "entities"}
    if not required.issubset(keys):
        if keys == {"text", "pii_candidate"}:
            raise ValueError(
                f"source_index={source_index}: schema changed: expected ['entities','text'] but found "
                "legacy ['pii_candidate','text']. Please regenerate/migrate your dataset."
            )
        missing = sorted(required - keys)
        raise ValueError(
            f"source_index={source_index}: missing required keys {missing}; present={sorted(keys)}"
        )

    text = obj.get("text")
    entities = obj.get("entities")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"source_index={source_index}: invalid text")
    if not isinstance(entities, list):
        raise ValueError(f"source_index={source_index}: entities must be a list")

    allowed = set(ALLOWED_ENTITY_TYPES)
    items: list[EntityItem] = []
    for i, it in enumerate(entities):
        if not isinstance(it, dict):
            raise ValueError(f"source_index={source_index}: entities[{i}] must be an object")
        t = it.get("type")
        v = it.get("value")
        if not isinstance(t, str) or t not in allowed:
            raise ValueError(f"source_index={source_index}: entities[{i}].type invalid: {t!r}")
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"source_index={source_index}: entities[{i}].value invalid/empty")
        if v not in text:
            raise ValueError(
                f"source_index={source_index}: entities[{i}].value does not appear literally in text: {v!r}"
            )
        # Fail-fast validation: keep entity values consistent with repo guidelines.
        # If this breaks an existing dataset, regenerate it with updated rules.
        validate_entity_value_format(entity_type=t, value=v)
        items.append(EntityItem(type=t, value=v))

    # CRITICAL invariant for this repo's {type,value} schema:
    # the same literal value cannot map to multiple types in the same text, because training expands
    # each value to *all* its literal occurrences.
    value_to_types: dict[str, set[str]] = {}
    for it in items:
        value_to_types.setdefault(it.value, set()).add(it.type)
    conflicts = [(v, sorted(ts)) for v, ts in value_to_types.items() if len(ts) > 1]
    if conflicts:
        raise ValueError(
            f"source_index={source_index}: same value mapped to multiple types: {conflicts}. "
            "Fix the source data so each literal value has a single type in this text."
        )

    _validate_no_cross_type_overlaps(
        text=text,
        items=items,
        context=f"source_index={source_index}",
    )

    return text, items


def _iter_literal_occurrences(text: str, value: str) -> Iterable[tuple[int, int]]:
    # Literal (non-regex) occurrences, including punctuation/spaces exactly as provided.
    # re.finditer doesn't overlap by default; that's fine for our use case.
    pat = re.compile(re.escape(value))
    for m in pat.finditer(text):
        yield (m.start(), m.end())


def _iter_literal_occurrences_overlapping(text: str, value: str) -> Iterable[tuple[int, int]]:
    """
    Literal occurrences allowing overlap (start+1), matching training-time behavior safely.
    """
    start = 0
    while True:
        i = text.find(value, start)
        if i < 0:
            break
        yield (i, i + len(value))
        start = i + 1


def _validate_no_cross_type_overlaps(
    *, text: str, items: list[EntityItem], context: str
) -> None:
    spans: list[tuple[int, int, str, str]] = []  # (start,end,type,value)
    for it in items:
        for (a, b) in _iter_literal_occurrences_overlapping(text, it.value):
            spans.append((a, b, it.type, it.value))
    spans.sort(key=lambda s: (s[0], s[1], s[2], s[3]))

    for i, (a, b, t, v) in enumerate(spans):
        for (a2, b2, t2, v2) in spans[i + 1 :]:
            if a2 >= b:
                break
            if b2 > a and t2 != t:
                raise RuntimeError(
                    f"{context}: overlapping spans with different types: "
                    f"{(a, b, t, v)} vs {(a2, b2, t2, v2)}"
                )




def _validate_coverage(
    *,
    text: str,
    chunks: list[Chunk],
    pii_items: list[EntityItem],
    source_index: int,
) -> None:
    if not pii_items:
        return

    for it in pii_items:
        any_occurrence = False
        for (s, e) in _iter_literal_occurrences(text, it.value):
            any_occurrence = True
            # With slicing, we require that the union of chunk intersections covers [s,e]
            # (i.e., no gaps).
            pieces: list[tuple[int, int]] = []
            for c in chunks:
                a = max(s, c.char_start)
                b = min(e, c.char_end)
                if a < b:
                    pieces.append((a, b))
            if not pieces:
                raise RuntimeError(
                    "Coverage failure: no chunk overlaps a PII occurrence (unexpected). "
                    f"source_index={source_index}, type={it.type}, value={it.value!r}, span=({s},{e})."
                )
            pieces.sort()
            cur_s, cur_e = pieces[0]
            if cur_s > s:
                raise RuntimeError(
                    "Coverage failure: gap before first slice of a PII occurrence. "
                    f"source_index={source_index}, type={it.type}, value={it.value!r}, span=({s},{e}). "
                    "Try setting --boundary_backoff=0 or increasing --stride."
                )
            for ps, pe in pieces[1:]:
                if ps > cur_e:
                    raise RuntimeError(
                        "Coverage failure: a PII occurrence is split with a gap between chunks. "
                        f"source_index={source_index}, type={it.type}, value={it.value!r}, span=({s},{e}). "
                        "Try setting --boundary_backoff=0 or increasing --stride."
                    )
                cur_e = max(cur_e, pe)
            if cur_e < e:
                raise RuntimeError(
                    "Coverage failure: gap after last slice of a PII occurrence. "
                    f"source_index={source_index}, type={it.type}, value={it.value!r}, span=({s},{e}). "
                    "Try setting --boundary_backoff=0 or increasing --stride."
                )
        if not any_occurrence:
            # Should not happen given generator invariants; still fail fast.
            raise RuntimeError(
                f"source_index={source_index}: expected at least one occurrence of value={it.value!r}"
            )


def _pii_slices_for_chunk(
    *,
    text: str,
    pii_items: list[EntityItem],
    chunk: Chunk,
) -> list[EntityItem]:
    if not pii_items:
        return []
    selected: list[EntityItem] = []
    seen: set[tuple[str, str]] = set()
    for it in pii_items:
        for (s, e) in _iter_literal_occurrences(text, it.value):
            a = max(s, chunk.char_start)
            b = min(e, chunk.char_end)
            if a < b:
                slice_val = text[a:b]
                key = (it.type, slice_val)
                if key not in seen:
                    seen.add(key)
                    selected.append(EntityItem(type=it.type, value=slice_val))
    return selected


def main() -> int:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("Tokenizer is not fast; need a fast tokenizer to use offset_mapping safely.")

    total_written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for source_index, obj in enumerate(tqdm(_read_jsonl(in_path), desc="building chunks")):
            text, entities = _parse_entities(obj, source_index=source_index)

            chunks = build_chunks(
                text=text,
                tokenizer=tokenizer,
                max_length=int(args.max_length),
                stride=int(args.stride),
                boundary_backoff=int(args.boundary_backoff),
            )
            _validate_coverage(text=text, chunks=chunks, pii_items=entities, source_index=source_index)

            for ch in chunks:
                pii_chunk = _pii_slices_for_chunk(text=text, pii_items=entities, chunk=ch)

                # Basic invariants (fail fast)
                for it in pii_chunk:
                    if it.value not in ch.text:
                        raise RuntimeError(
                            f"source_index={source_index} chunk_index={ch.chunk_index}: "
                            f"value not found in chunk text: {it.value!r}"
                        )

                # Same invariant at chunk-level (slicing can create collisions).
                value_to_types: dict[str, set[str]] = {}
                for it in pii_chunk:
                    value_to_types.setdefault(it.value, set()).add(it.type)
                conflicts = [(v, sorted(ts)) for v, ts in value_to_types.items() if len(ts) > 1]
                if conflicts:
                    raise RuntimeError(
                        f"source_index={source_index} chunk_index={ch.chunk_index}: "
                        f"same value mapped to multiple types after slicing: {conflicts}. "
                        "Fix the source data so each literal value has a single type in the chunk."
                    )

                _validate_no_cross_type_overlaps(
                    text=ch.text,
                    items=pii_chunk,
                    context=f"source_index={source_index} chunk_index={ch.chunk_index}",
                )

                rec = {
                    "text": ch.text,
                    "entities": [{"type": it.type, "value": it.value} for it in pii_chunk],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"Wrote {total_written} chunks to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

