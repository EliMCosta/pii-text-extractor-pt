"""
PII span data structures and manipulation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import re
import numpy as np

from inference.decoding import is_bio_label, split_bio, softmax_last_dim


_RE_CNPJ = re.compile(r"^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$")


def _looks_like_numeric_company_id(value: str) -> bool:
    v = str(value).strip()
    if not v:
        return False
    if any(ch.isspace() for ch in v):
        return False
    if any(ch.isalpha() for ch in v):
        return False
    digits = sum(ch.isdigit() for ch in v)
    if digits < 6:
        return False
    return bool(_RE_CNPJ.fullmatch(v) or re.fullmatch(r"[0-9][0-9.\-/]{5,}", v))


def _has_company_doc_keyword_near_value(*, text: str, value: str, window: int = 40) -> bool:
    i = text.find(value)
    if i < 0:
        return False
    left = text[max(0, i - window) : i].casefold()
    if "cnpj" in left:
        return True
    if "inscri" in left and ("estadual" in left or "municipal" in left):
        return True
    if re.search(r"(?i)(?:^|[^a-z])i\.?\s*e\.?(?:[^a-z]|$)", left):
        return True
    if re.search(r"(?i)(?:^|[^a-z])i\.?\s*m\.?(?:[^a-z]|$)", left):
        return True
    return False


@dataclass(frozen=True)
class PiiSpan:
    """A detected PII span with character offsets."""

    start: int  # char start (inclusive)
    end: int  # char end (exclusive)
    pii_type: str


@dataclass(frozen=True)
class PiiSpanScored:
    """A detected PII span with character offsets and confidence score."""

    start: int  # char start (inclusive)
    end: int  # char end (exclusive)
    pii_type: str
    confidence: float  # [0,1]
    n_tokens: int


def spans_from_token_predictions(
    *,
    offsets: list[tuple[int, int]],
    labels: list[str],
) -> list[PiiSpan]:
    """
    Convert per-token predicted IOB labels + offset mappings into character spans.
    Offsets are in the local chunk string coordinates.
    """
    if len(offsets) != len(labels):
        raise ValueError(f"offsets/labels length mismatch: {len(offsets)} != {len(labels)}")

    spans: list[PiiSpan] = []
    cur_type: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    prev_type: str | None = None
    prev_is_entity = False

    for (a, b), lab in zip(offsets, labels):
        is_special = a == 0 and b == 0
        if is_special or lab == "O":
            if cur_type is not None and cur_start is not None and cur_end is not None:
                spans.append(PiiSpan(start=cur_start, end=cur_end, pii_type=cur_type))
            cur_type = None
            cur_start = None
            cur_end = None
            prev_type = None
            prev_is_entity = False
            continue

        if "-" not in lab:
            # Unknown label shape; skip rather than inventing behavior.
            continue
        prefix, typ = lab.split("-", 1)
        if not typ:
            continue

        start_new = False
        if prefix == "B":
            start_new = True
        elif not prev_is_entity or prev_type != typ:
            # I- without a matching previous entity -> start a new span.
            start_new = True

        if start_new:
            if cur_type is not None and cur_start is not None and cur_end is not None:
                spans.append(PiiSpan(start=cur_start, end=cur_end, pii_type=cur_type))
            cur_type = typ
            cur_start = int(a)
            cur_end = int(b)
        else:
            # Continuation of the current entity
            if cur_type != typ or cur_start is None or cur_end is None:
                cur_type = typ
                cur_start = int(a)
                cur_end = int(b)
            else:
                cur_end = max(cur_end, int(b))

        prev_type = typ
        prev_is_entity = True

    if cur_type is not None and cur_start is not None and cur_end is not None:
        spans.append(PiiSpan(start=cur_start, end=cur_end, pii_type=cur_type))
    spans.sort(key=lambda s: (s.start, s.end, s.pii_type))
    return spans


def spans_from_token_predictions_scored(
    *,
    offsets: list[tuple[int, int]],
    pred_ids: list[int],
    logits: np.ndarray,  # (T,C) aligned with pred_ids
    id2label: dict[int, str],
    o_id: int,
    conf_agg: str,
) -> list[PiiSpanScored]:
    """
    Convert per-token predicted label IDs + logits into scored character spans.
    """
    if len(offsets) != len(pred_ids):
        raise ValueError(f"offsets/pred_ids length mismatch: {len(offsets)} != {len(pred_ids)}")
    if logits.ndim != 2 or logits.shape[0] != len(pred_ids):
        raise ValueError(f"logits shape mismatch: got {logits.shape}, expected ({len(pred_ids)}, C)")

    conf_agg_n = str(conf_agg).strip().lower()
    if conf_agg_n not in ("mean", "min"):
        raise ValueError(f"Invalid conf_agg={conf_agg!r}; expected 'mean' or 'min'")

    probs = softmax_last_dim(logits)

    spans: list[PiiSpanScored] = []
    cur_type: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    cur_confs: list[float] = []
    prev_type: str | None = None
    prev_is_entity = False

    def _flush() -> None:
        nonlocal cur_type, cur_start, cur_end, cur_confs
        if cur_type is None or cur_start is None or cur_end is None or not cur_confs:
            cur_type = None
            cur_start = None
            cur_end = None
            cur_confs = []
            return
        if conf_agg_n == "min":
            conf = float(min(cur_confs))
        else:
            conf = float(sum(cur_confs) / len(cur_confs))
        spans.append(
            PiiSpanScored(
                start=int(cur_start),
                end=int(cur_end),
                pii_type=str(cur_type),
                confidence=conf,
                n_tokens=int(len(cur_confs)),
            )
        )
        cur_type = None
        cur_start = None
        cur_end = None
        cur_confs = []

    for i, ((a, b), pid) in enumerate(zip(offsets, pred_ids)):
        a_i = int(a)
        b_i = int(b)
        is_special = a_i == 0 and b_i == 0
        pid_i = int(pid)
        lab = id2label.get(pid_i, "O")

        if is_special or lab == "O" or not is_bio_label(lab):
            _flush()
            prev_type = None
            prev_is_entity = False
            continue

        prefix, typ = split_bio(lab)
        if typ is None:
            _flush()
            prev_type = None
            prev_is_entity = False
            continue

        token_conf = float(probs[i, pid_i])

        start_new = False
        if prefix == "B":
            start_new = True
        elif not prev_is_entity or prev_type != typ:
            start_new = True

        if start_new:
            _flush()
            cur_type = typ
            cur_start = a_i
            cur_end = b_i
            cur_confs = [token_conf]
        else:
            if cur_type != typ or cur_start is None or cur_end is None:
                _flush()
                cur_type = typ
                cur_start = a_i
                cur_end = b_i
                cur_confs = [token_conf]
            else:
                cur_end = max(cur_end, b_i)
                cur_confs.append(token_conf)

        prev_type = typ
        prev_is_entity = True

    _flush()
    spans.sort(key=lambda s: (s.start, s.end, s.pii_type))
    return spans


def filter_scored_spans(
    spans: list[PiiSpanScored],
    *,
    conf_threshold: float,
    conf_threshold_by_type: dict[str, float],
    min_span_tokens: int,
    min_span_tokens_by_type: dict[str, int],
) -> list[PiiSpanScored]:
    """Filter scored spans by confidence threshold and minimum token count."""
    if not spans:
        return []
    out: list[PiiSpanScored] = []
    for s in spans:
        t = str(s.pii_type)
        th = float(conf_threshold_by_type.get(t, conf_threshold))
        min_tok = int(min_span_tokens_by_type.get(t, min_span_tokens))
        if min_tok > 0 and s.n_tokens < min_tok:
            continue
        if th > 0.0 and s.confidence < th:
            continue
        out.append(s)
    return out


def merge_and_resolve_scored_spans(
    spans: list[PiiSpanScored], *, resolve_overlaps: bool
) -> list[PiiSpanScored]:
    """
    1) Merge overlapping/contiguous spans of the same type.
    2) Optionally resolve overlaps of different types by keeping the higher-confidence span.
    """
    if not spans:
        return []

    # Merge same-type spans first (common due to chunk overlap).
    spans_sorted = sorted(spans, key=lambda s: (s.pii_type, s.start, s.end))
    merged: list[PiiSpanScored] = []
    cur = spans_sorted[0]
    for s in spans_sorted[1:]:
        if s.pii_type == cur.pii_type and s.start <= cur.end:
            cur = PiiSpanScored(
                start=cur.start,
                end=max(cur.end, s.end),
                pii_type=cur.pii_type,
                confidence=max(float(cur.confidence), float(s.confidence)),
                n_tokens=int(cur.n_tokens + s.n_tokens),
            )
        else:
            merged.append(cur)
            cur = s
    merged.append(cur)

    merged.sort(key=lambda s: (s.start, s.end, s.pii_type))
    if not resolve_overlaps:
        return merged

    # Resolve overlaps of different types (keep the best-scored span).
    out: list[PiiSpanScored] = []
    for s in merged:
        if not out:
            out.append(s)
            continue
        prev = out[-1]
        if s.start < prev.end and s.pii_type != prev.pii_type:
            # overlap
            keep_s = False
            if s.confidence > prev.confidence:
                keep_s = True
            elif s.confidence == prev.confidence and (s.end - s.start) > (prev.end - prev.start):
                keep_s = True

            if keep_s:
                out[-1] = s
            # else: drop s
        else:
            out.append(s)
    return out


def merge_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    """
    Merge overlapping/contiguous spans of the same type (common due to chunk overlap).
    """
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda s: (s.pii_type, s.start, s.end))
    out: list[PiiSpan] = []
    cur = spans_sorted[0]
    for s in spans_sorted[1:]:
        if s.pii_type == cur.pii_type and s.start <= cur.end:
            cur = PiiSpan(start=cur.start, end=max(cur.end, s.end), pii_type=cur.pii_type)
        else:
            out.append(cur)
            cur = s
    out.append(cur)
    out.sort(key=lambda s: (s.start, s.end, s.pii_type))
    return out


def _iter_literal_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Find all literal occurrences of a value in text."""
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        i = text.find(value, start)
        if i < 0:
            break
        out.append((i, i + len(value)))
        start = i + len(value)
    return out


def build_gold_spans(text: str, entities: list[dict[str, Any]], *, row_index: int) -> list[PiiSpan]:
    """
    Build gold-standard PII spans from entity annotations.

    The dataset provides only (type, value) without character offsets. Expanding values to *all*
    literal occurrences can create overlapping spans across different types. BIO token classification
    can't represent overlapping entities, so we deterministically keep a non-overlapping subset,
    preferring longer spans.
    """
    from ner_labels import ALLOWED_ENTITY_TYPES, sanitize_entity_value

    allowed = set(ALLOWED_ENTITY_TYPES)
    spans: list[PiiSpan] = []
    for j, it in enumerate(entities):
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
        # Keep gold values aligned with repo-wide sanitization rules.
        v = sanitize_entity_value(entity_type=t, value=v, text=text)
        if t in ("ORG_JURIDICA", "ID_PROCESSUAL"):
            v2 = sanitize_entity_value(entity_type="DOC_EMPRESA", value=v, text=text)
            if isinstance(v2, str) and v2 and v2 in text:
                v = v2
        # Migration shim:
        # Historically, CNPJ/IE/IM could be labeled as ORG_JURIDICA (or even ID_PROCESSUAL in some datasets).
        # The canonical taxonomy now uses DOC_EMPRESA for those identifiers.
        if t in ("ORG_JURIDICA", "ID_PROCESSUAL") and _looks_like_numeric_company_id(v):
            if _has_company_doc_keyword_near_value(text=text, value=v):
                t = "DOC_EMPRESA"
        for a, b in _iter_literal_occurrences(text, v):
            spans.append(PiiSpan(start=a, end=b, pii_type=t))

    if not spans:
        return []

    # Deduplicate exact spans first.
    uniq = sorted({(s.start, s.end, s.pii_type) for s in spans})
    candidates = [PiiSpan(start=a, end=b, pii_type=t) for (a, b, t) in uniq]

    # Prefer longer spans, then earlier ones, then stable by type.
    candidates.sort(key=lambda s: (-(s.end - s.start), s.start, s.end, s.pii_type))
    kept: list[PiiSpan] = []
    for s in candidates:
        overlaps = any((s.start < k.end and s.end > k.start) for k in kept)
        if overlaps:
            continue
        kept.append(s)

    # Deterministic output for downstream processing.
    kept.sort(key=lambda s: (s.start, s.end, s.pii_type))
    return kept


def token_labels_from_gold_spans(
    *,
    offsets: list[tuple[int, int]],
    spans: list[PiiSpan],
    label2id: dict[str, int],
    row_index: int,
) -> list[int]:
    """
    Convert gold spans to per-token BIO label IDs.

    Handles truncation gracefully: spans beyond the covered window are ignored.
    """
    o_id = label2id["O"]
    labels = [o_id for _ in offsets]

    special = [a == 0 and b == 0 for (a, b) in offsets]
    for i, is_special in enumerate(special):
        if is_special:
            labels[i] = -100

    # When `truncation=True` is used in tokenization, `offsets` only cover the visible window.
    # Gold spans may still refer to parts of the original text beyond this window; those spans
    # cannot be represented in the token labels and should be ignored (not treated as an error).
    covered_ends = [
        b for (a, b), lab in zip(offsets, labels, strict=True) if lab != -100 and (a != 0 or b != 0)
    ]
    if not covered_ends:
        raise ValueError(f"row={row_index}: could not determine covered text window from offsets")
    covered_max_end = max(int(b) for b in covered_ends)

    for span in spans:
        token_idxs: list[int] = []
        for i, (a, b) in enumerate(offsets):
            if labels[i] == -100:
                continue
            if a < span.end and b > span.start:
                token_idxs.append(i)
        if not token_idxs:
            # If the whole span starts beyond the truncated window, ignore it.
            if int(span.start) >= covered_max_end:
                continue
            raise ValueError(
                f"row={row_index}: span {span} did not align to any token (offset mapping issue)"
            )
        for k, ti in enumerate(token_idxs):
            tag = ("B-" if k == 0 else "I-") + span.pii_type
            new_id = label2id[tag]
            cur = labels[ti]
            if cur not in (-100, o_id) and cur != new_id:
                raise ValueError(
                    f"row={row_index}: token label conflict at token={ti}: cur={cur} new={new_id}"
                )
            labels[ti] = new_id

    return labels
