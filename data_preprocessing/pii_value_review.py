"""
PII value review workflow for datasets in jsonl format:

Each input line must be a JSON object with:
- "text": str
- "entities": list[{"type": str, "value": str}]

Command 1) extract:
  Generates a JSON file mapping label -> unique list of values (in first-seen order).

Command 2) apply:
  Reads the reviewed JSON (label -> values) and rewrites a jsonl dataset:
  - removes entities whose (type,value) are NOT in the reviewed list
  - optionally adds entities from the reviewed list that appear in the text and are missing

Order & literal preservation (default behavior):
- By default, the script does NOT sanitize/normalize values. It treats values as literal substrings.
- When applying a review JSON, the script preserves the original `entities` order from the dataset.
  Filtering only removes items; it does not reorder them. Optional additions are appended.

Design goals:
- fail fast on malformed inputs (dataset or review JSON)
- deterministic output (stable ordering)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ner_labels import ALLOWED_ENTITY_TYPES, PII_TYPES, sanitize_entity_value, validate_entity_value_format


_RE_CNPJ = re.compile(r"^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$")


def _looks_like_numeric_company_id(value: str) -> bool:
    """
    Conservative heuristic (mirrors training-time normalization in `inference/spans.py`).
    Used to fix legacy cases where a CNPJ/IE/IM was labeled as ORG_JURIDICA/ID_PROCESSUAL.
    """
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
class Entity:
    entity_type: str
    value: str


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                raise ValueError(f"{path}: empty line at {i}")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON at line {i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: expected JSON object at line {i}")
            yield obj


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_review_json(path: Path, *, validate_values: bool) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object (label -> list[str])")

    allowed = set(ALLOWED_ENTITY_TYPES)
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or k not in allowed:
            raise ValueError(f"{path}: unknown/invalid label key: {k!r}")
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"{path}: label {k!r} must map to list[str]")
        cleaned: list[str] = []
        seen: set[str] = set()
        for s in v:
            if not s or s != s.strip():
                raise ValueError(f"{path}: label {k!r} has invalid value (empty/whitespace): {s!r}")
            if validate_values:
                if "\n" in s or "\r" in s:
                    raise ValueError(
                        f"{path}: label {k!r} has value with newline (disallowed when --validate-values): {s!r}"
                    )
                # Optional: enforce repo's value format rules (can be stricter than some legacy datasets).
                validate_entity_value_format(entity_type=k, value=s)
            if s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        out[k] = cleaned
    return out


def _canonicalize_entity(
    *,
    text: str,
    entity_type: str,
    value: str,
    normalize: bool,
    migrate_company_ids: bool,
    validate_values: bool,
) -> Entity:
    if entity_type not in set(ALLOWED_ENTITY_TYPES):
        raise ValueError(f"invalid entity_type: {entity_type!r}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("entity value must be a non-empty string")

    t = entity_type
    v = value
    if normalize:
        v = sanitize_entity_value(entity_type=t, value=v, text=text)
        if migrate_company_ids:
            # Optional migration for legacy cases where CNPJ/IE/IM were labeled as ORG_JURIDICA/ID_PROCESSUAL.
            # WARNING: This is heuristic and can misclassify some identifiers; keep it opt-in.
            if t in ("ORG_JURIDICA", "ID_PROCESSUAL") and _looks_like_numeric_company_id(v):
                if v in text and _has_company_doc_keyword_near_value(text=text, value=v):
                    t = "DOC_EMPRESA"
            # One more sanitize pass if the type changed.
            v = sanitize_entity_value(entity_type=t, value=v, text=text)

    if validate_values:
        # Optional: enforce repo's value format rules (can be stricter than some legacy datasets).
        validate_entity_value_format(entity_type=t, value=v)
    if v not in text:
        raise ValueError(f"entity value not found in text: type={t!r} value={v!r}")
    return Entity(entity_type=t, value=v)


def extract_values(
    *,
    input_jsonl: Path,
    output_json: Path,
    normalize: bool,
    migrate_company_ids: bool,
    validate_values: bool,
    pii_only: bool,
) -> None:
    by_label: dict[str, list[str]] = defaultdict(list)
    seen_by_label: dict[str, set[str]] = defaultdict(set)
    allowed = set(PII_TYPES) if pii_only else set(ALLOWED_ENTITY_TYPES)

    for row_idx, row in enumerate(_read_jsonl(input_jsonl), start=1):
        text = row.get("text")
        entities = row.get("entities")
        if not isinstance(text, str):
            raise ValueError(f"{input_jsonl}: row={row_idx}: 'text' must be a string")
        if not isinstance(entities, list):
            raise ValueError(f"{input_jsonl}: row={row_idx}: 'entities' must be a list")

        for j, it in enumerate(entities):
            if not isinstance(it, dict) or set(it.keys()) != {"type", "value"}:
                raise ValueError(
                    f"{input_jsonl}: row={row_idx}: entities[{j}] must be an object with keys ['type','value']"
                )
            t = it.get("type")
            v = it.get("value")
            if not isinstance(t, str) or t not in set(ALLOWED_ENTITY_TYPES):
                raise ValueError(f"{input_jsonl}: row={row_idx}: entities[{j}].type invalid: {t!r}")
            if t not in allowed:
                continue
            if not isinstance(v, str):
                raise ValueError(f"{input_jsonl}: row={row_idx}: entities[{j}].value must be a string")
            ent = _canonicalize_entity(
                text=text,
                entity_type=t,
                value=v,
                normalize=normalize,
                migrate_company_ids=migrate_company_ids,
                validate_values=validate_values,
            )
            if ent.value in seen_by_label[ent.entity_type]:
                continue
            seen_by_label[ent.entity_type].add(ent.value)
            by_label[ent.entity_type].append(ent.value)

    # Preserve label order by first appearance in the dataset.
    out: dict[str, list[str]] = dict(by_label)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_review(
    *,
    input_jsonl: Path,
    review_json: Path,
    output_jsonl: Path,
    normalize: bool,
    migrate_company_ids: bool,
    validate_values: bool,
    add_missing: bool,
    require_all_labels: bool,
    forbid_value_in_multiple_labels: bool,
) -> None:
    review = _load_review_json(review_json, validate_values=validate_values)
    review_sets: dict[str, set[str]] = {k: set(v) for k, v in review.items()}

    if forbid_value_in_multiple_labels:
        seen: dict[str, str] = {}
        for lab, vals in review_sets.items():
            for val in vals:
                prev = seen.get(val)
                if prev is not None and prev != lab:
                    raise ValueError(
                        f"{review_json}: value appears under multiple labels: {val!r} in {prev!r} and {lab!r}"
                    )
                seen[val] = lab

    allowed_types = set(ALLOWED_ENTITY_TYPES)

    def _rewrite_row(row: dict[str, Any], row_idx: int) -> dict[str, Any]:
        text = row.get("text")
        entities = row.get("entities")
        if not isinstance(text, str):
            raise ValueError(f"{input_jsonl}: row={row_idx}: 'text' must be a string")
        if not isinstance(entities, list):
            raise ValueError(f"{input_jsonl}: row={row_idx}: 'entities' must be a list")

        # Preserve original order: filter in-place without reordering.
        kept_entities: list[dict[str, str]] = []
        kept_keys: set[tuple[str, str]] = set()
        for j, it in enumerate(entities):
            if not isinstance(it, dict) or set(it.keys()) != {"type", "value"}:
                raise ValueError(
                    f"{input_jsonl}: row={row_idx}: entities[{j}] must be an object with keys ['type','value']"
                )
            t = it.get("type")
            v = it.get("value")
            if not isinstance(t, str) or t not in allowed_types:
                raise ValueError(f"{input_jsonl}: row={row_idx}: entities[{j}].type invalid: {t!r}")
            if not isinstance(v, str):
                raise ValueError(f"{input_jsonl}: row={row_idx}: entities[{j}].value must be a string")

            ent = _canonicalize_entity(
                text=text,
                entity_type=t,
                value=v,
                normalize=normalize,
                migrate_company_ids=migrate_company_ids,
                validate_values=validate_values,
            )

            # If review doesn't include this label, fail fast unless explicitly allowed.
            if require_all_labels and ent.entity_type not in review_sets:
                raise ValueError(
                    f"{review_json}: missing label {ent.entity_type!r} required by dataset "
                    f"(found in row={row_idx})"
                )

            allowed_vals = review_sets.get(ent.entity_type)
            if allowed_vals is None:
                # label not in review file -> keep as-is (unless require_all_labels=True).
                key = (ent.entity_type, ent.value)
                kept_entities.append({"type": ent.entity_type, "value": ent.value})
                kept_keys.add(key)
            else:
                if ent.value in allowed_vals:
                    key = (ent.entity_type, ent.value)
                    kept_entities.append({"type": ent.entity_type, "value": ent.value})
                    kept_keys.add(key)
                # else: drop

        if add_missing:
            # Preserve order: append missing items in the order they appear in the review file.
            for lab, vals in review.items():
                for val in vals:
                    if val not in text:
                        continue
                    ent = _canonicalize_entity(
                        text=text,
                        entity_type=lab,
                        value=val,
                        normalize=normalize,
                        migrate_company_ids=migrate_company_ids,
                        validate_values=validate_values,
                    )
                    key = (ent.entity_type, ent.value)
                    if key in kept_keys:
                        continue
                    kept_entities.append({"type": ent.entity_type, "value": ent.value})
                    kept_keys.add(key)

        # Keep any extra keys in row, but overwrite entities with the rewritten version.
        out_row = dict(row)
        out_row["entities"] = kept_entities
        return out_row

    rows_out = (_rewrite_row(row, i) for i, row in enumerate(_read_jsonl(input_jsonl), start=1))
    _write_jsonl(output_jsonl, rows_out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract and apply reviewed PII values grouped by label.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract values per label from a jsonl dataset.")
    p_ext.add_argument("--input", required=True, type=Path, help="Input jsonl (e.g. data/esic_sample_eval.jsonl)")
    p_ext.add_argument("--output", required=True, type=Path, help="Output json (label -> values)")
    p_ext.add_argument(
        "--normalize",
        action="store_true",
        help="Sanitize values (strip common prefixes like 'CPF:', 'SEI nº', etc.).",
    )
    p_ext.add_argument(
        "--migrate-company-ids",
        action="store_true",
        help="Heuristically migrate legacy CNPJ/IE/IM mislabeled as ORG_JURIDICA/ID_PROCESSUAL to DOC_EMPRESA.",
    )
    p_ext.add_argument(
        "--validate-values",
        action="store_true",
        help="Validate values against repo rules (can fail on legacy datasets).",
    )
    p_ext.add_argument(
        "--pii-only",
        action="store_true",
        help="Only extract PII labels (excludes ORG_JURIDICA and DOC_EMPRESA).",
    )

    p_app = sub.add_parser("apply", help="Apply a reviewed label->values JSON to rewrite a jsonl dataset.")
    p_app.add_argument("--input", required=True, type=Path, help="Input jsonl to be rewritten")
    p_app.add_argument("--review", required=True, type=Path, help="Reviewed json (label -> values)")
    p_app.add_argument("--output", required=True, type=Path, help="Output rewritten jsonl")
    p_app.add_argument(
        "--normalize",
        action="store_true",
        help="Sanitize values (strip common prefixes like 'CPF:', 'SEI nº', etc.).",
    )
    p_app.add_argument(
        "--migrate-company-ids",
        action="store_true",
        help="Heuristically migrate legacy CNPJ/IE/IM mislabeled as ORG_JURIDICA/ID_PROCESSUAL to DOC_EMPRESA.",
    )
    p_app.add_argument(
        "--validate-values",
        action="store_true",
        help="Validate dataset + review values against repo rules (can fail on legacy datasets).",
    )
    p_app.add_argument(
        "--require-all-labels",
        action="store_true",
        help="Fail if the review JSON does not include a label found in the dataset.",
    )
    p_app.add_argument(
        "--allow-value-in-multiple-labels",
        action="store_true",
        help="Allow the same literal value to be listed under multiple labels in the review JSON.",
    )
    p_app.add_argument(
        "--add-missing",
        action="store_true",
        help="Add reviewed values that appear in the text but are missing from entities.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.cmd == "extract":
        extract_values(
            input_jsonl=args.input,
            output_json=args.output,
            normalize=bool(args.normalize),
            migrate_company_ids=bool(args.migrate_company_ids),
            validate_values=bool(args.validate_values),
            pii_only=bool(args.pii_only),
        )
        return

    if args.cmd == "apply":
        apply_review(
            input_jsonl=args.input,
            review_json=args.review,
            output_jsonl=args.output,
            normalize=bool(args.normalize),
            migrate_company_ids=bool(args.migrate_company_ids),
            validate_values=bool(args.validate_values),
            add_missing=bool(args.add_missing),
            require_all_labels=bool(args.require_all_labels),
            forbid_value_in_multiple_labels=not bool(args.allow_value_in_multiple_labels),
        )
        return

    raise SystemExit(f"unknown command: {args.cmd!r}")


if __name__ == "__main__":
    main()

