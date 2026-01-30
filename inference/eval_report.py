"""
Evaluation report generation for PII token classification.

Handles metrics computation and markdown report generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenLevelMetrics:
    """Token-level evaluation metrics."""

    tp: int
    pred_pos: int
    true_pos: int
    precision: float
    recall: float
    f1: float


@dataclass
class BinaryMetrics:
    """Binary (document-level) evaluation metrics."""

    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass
class SpanLevelMetrics:
    """Span-level evaluation metrics (strict exact match)."""

    tp: int
    pred_pos: int
    true_pos: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvalMetrics:
    """All evaluation metrics computed during eval."""

    row_count: int

    # PII-only metrics (excludes ORG_JURIDICA, DOC_EMPRESA)
    pii_token: TokenLevelMetrics
    pii_binary: BinaryMetrics
    pii_span: SpanLevelMetrics

    # All entities metrics (includes ORG_JURIDICA, DOC_EMPRESA)
    ent_token: TokenLevelMetrics
    ent_binary: BinaryMetrics
    ent_span: SpanLevelMetrics

    # Per-type span metrics
    per_type_span: dict[str, dict[str, float]]

    # Sample buckets for report
    samples_any_error: list[dict[str, Any]]
    samples_token_error: list[dict[str, Any]]
    samples_binary_wrong: list[dict[str, Any]]
    samples_fp: list[dict[str, Any]]
    samples_fn: list[dict[str, Any]]


def compute_prf(tp: int, pred_pos: int, true_pos: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from counts."""
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / true_pos if true_pos else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> BinaryMetrics:
    """Compute binary classification metrics."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    return BinaryMetrics(
        tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, accuracy=accuracy
    )


def compute_token_metrics(tp: int, pred_pos: int, true_pos: int) -> TokenLevelMetrics:
    """Compute token-level metrics."""
    precision, recall, f1 = compute_prf(tp, pred_pos, true_pos)
    return TokenLevelMetrics(
        tp=tp, pred_pos=pred_pos, true_pos=true_pos, precision=precision, recall=recall, f1=f1
    )


def compute_span_metrics(tp: int, pred_pos: int, true_pos: int) -> SpanLevelMetrics:
    """Compute span-level metrics."""
    precision, recall, f1 = compute_prf(tp, pred_pos, true_pos)
    return SpanLevelMetrics(
        tp=tp, pred_pos=pred_pos, true_pos=true_pos, precision=precision, recall=recall, f1=f1
    )


# --- Markdown formatting utilities ---


def _md_escape_html(s: str) -> str:
    """Escape HTML special characters for use inside <summary> tags."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def _md_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    align: list[str] | None = None,
) -> str:
    """Generate a markdown table."""

    def _cell(v: Any) -> str:
        s = str(v)
        s = s.replace("|", r"\|").replace("\n", " ")
        return s

    def _sep(a: str) -> str:
        a = (a or "").strip().lower()
        if a == "right":
            return "---:"
        if a == "center":
            return ":---:"
        if a == "left":
            return ":---"
        return "---"

    if align is None:
        align = ["left"] * len(headers)
    if len(align) != len(headers):
        raise ValueError("align must match number of headers")

    out: list[str] = []
    out.append("| " + " | ".join(_cell(h) for h in headers) + " |")
    out.append("| " + " | ".join(_sep(a) for a in align) + " |")
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    out.append("")
    return "\n".join(out)


def _fmt_int(n: int) -> str:
    """Format integer with pt-BR friendly thousands separator."""
    return f"{int(n):,}".replace(",", ".")


def _fmt_score(x: float) -> str:
    """Format score as fraction and percentage."""
    return f"{x:.4f} ({(100.0 * x):.2f}%)"


def _md_samples(title: str, items: list[dict[str, Any]], *, max_items: int | None = None) -> str:
    """Generate markdown section with sample details."""
    total = len(items)
    out_lines = [f"## {title} (n={_fmt_int(total)})", ""]
    if total == 0:
        out_lines.append("_None_")
        out_lines.append("")
        return "\n".join(out_lines)

    shown = items
    if max_items is not None and total > max_items:
        shown = items[:max_items]
        out_lines.append(f"_Showing first {len(shown)} of {total} samples._")
        out_lines.append("")

    for s in shown:
        bucket_pt = {
            "tp": "verdadeiro positivo",
            "tn": "verdadeiro negativo",
            "fp": "falso positivo",
            "fn": "falso negativo",
        }
        ents = s.get("entities") or []
        pii_bucket = str(s.get("pii_binary_bucket"))
        pii_bucket_h = bucket_pt.get(pii_bucket, pii_bucket)
        # `row_index` is 0-based (internal counter). Humans typically search JSONL by 1-based file lines.
        line_no = int(s["row_index"]) + 1
        summary_bits = [
            f"linha={line_no}",
            f"doc (PII)={pii_bucket_h}",
            f"publicável (referência)={s['gold_should_be_public']}",
            f"publicável (modelo)={s['pred_should_be_public']}",
            f"erro em tokens (PII)={s['pii_token_error']}",
        ]
        summary = " | ".join(summary_bits)
        excerpt = str(s.get("excerpt") or "").strip()

        out_lines.append("<details>")
        out_lines.append(f"<summary>{_md_escape_html(summary)} — {_md_escape_html(excerpt)}</summary>")
        out_lines.append("")
        out_lines.append(f"- **linha (arquivo .jsonl, 1-based)**: `{line_no}`")
        out_lines.append(f"- **row_index (0-based)**: `{s['row_index']}`")
        out_lines.append(f"- **publicável (referência)**: `{s['gold_should_be_public']}`")
        out_lines.append(f"- **publicável (modelo)**: `{s['pred_should_be_public']}`")
        out_lines.append(f"- **resultado do documento (PII)**: `{pii_bucket}` ({pii_bucket_h})")
        out_lines.append(
            f"- **resultado do documento (qualquer entidade)**: `{s['all_entities_binary_bucket']}`"
        )
        out_lines.append(f"- **erro em tokens (PII)**: `{s['pii_token_error']}`")
        out_lines.append(f"- **erro em tokens (qualquer entidade)**: `{s['all_entities_token_error']}`")
        out_lines.append("")
        if excerpt:
            out_lines.append("**trecho (excerpt)**:")
            out_lines.append("")
            out_lines.append(f"> {excerpt}")
            out_lines.append("")
        if ents:
            out_lines.append("**entidades (entities)**:")
            out_lines.append("")
            out_lines.append("```json")
            out_lines.append(json.dumps(ents, ensure_ascii=False, indent=2))
            out_lines.append("```")
            out_lines.append("")
        out_lines.append("</details>")
        out_lines.append("")

    return "\n".join(out_lines)


def generate_eval_report(metrics: EvalMetrics) -> str:
    """
    Generate the full markdown evaluation report.

    Args:
        metrics: Computed evaluation metrics.

    Returns:
        The complete markdown report as a string.
    """
    lines: list[str] = []

    # Header and TOC
    lines.append("# Relatório de Avaliação — Identificação de PII")
    lines.append("")
    lines.append("## Conteúdo")
    lines.append("")
    lines.append("- [Resumo](#resumo)")
    lines.append("- [Como ler (glossário)](#como-ler-glossário)")
    lines.append("- [Apenas PII](#apenas-pii-exclui-org_juridica-e-doc_empresa)")
    lines.append("- [Qualquer entidade](#qualquer-entidade-inclui-org_juridica-e-doc_empresa)")
    lines.append("- [Erros](#erros)")
    lines.append("")

    # Summary
    lines.append("## Resumo")
    lines.append("")
    lines.append(
        _md_table(
            ["item", "valor"],
            [["total de linhas avaliadas", _fmt_int(metrics.row_count)]],
            align=["left", "right"],
        )
    )

    # Glossary
    lines.append("## Como ler (glossário)")
    lines.append("")
    lines.append(
        '- **Referência**: o rótulo correto do dataset ("gold").\n'
        "- **Modelo**: a predição do modelo.\n"
        "- **Documento (binário)**: considera apenas se o texto tem PII (sim/não).\n"
        "- **PII-only**: métricas que ignoram `ORG_JURIDICA` e `DOC_EMPRESA`.\n"
        "- **Qualquer entidade**: inclui também `ORG_JURIDICA` e `DOC_EMPRESA`.\n"
        "- **Falso positivo**: modelo disse que tem PII, mas a referência diz que não.\n"
        "- **Falso negativo**: modelo disse que não tem PII, mas a referência diz que tem.\n"
    )
    lines.append("")

    # PII-only section
    lines.append("### Apenas PII (exclui ORG_JURIDICA e DOC_EMPRESA)")
    lines.append("")

    # PII Binary
    lines.append("#### Documento (binário) — positivo = contém PII")
    lines.append("")
    pii_bin = metrics.pii_binary
    lines.append(
        _md_table(
            ["referência \\ modelo", "predisse PII", "predisse sem PII"],
            [
                ["contém PII", _fmt_int(pii_bin.tp), _fmt_int(pii_bin.fn)],
                ["não contém PII", _fmt_int(pii_bin.fp), _fmt_int(pii_bin.tn)],
            ],
            align=["left", "right", "right"],
        )
    )
    lines.append(
        _md_table(
            ["métrica", "valor"],
            [
                ["precisão", _fmt_score(pii_bin.precision)],
                ["revocação (recall)", _fmt_score(pii_bin.recall)],
                ["F1 (P1)", _fmt_score(pii_bin.f1)],
                ["acurácia", _fmt_score(pii_bin.accuracy)],
            ],
            align=["left", "right"],
        )
    )

    # All entities section
    lines.append("### Qualquer entidade (inclui ORG_JURIDICA e DOC_EMPRESA)")
    lines.append("")

    # Entity Binary
    lines.append("#### Documento (binário) — positivo = contém qualquer entidade")
    lines.append("")
    ent_bin = metrics.ent_binary
    lines.append(
        _md_table(
            ["referência \\ modelo", "predisse entidade", "predisse sem entidade"],
            [
                ["contém entidade", _fmt_int(ent_bin.tp), _fmt_int(ent_bin.fn)],
                ["não contém entidade", _fmt_int(ent_bin.fp), _fmt_int(ent_bin.tn)],
            ],
            align=["left", "right", "right"],
        )
    )
    lines.append(
        _md_table(
            ["métrica", "valor"],
            [
                ["precisão", _fmt_score(ent_bin.precision)],
                ["revocação (recall)", _fmt_score(ent_bin.recall)],
                ["F1", _fmt_score(ent_bin.f1)],
                ["acurácia", _fmt_score(ent_bin.accuracy)],
            ],
            align=["left", "right"],
        )
    )

    # Errors section
    lines.append("## Erros")
    lines.append("")
    lines.append(
        _md_table(
            ["categoria", "quantidade"],
            [
                [
                    "erro binário no documento (falso pos. + falso neg.)",
                    _fmt_int(len(metrics.samples_binary_wrong)),
                ],
                ["falsos positivos (documento)", _fmt_int(len(metrics.samples_fp))],
                ["falsos negativos (documento)", _fmt_int(len(metrics.samples_fn))],
            ],
            align=["left", "right"],
        )
    )

    # Sample sections (only false negatives)
    lines.append(_md_samples("Falsos negativos (documento)", metrics.samples_fn))

    return "\n".join(lines).rstrip() + "\n"


def write_eval_report(metrics: EvalMetrics, report_path: Path) -> None:
    """Generate and write the evaluation report to a file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = generate_eval_report(metrics)
    report_path.write_text(report_content, encoding="utf-8")


def metrics_to_json_dict(metrics: EvalMetrics, report_path: Path) -> dict[str, Any]:
    """Convert metrics to the JSON dict format for stdout output."""
    return {
        "pii_token_level": {
            "precision": metrics.pii_token.precision,
            "recall": metrics.pii_token.recall,
            "f1": metrics.pii_token.f1,
            "tp": metrics.pii_token.tp,
            "pred_pos": metrics.pii_token.pred_pos,
            "true_pos": metrics.pii_token.true_pos,
        },
        "pii_binary": {
            "precision": metrics.pii_binary.precision,
            "recall": metrics.pii_binary.recall,
            "f1": metrics.pii_binary.f1,
            "accuracy": metrics.pii_binary.accuracy,
            "tp": metrics.pii_binary.tp,
            "tn": metrics.pii_binary.tn,
            "fp": metrics.pii_binary.fp,
            "fn": metrics.pii_binary.fn,
        },
        "pii_p1": metrics.pii_binary.f1,
        "all_entities_token_level": {
            "precision": metrics.ent_token.precision,
            "recall": metrics.ent_token.recall,
            "f1": metrics.ent_token.f1,
            "tp": metrics.ent_token.tp,
            "pred_pos": metrics.ent_token.pred_pos,
            "true_pos": metrics.ent_token.true_pos,
        },
        "all_entities_binary": {
            "precision": metrics.ent_binary.precision,
            "recall": metrics.ent_binary.recall,
            "f1": metrics.ent_binary.f1,
            "accuracy": metrics.ent_binary.accuracy,
            "tp": metrics.ent_binary.tp,
            "tn": metrics.ent_binary.tn,
            "fp": metrics.ent_binary.fp,
            "fn": metrics.ent_binary.fn,
        },
        "all_entities_p1": metrics.ent_binary.f1,
        "pii_span_level": {
            "precision": metrics.pii_span.precision,
            "recall": metrics.pii_span.recall,
            "f1": metrics.pii_span.f1,
            "tp": metrics.pii_span.tp,
            "pred_pos": metrics.pii_span.pred_pos,
            "true_pos": metrics.pii_span.true_pos,
        },
        "all_entities_span_level": {
            "precision": metrics.ent_span.precision,
            "recall": metrics.ent_span.recall,
            "f1": metrics.ent_span.f1,
            "tp": metrics.ent_span.tp,
            "pred_pos": metrics.ent_span.pred_pos,
            "true_pos": metrics.ent_span.true_pos,
        },
        "per_type_span_level": metrics.per_type_span,
        "report_path": str(report_path),
    }
