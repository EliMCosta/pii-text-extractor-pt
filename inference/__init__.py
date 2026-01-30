"""
Inference utilities for the PII token-classification model.
"""

from inference.spans import (
    PiiSpan,
    PiiSpanScored,
    spans_from_token_predictions,
    spans_from_token_predictions_scored,
    filter_scored_spans,
    merge_spans,
    merge_and_resolve_scored_spans,
    build_gold_spans,
    token_labels_from_gold_spans,
)

from inference.decoding import (
    softmax_last_dim,
    is_bio_label,
    split_bio,
    build_bio_transition_scores,
    viterbi_decode_bio,
    get_label_maps_from_model,
)

from inference.eval_report import (
    TokenLevelMetrics,
    BinaryMetrics,
    SpanLevelMetrics,
    EvalMetrics,
    compute_prf,
    compute_binary_metrics,
    compute_token_metrics,
    compute_span_metrics,
    generate_eval_report,
    write_eval_report,
    metrics_to_json_dict,
)

__all__ = [
    # spans.py
    "PiiSpan",
    "PiiSpanScored",
    "spans_from_token_predictions",
    "spans_from_token_predictions_scored",
    "filter_scored_spans",
    "merge_spans",
    "merge_and_resolve_scored_spans",
    "build_gold_spans",
    "token_labels_from_gold_spans",
    # decoding.py
    "softmax_last_dim",
    "is_bio_label",
    "split_bio",
    "build_bio_transition_scores",
    "viterbi_decode_bio",
    "get_label_maps_from_model",
    # eval_report.py
    "TokenLevelMetrics",
    "BinaryMetrics",
    "SpanLevelMetrics",
    "EvalMetrics",
    "compute_prf",
    "compute_binary_metrics",
    "compute_token_metrics",
    "compute_span_metrics",
    "generate_eval_report",
    "write_eval_report",
    "metrics_to_json_dict",
]
