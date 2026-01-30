"""
BIO label utilities and Viterbi decoding for token classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_label_maps_from_model(model: Any) -> tuple[dict[str, int], dict[int, str]]:
    """
    Extract label2id and id2label from a model's config.

    Falls back to the repository label space (from ner_labels.py) if the model
    config does not have valid mappings.
    """
    from ner_labels import ALLOWED_ENTITY_TYPES

    label2id = getattr(model.config, "label2id", None)
    id2label = getattr(model.config, "id2label", None)
    if isinstance(label2id, dict) and isinstance(id2label, dict) and label2id and id2label:
        # Normalize keys to expected types.
        label2id_n = {str(k): int(v) for k, v in label2id.items()}
        id2label_n = {int(k): str(v) for k, v in id2label.items()}
        return label2id_n, id2label_n

    # Fall back to repository label space (fail-fast if constants are missing).
    if not ALLOWED_ENTITY_TYPES:
        raise RuntimeError("ALLOWED_ENTITY_TYPES is empty. Check `ner_labels.py`.")
    label_list: list[str] = ["O"]
    for t in ALLOWED_ENTITY_TYPES:
        label_list.append(f"B-{t}")
        label_list.append(f"I-{t}")
    label2id_n = {l: i for i, l in enumerate(label_list)}
    id2label_n = {i: l for l, i in label2id_n.items()}
    return label2id_n, id2label_n


def softmax_last_dim(x: np.ndarray) -> np.ndarray:
    """
    Stable softmax over last dimension.
    """
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=-1, keepdims=True)
    z = x - m
    e = np.exp(z)
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s


def is_bio_label(lab: str) -> bool:
    """Check if a label is a valid BIO label (O, B-*, or I-*)."""
    if lab == "O":
        return True
    if "-" not in lab:
        return False
    prefix, typ = lab.split("-", 1)
    return prefix in ("B", "I") and bool(typ)


def split_bio(lab: str) -> tuple[str, str | None]:
    """
    Split a BIO label into (prefix, type).

    Returns ("O", None) for the O label or invalid labels.
    """
    if lab == "O":
        return ("O", None)
    if "-" not in lab:
        return ("O", None)
    prefix, typ = lab.split("-", 1)
    if prefix not in ("B", "I") or not typ:
        return ("O", None)
    return (prefix, typ)


def build_bio_transition_scores(*, id2label: dict[int, str], o_id: int) -> np.ndarray:
    """
    Build a (C,C) transition score matrix for BIO constraints.

    trans[p, c] is added when moving from prev label p to current label c.
    Allowed transitions have 0.0; disallowed transitions have a large negative penalty.
    """
    c = len(id2label)
    bad = np.float32(-1e4)
    trans = np.full((c, c), bad, dtype=np.float32)

    # Default: allow everything to/from O only if labels are unknown.
    for p in range(c):
        for n in range(c):
            prev = id2label.get(int(p), "O")
            nxt = id2label.get(int(n), "O")
            if not is_bio_label(prev) or not is_bio_label(nxt):
                continue

            if prev == "O":
                # O -> O or O -> B-X
                if nxt == "O":
                    trans[p, n] = 0.0
                else:
                    n_prefix, _n_typ = split_bio(nxt)
                    if n_prefix == "B":
                        trans[p, n] = 0.0
            else:
                p_prefix, p_typ = split_bio(prev)
                if nxt == "O":
                    trans[p, n] = 0.0
                else:
                    n_prefix, n_typ = split_bio(nxt)
                    if n_prefix == "B":
                        trans[p, n] = 0.0
                    elif n_prefix == "I" and p_typ is not None and n_typ == p_typ:
                        # Continue entity only with same type.
                        trans[p, n] = 0.0

    # Ensure O->O is always allowed.
    if 0 <= o_id < c:
        trans[o_id, o_id] = 0.0
    return trans


def viterbi_decode_bio(
    *,
    emissions: np.ndarray,  # (T, C)
    id2label: dict[int, str],
    o_id: int,
    force_o_mask: np.ndarray | None,  # (T,) True => force O
) -> list[int]:
    """
    BIO-constrained Viterbi decoding on raw logits.

    This runs purely as post-processing: ONNX-friendly (doesn't touch model graph).
    """
    emissions = np.asarray(emissions, dtype=np.float32)
    if emissions.ndim != 2:
        raise ValueError(f"emissions must be 2D (T,C), got shape={emissions.shape}")
    t, c = emissions.shape
    if c != len(id2label):
        raise ValueError(f"emissions C={c} != len(id2label)={len(id2label)}")
    if t == 0:
        return []
    if not (0 <= int(o_id) < c):
        raise ValueError(f"Invalid o_id={o_id} for num_labels={c}")

    trans = build_bio_transition_scores(id2label=id2label, o_id=int(o_id))
    bad = np.float32(-1e4)

    # Force some positions to be O (special tokens / padding).
    em = emissions.copy()
    if force_o_mask is not None:
        m = np.asarray(force_o_mask, dtype=bool)
        if m.shape != (t,):
            raise ValueError(f"force_o_mask must have shape (T,), got {m.shape}")
        if np.any(m):
            em[m, :] = bad
            em[m, int(o_id)] = 0.0

    # Start transitions: forbid starting with I-*
    start = np.zeros((c,), dtype=np.float32)
    for i in range(c):
        lab = id2label.get(int(i), "O")
        prefix, _typ = split_bio(lab)
        if prefix == "I":
            start[i] = bad

    dp = np.full((t, c), bad, dtype=np.float32)
    bp = np.zeros((t, c), dtype=np.int32)

    dp[0] = em[0] + start
    bp[0] = 0
    for ti in range(1, t):
        prev = dp[ti - 1][:, None]  # (C,1)
        scores = prev + trans  # (C,C)
        best_prev = np.argmax(scores, axis=0)  # (C,)
        best_score = scores[best_prev, np.arange(c)]
        dp[ti] = em[ti] + best_score
        bp[ti] = best_prev.astype(np.int32)

    last = int(np.argmax(dp[t - 1]))
    path = [0] * t
    path[t - 1] = last
    for ti in range(t - 1, 0, -1):
        last = int(bp[ti, last])
        path[ti - 1] = last
    return [int(x) for x in path]
