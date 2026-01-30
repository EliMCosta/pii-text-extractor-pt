"""
Shared text chunking utilities used by both training-data generation and inference.

Design goals:
- Token-budget aware (max_length includes special tokens).
- Sliding-window in token space with overlap (stride) for boundary robustness.
- Avoid splitting inside "words" when chunk windows start/end on continuation subwords.
- Prefer ending chunks on sentence boundaries (.,!,?, newline) when close to max_length.
- Include trailing separator characters (whitespace/punctuation) up to the next token start
  so we don't create character-level gaps between chunks (important when PIIs contain spaces).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Characters that we treat as "part of a word" for boundary checks.
# We include apostrophes and hyphens to avoid splitting names like O'Neil or João-Paulo.
_WORD_EXTRA = {"'", "’", "-", "_"}

# Preferred sentence terminators for end-of-chunk (higher priority than generic boundaries).
_SENTENCE_END = {".", "!", "?", "\n"}


@dataclass(frozen=True)
class Chunk:
    text: str
    char_start: int
    char_end: int
    chunk_index: int


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch in _WORD_EXTRA


def _is_word_boundary(text: str, char_pos: int) -> bool:
    """
    True if `char_pos` is a boundary between "word-ish" characters and non-word.
    Interpreting `char_pos` as a cursor position between chars, i.e.:
      left char is text[char_pos-1], right char is text[char_pos].
    """
    if char_pos <= 0 or char_pos >= len(text):
        return True
    return not (_is_word_char(text[char_pos - 1]) and _is_word_char(text[char_pos]))


def _back_to_word_start(text: str, char_pos: int) -> int:
    if char_pos <= 0:
        return 0
    p = min(char_pos, len(text))
    while p > 0 and _is_word_char(text[p - 1]):
        p -= 1
    return p


def _forward_over_separators(text: str, char_pos: int, *, limit: int) -> int:
    """
    Extend `char_pos` forward over non-word separators, but never past `limit`.
    (Used to include whitespace/punct between tokens so chunk unions don't have gaps.)
    """
    p = min(max(char_pos, 0), len(text))
    lim = min(max(limit, 0), len(text))
    while p < lim and not _is_word_char(text[p]):
        p += 1
    return p


def _last_non_space_char(text: str, end_pos: int) -> str | None:
    """
    Inspect backwards from end_pos-1 to find the last non-whitespace char.
    Returns the char or None if none exists.
    """
    i = min(end_pos, len(text)) - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return text[i] if i >= 0 else None


def _retokenized_length(tokenizer: Any, *, chunk_text: str) -> int:
    enc = tokenizer(
        chunk_text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
        verbose=False,
    )
    return len(enc["input_ids"])


def _pick_end_tok(
    *,
    text: str,
    offsets: list[tuple[int, int]],
    start_tok: int,
    tentative_end_exclusive: int,
    hard_min_end_exclusive: int,
    backoff_tokens: int,
) -> int:
    """
    Pick an end token (exclusive) <= tentative_end_exclusive.
    Preference order (within `backoff_tokens`):
    - End on sentence terminator (.,!,?, newline) near the limit
    - End on a word boundary (no mid-word cut)
    - Otherwise fall back to tentative_end_exclusive
    """
    if tentative_end_exclusive <= hard_min_end_exclusive:
        return tentative_end_exclusive
    if backoff_tokens <= 0:
        return tentative_end_exclusive

    best_end: int | None = None
    best_score = -1
    # Search from closest to limit to further back.
    max_back = min(backoff_tokens, tentative_end_exclusive - hard_min_end_exclusive)
    for d in range(0, max_back + 1):
        end_tok = tentative_end_exclusive - d
        if end_tok <= hard_min_end_exclusive:
            break
        char_end = offsets[end_tok - 1][1]
        if not _is_word_boundary(text, char_end):
            continue
        last = _last_non_space_char(text, char_end)
        if last is None:
            continue
        score = 1
        if last in _SENTENCE_END:
            score = 3
        # Slightly prefer ends that are very close to the limit.
        score = score * 10 - d
        if score > best_score:
            best_score = score
            best_end = end_tok
            if last in _SENTENCE_END and d == 0:
                # Perfect: at max and ends sentence.
                break

    return best_end if best_end is not None else tentative_end_exclusive


def build_chunks(
    *,
    text: str,
    tokenizer: Any,
    max_length: int,
    stride: int,
    boundary_backoff: int = 32,
) -> list[Chunk]:
    """
    Sliding window over tokens with overlap (stride). Token-budget aware.

    Notes:
    - Uses fast-tokenizer offset mappings (required).
    - Adjusts chunk boundaries to avoid splitting within word-like spans.
    - Prefers sentence endings when close to max_length.
    """
    if not isinstance(text, str) or not text:
        return [Chunk(text="", char_start=0, char_end=0, chunk_index=0)]
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("build_chunks requires a fast tokenizer (offset_mapping).")
    if max_length <= 8:
        raise ValueError(f"max_length too small: {max_length}")

    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    token_budget = int(max_length) - specials
    if token_budget <= 0:
        raise ValueError(f"max_length={max_length} too small for tokenizer special tokens={specials}")
    if stride < 0 or stride >= token_budget:
        raise ValueError(f"stride must satisfy 0 <= stride < token_budget ({token_budget}); got {stride}")
    step = token_budget - stride
    if step <= 0:
        raise ValueError(f"Invalid step=token_budget-stride: {step}")

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
        verbose=False,
    )
    input_ids: list[int] = enc["input_ids"]
    offsets_raw = enc["offset_mapping"]
    offsets: list[tuple[int, int]] = [(int(a), int(b)) for (a, b) in offsets_raw]
    if len(input_ids) != len(offsets):
        raise RuntimeError("Tokenizer returned mismatched input_ids and offset_mapping lengths")
    if not input_ids:
        return [Chunk(text=text, char_start=0, char_end=len(text), chunk_index=0)]

    # If the entire text fits in one window, do not chunk: it is both faster and avoids
    # accidentally trimming a tail segment due to boundary heuristics.
    n = len(input_ids)
    if n <= token_budget:
        return [Chunk(text=text, char_start=0, char_end=len(text), chunk_index=0)]

    chunks: list[Chunk] = []
    start_tok = 0
    chunk_idx = 0
    while start_tok < n:
        tentative_end = min(start_tok + token_budget, n)

        # To avoid gaps when we back off ends, enforce that the chunk end cannot move
        # before the next chunk's token start (start_tok + step).
        min_end_exclusive = min(tentative_end, start_tok + step)
        hard_min_end_exclusive = max(start_tok + 1, min_end_exclusive)

        # If this is the last possible window (i.e., there will be no next start_tok),
        # we must cover the tail: don't apply end-backoff heuristics.
        is_last_window = (start_tok + step) >= n
        if is_last_window:
            end_tok = tentative_end
        else:
            end_tok = _pick_end_tok(
                text=text,
                offsets=offsets,
                start_tok=start_tok,
                tentative_end_exclusive=tentative_end,
                hard_min_end_exclusive=hard_min_end_exclusive,
                backoff_tokens=int(boundary_backoff),
            )
        if end_tok <= start_tok:
            raise RuntimeError(f"Failed to make progress: start_tok={start_tok} end_tok={end_tok}")
        if end_tok < hard_min_end_exclusive:
            raise RuntimeError(
                "Chunk end moved before hard minimum (unexpected). "
                f"start_tok={start_tok} end_tok={end_tok} hard_min_end_exclusive={hard_min_end_exclusive}"
            )

        # Character spans.
        base_char_start = offsets[start_tok][0]
        base_char_end = offsets[end_tok - 1][1]
        if base_char_end <= base_char_start:
            raise RuntimeError(
                f"Invalid char span from offsets: start_tok={start_tok}, end_tok={end_tok}, "
                f"char_start={base_char_start}, char_end={base_char_end}"
            )

        # Avoid starting mid-word by backing up to word start.
        char_start = _back_to_word_start(text, base_char_start)

        # Avoid ending mid-word by backing off tokens if needed.
        # (E.g. WordPiece can end at 'play' then next token is '##ing'.)
        while end_tok > hard_min_end_exclusive:
            char_end_try = offsets[end_tok - 1][1]
            if _is_word_boundary(text, char_end_try):
                break
            end_tok -= 1

        char_end = offsets[end_tok - 1][1]
        if char_end <= char_start:
            raise RuntimeError(
                f"Invalid char span after boundary tightening: start_tok={start_tok}, end_tok={end_tok}, "
                f"char_start={char_start}, char_end={char_end}"
            )

        # Include trailing separators up to the next token start so we don't create char gaps.
        next_tok_char_start = offsets[end_tok][0] if end_tok < n else len(text)
        char_end = _forward_over_separators(text, char_end, limit=next_tok_char_start)

        # Ensure re-tokenization fits max_length once we have word-aligned boundaries.
        # Tighten end_tok (and derived char_end) until it fits, without creating token gaps.
        while True:
            chunk_text = text[char_start:char_end]
            if _retokenized_length(tokenizer, chunk_text=chunk_text) <= max_length:
                break
            if end_tok <= hard_min_end_exclusive:
                raise RuntimeError(
                    "Chunk exceeds max_length after retokenization even at hard minimum end. "
                    f"start_tok={start_tok} end_tok={end_tok} hard_min_end_exclusive={hard_min_end_exclusive} "
                    f"max_length={max_length}. Try increasing --max_length or --stride."
                )
            # Reduce by one token and recompute char_end + separator extension.
            end_tok -= 1
            char_end = offsets[end_tok - 1][1]
            next_tok_char_start = offsets[end_tok][0] if end_tok < n else len(text)
            char_end = _forward_over_separators(text, char_end, limit=next_tok_char_start)
            if char_end <= char_start:
                raise RuntimeError(
                    f"Invalid char span while tightening: start_tok={start_tok}, end_tok={end_tok}, "
                    f"char_start={char_start}, char_end={char_end}"
                )

        chunks.append(
            Chunk(
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                chunk_index=chunk_idx,
            )
        )
        chunk_idx += 1

        if end_tok >= n:
            break
        start_tok += step

    return chunks

