"""
Task-specific outcome reward functions for IGPO multi-turn training.

Provides a unified compute_score interface that routes to the correct
scoring function based on the data_source (or an explicit task_type in
extra_info).  This replaces the retrieval-only F1/EM implementation in
the original IGPO codebase with a generalised version covering:

  - Retrieval / open-domain QA  (F1 + EM, answer extracted from <answer> tags)
  - Mathematical reasoning       (exact match / numerical accuracy)
  - Coding / program synthesis   (pass@1 via an external sandbox)

The "score" key in the returned dict is what IGPORewardManager places as
the outcome reward (final-turn scalar).  Additional keys are logged as
reward_extra_info for monitoring.

Extending to new task types
---------------------------
Register a new scorer in TASK_SCORE_REGISTRY:

    from verl.utils.reward_score.igpo_tasks import TASK_SCORE_REGISTRY

    def my_score_fn(solution_str, ground_truth, extra_info=None):
        ...
        return {"score": ..., "my_metric": ...}

    TASK_SCORE_REGISTRY["my_source_prefix"] = my_score_fn
"""

import re
import string
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _preprocess(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_answer_tag(solution_str: str) -> Optional[str]:
    """Extract content of the first <answer>...</answer> tag (case-insensitive)."""
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _tags_balanced(solution_str: str) -> bool:
    """Check that key structural tags are properly paired (not interleaved)."""
    s = solution_str.lower()
    for tag in ("think", "answer", "tool_call", "code"):
        opens = s.count(f"<{tag}>")
        closes = s.count(f"</{tag}>")
        if opens != closes:
            return False
    return True


def _token_f1(pred: str, ref: str) -> float:
    """Word-level F1 between two preprocessed strings."""
    pred_toks = set(pred.split())
    ref_toks = set(ref.split())
    if not pred_toks or not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Retrieval / open-domain QA scorer
# ---------------------------------------------------------------------------

# Penalty returned when the response has structural format errors
_FORMAT_ERROR_PENALTY: float = -2.0


def compute_retrieval_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict[str, Any]:
    """F1 + EM score for retrieval and open-domain QA tasks.

    The function extracts the answer from <answer>...</answer> tags, handles
    multi-label answers separated by "<|answer_split|>", and computes:
      - F1: token-level overlap (primary metric, used as 'score')
      - EM: exact match
      - noformat_f1: F1 ignoring tag-balance check (for diagnostics)

    Args:
        solution_str: Full decoded model response.
        ground_truth: Reference answer string, possibly containing
            "<|answer_split|>" to indicate multiple valid answers.
        extra_info: Optional dict; currently unused for this scorer.

    Returns:
        {"score": f1, "f1": f1, "em": em, "noformat_f1": noformat_f1}
    """
    sol = solution_str.lower()
    gt_variants = [_preprocess(g) for g in ground_truth.split("<|answer_split|>")]

    # Format check
    if not _tags_balanced(sol):
        noformat_f1 = 0.0
        # Still try to extract answer for noformat_f1
        raw_answer = _extract_answer_tag(sol)
        if raw_answer:
            noformat_f1 = max(
                _token_f1(_preprocess(raw_answer), g) for g in gt_variants
            )
        return {
            "score": _FORMAT_ERROR_PENALTY,
            "f1": _FORMAT_ERROR_PENALTY,
            "em": 0.0,
            "noformat_f1": noformat_f1,
        }

    raw_answer = _extract_answer_tag(sol)
    if raw_answer is None:
        return {
            "score": _FORMAT_ERROR_PENALTY,
            "f1": _FORMAT_ERROR_PENALTY,
            "em": 0.0,
            "noformat_f1": 0.0,
        }

    pred = _preprocess(raw_answer)
    f1 = max(_token_f1(pred, g) for g in gt_variants)
    em = float(any(pred == g for g in gt_variants))
    noformat_f1 = f1  # Already passed format check

    return {"score": f1, "f1": f1, "em": em, "noformat_f1": noformat_f1}


# ---------------------------------------------------------------------------
# Mathematical reasoning scorer
# ---------------------------------------------------------------------------

def compute_math_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict[str, Any]:
    """Exact-match / numerical accuracy for math tasks.

    Delegates to verl's existing math_dapo scorer (which handles LaTeX,
    numeric equivalence, etc.) and wraps the result in the unified dict
    format so IGPORewardManager can log the "acc" metric.

    Falls back to a simple normalised exact-match if the math_dapo module
    is unavailable.

    Args:
        solution_str: Full decoded model response.
        ground_truth: Reference answer (number, expression, or LaTeX).
        extra_info: Optional; currently unused.

    Returns:
        {"score": acc, "acc": acc}
    """
    try:
        from verl.utils.reward_score import math_dapo
        acc = float(math_dapo.compute_score(solution_str, ground_truth))
    except Exception:
        # Fallback: simple case-insensitive exact match on extracted answer
        raw = _extract_answer_tag(solution_str.lower()) or ""
        pred = _preprocess(raw)
        ref = _preprocess(ground_truth.lower())
        acc = float(pred == ref)

    return {"score": acc, "acc": acc}


# ---------------------------------------------------------------------------
# Code / program synthesis scorer
# ---------------------------------------------------------------------------

def compute_code_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict[str, Any]:
    """Pass@1 for coding tasks via sandbox execution.

    ground_truth is expected to contain serialised test cases that the
    execution sandbox can run.  Delegates to sandbox_fusion when a URL is
    available in extra_info, otherwise falls back to prime_code.

    Args:
        solution_str: Full decoded model response (should contain code).
        ground_truth: Serialised test-case specification.
        extra_info: Optional dict; may contain:
            "sandbox_fusion_url" (str): URL for the sandbox execution service.
            "memory_limit_mb" (int): Memory limit for sandbox.

    Returns:
        {"score": pass_rate, "pass_rate": pass_rate}
    """
    extra_info = extra_info or {}
    sandbox_url = extra_info.get("sandbox_fusion_url")

    try:
        if sandbox_url:
            from verl.utils.reward_score import sandbox_fusion
            semaphore = extra_info.get("concurrent_semaphore")
            memory_limit = extra_info.get("memory_limit_mb")
            pass_rate = float(
                sandbox_fusion.compute_score(
                    sandbox_url, semaphore, memory_limit,
                    solution_str, ground_truth, continuous=True,
                )
            )
        else:
            from verl.utils.reward_score import prime_code
            pass_rate = float(
                prime_code.compute_score(
                    solution_str, ground_truth, continuous=True
                )
            )
    except Exception:
        pass_rate = 0.0

    return {"score": pass_rate, "pass_rate": pass_rate}


# ---------------------------------------------------------------------------
# Registry and unified entry point
# ---------------------------------------------------------------------------

# Maps data_source prefixes / exact names to scorer functions.
# Keys are matched with startswith() so prefixes like "retrieval_" work.
TASK_SCORE_REGISTRY: dict[str, Callable] = {
    # Retrieval / QA
    "searchR1": compute_retrieval_score,
    "nq": compute_retrieval_score,
    "triviaqa": compute_retrieval_score,
    "popqa": compute_retrieval_score,
    "hotpotqa": compute_retrieval_score,
    "2wikimultihop": compute_retrieval_score,
    "musique": compute_retrieval_score,
    "bamboogle": compute_retrieval_score,
    "factbench": compute_retrieval_score,
    "politifact": compute_retrieval_score,
    "liar": compute_retrieval_score,
    "retrieval": compute_retrieval_score,
    "qa": compute_retrieval_score,
    # Math
    "openai/gsm8k": compute_math_score,
    "math": compute_math_score,
    "gsm8k": compute_math_score,
    "aime": compute_math_score,
    "numina": compute_math_score,
    # Code
    "code": compute_code_score,
    "codecontests": compute_code_score,
    "apps": compute_code_score,
    "codeforces": compute_code_score,
    "taco": compute_code_score,
}


def igpo_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict[str, Any]:
    """Unified compute_score entry point for IGPO tasks.

    Routes to the appropriate task-specific scorer based on data_source.
    Matching order: exact key → prefix match (longest wins) → fallback to
    retrieval scorer with a warning.

    The returned dict always contains "score" (the primary reward scalar)
    plus task-specific diagnostic keys.

    Args:
        data_source: Dataset identifier string (e.g. "searchR1_nq", "math",
            "codecontests").
        solution_str: Full decoded model response.
        ground_truth: Reference answer.
        extra_info: Optional metadata dict passed through to scorer.

    Returns:
        dict with at least {"score": float}.
    """
    # Exact match first
    if data_source in TASK_SCORE_REGISTRY:
        scorer = TASK_SCORE_REGISTRY[data_source]
        return scorer(solution_str, ground_truth, extra_info)

    # Case-insensitive prefix match (longest matching prefix wins)
    ds_lower = data_source.lower()
    best_key: Optional[str] = None
    for key in TASK_SCORE_REGISTRY:
        if ds_lower.startswith(key.lower()):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is not None:
        return TASK_SCORE_REGISTRY[best_key](solution_str, ground_truth, extra_info)

    # Final fallback: retrieval / F1 scorer with a warning
    import warnings
    warnings.warn(
        f"igpo_compute_score: no scorer registered for data_source={data_source!r}. "
        "Falling back to retrieval F1 scorer.",
        stacklevel=2,
    )
    return compute_retrieval_score(solution_str, ground_truth, extra_info)
