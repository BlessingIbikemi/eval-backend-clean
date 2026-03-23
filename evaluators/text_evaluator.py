"""
text_evaluator.py
-----------------
Computes evaluation metrics. All run instantly on CPU — no downloads needed.

Metrics:
  ROUGE-1, ROUGE-2, ROUGE-L  →  lexical overlap (standard for QA/summarization)
  Exact Match                →  binary — did the model get it exactly right
  Token Precision            →  of words the model said, how many were correct
  Token Recall               →  of words in the reference, how many did model cover
  Token F1                   →  harmonic mean of precision and recall (SQuAD standard)
  Length stats               →  word counts and length ratio
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens."""
    return re.findall(r"\w+", text.lower())


def _token_prf(prediction: str, reference: str) -> tuple[float, float, float]:
    """
    Token-level Precision, Recall, F1.
    Same approach used in the SQuAD QA benchmark.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter  = Counter(ref_tokens)

    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0, 0.0, 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(ref_tokens)
    f1        = 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_text(prediction: str, reference: str) -> dict:
    prediction = prediction.strip()
    reference  = reference.strip()
    result     = {}

    # ── ROUGE ──────────────────────────────────────────────────────────────
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        result["rouge1_f"] = round(scores["rouge1"].fmeasure, 4)
        result["rouge2_f"] = round(scores["rouge2"].fmeasure, 4)
        result["rougeL_f"] = round(scores["rougeL"].fmeasure, 4)
    except ImportError:
        result["rouge1_f"] = result["rouge2_f"] = result["rougeL_f"] = None
        result["rouge_note"] = "Install rouge-score: pip install rouge-score"

    # ── Exact Match ────────────────────────────────────────────────────────
    result["exact_match"] = prediction.lower() == reference.lower()

    # ── Token Precision, Recall, F1 (SQuAD-style) ─────────────────────────
    precision, recall, f1 = _token_prf(prediction, reference)
    result["token_precision"] = precision
    result["token_recall"]    = recall
    result["token_f1"]        = f1

    # ── Length Stats ───────────────────────────────────────────────────────
    pred_words = len(_tokenize(prediction))
    ref_words  = len(_tokenize(reference))
    result["prediction_word_count"] = pred_words
    result["reference_word_count"]  = ref_words
    result["length_ratio"]          = round(pred_words / ref_words, 3) if ref_words > 0 else 0.0

    return result


def evaluate_batch(pairs: list[dict]) -> list[dict]:
    """pairs = [{"prompt":..., "prediction":..., "reference":...}, ...]"""
    return [
        {**p, "metrics": evaluate_text(p["prediction"], p["reference"])}
        for p in pairs
    ]


def aggregate(results: list[dict]) -> dict:
    """Average all numeric metrics across a batch."""
    if not results:
        return {}
    metric_keys = [k for k, v in results[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    agg = {}
    for key in metric_keys:
        vals = [r[key] for r in results if r.get(key) is not None]
        agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4) if vals else None
    agg["n_samples"] = len(results)
    return agg