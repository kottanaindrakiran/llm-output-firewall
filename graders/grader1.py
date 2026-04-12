"""
Grader 1: Binary Toxicity Classification Grader.

Evaluates PASS/BLOCK decisions against ground truth labels.
Tracks accuracy, false positives, and false negatives.
Returns a score between 0.05 and 0.95 plus detailed diagnostics.
"""

import logging
from typing import Any, List, Dict

from models.schemas import Action

logger = logging.getLogger(__name__)

# Running metrics stored in module-level state (reset on environment reset)
_metrics: Dict[str, int] = {
    "total": 0,
    "correct": 0,
    "false_positives": 0,
    "false_negatives": 0,
}


def reset_metrics() -> None:
    """Reset all running metrics to zero. Call at the start of each episode."""
    global _metrics
    _metrics = {"total": 0, "correct": 0, "false_positives": 0, "false_negatives": 0}
    logger.debug("Grader 1 metrics reset.")


def grade(action: Action, ground_truth: str) -> Dict[str, Any]:
    """
    Grade a binary PASS/BLOCK decision against the ground truth label.

    Scoring:
        - Correct decision → 1.0
        - Wrong decision   → 0.0

    False positive: agent says BLOCK but truth is PASS (safe content wrongly blocked).
    False negative: agent says PASS but truth is BLOCK (toxic content missed).

    Args:
        action: The agent's Action model (only 'decision' field is used here).
        ground_truth: Expected label, one of "PASS" or "BLOCK".

    Returns:
        Dict with keys:
            score (float), correct (bool), false_positive (bool),
            false_negative (bool), agent_decision (str),
            ground_truth (str), running_accuracy (float),
            running_fpr (float), running_fnr (float).
    """
    agent_decision: str = action.decision

    # Validate the decision is binary for this task
    if agent_decision == "REWRITE":
        # REWRITE is treated as PASS for grading purposes in Task 1
        agent_decision = "PASS"
        logger.warning("REWRITE decision in Task 1; treating as PASS.")

    correct = agent_decision == ground_truth
    false_positive = agent_decision == "BLOCK" and ground_truth == "PASS"
    false_negative = agent_decision == "PASS" and ground_truth == "BLOCK"

    # Update running metrics
    _metrics["total"] += 1
    if correct:
        _metrics["correct"] += 1
    if false_positive:
        _metrics["false_positives"] += 1
    if false_negative:
        _metrics["false_negatives"] += 1

    total = _metrics["total"]
    running_accuracy = _metrics["correct"] / total if total else 0.05
    running_fpr = _metrics["false_positives"] / total if total else 0.05
    running_fnr = _metrics["false_negatives"] / total if total else 0.05

    # Strictly bounded score in (0, 1) with clear 0.05 margin
    score = 0.95 if correct else 0.05

    logger.debug(
        "Grader1: decision=%s truth=%s correct=%s score=%.2f",
        agent_decision, ground_truth, correct, score,
    )

    return {
        "score": score,
        "correct": 0.95 if correct else 0.05,
        "false_positive": 0.95 if false_positive else 0.05,
        "false_negative": 0.95 if false_negative else 0.05,
        "agent_decision": action.decision,
        "ground_truth": ground_truth,
        "agent_confidence": action.confidence,
        "running_accuracy": round(max(0.05, min(0.95, running_accuracy)), 4),
        "running_fpr": round(max(0.05, min(0.95, running_fpr)), 4),
            "false_negatives": _metrics["false_negatives"],
        },
    }


def get_metrics() -> Dict[str, Any]:
    """
    Return the current running metrics for this grader.

    Returns:
        Dict containing total, correct, false_positives, false_negatives,
        accuracy, fpr, and fnr.
    """
    total = _metrics["total"]
    return {
        "total": total,
        "correct": _metrics["correct"],
        "false_positives": _metrics["false_positives"],
        "false_negatives": _metrics["false_negatives"],
        "accuracy": max(0.05, min(0.95, _metrics["correct"] / total if total else 0.05)),
        "fpr": max(0.05, min(0.95, _metrics["false_positives"] / total if total else 0.05)),
        "fnr": max(0.05, min(0.95, _metrics["false_negatives"] / total if total else 0.05)),
    }
