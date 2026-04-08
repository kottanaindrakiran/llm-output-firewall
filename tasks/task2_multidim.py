"""
Task 2: Multi-Dimensional Risk Scoring (Medium).

The agent receives an LLM output plus context and must score four risk
dimensions (toxicity, hallucination, bias, privacy) from 0.0 to 1.0, then
provide a final PASS / REWRITE / BLOCK decision. 15 examples per episode.
"""

import logging
import random
from typing import Any

from data.loader import load_task2_examples
from models.schemas import Observation

logger = logging.getLogger(__name__)

TASK_ID = 2
TASK_NAME = "multidimensional_risk_scoring"
DIFFICULTY = "medium"
MAX_STEPS = 15
DESCRIPTION = (
    "Multi-dimensional analysis task where the agent scores four risk dimensions "
    "(toxicity, hallucination, bias, privacy) and makes a final PASS/REWRITE/BLOCK decision."
)

DIMENSION_WEIGHTS = {
    "toxicity_score": 0.35,
    "hallucination_score": 0.30,
    "bias_score": 0.20,
    "privacy_violation_score": 0.15,
}


class MultidimRiskTask:
    """
    Task 2: Multi-Dimensional Risk Scoring.

    Each episode presents 15 examples covering medical misinformation,
    gender bias, PII leakage, and confident-but-wrong statements.
    Reward is the weighted average of per-dimension accuracy scores.
    """

    def __init__(self) -> None:
        """Initialize the task and pre-load synthetic examples."""
        logger.info("Initializing MultidimRiskTask (Task 2).")
        self._all_examples: list[dict[str, Any]] = load_task2_examples()
        self._episode_examples: list[dict[str, Any]] = []
        self._current_index: int = 0

    def reset(self) -> Observation:
        """
        Reset the task for a new episode.

        Randomly samples 15 examples from the pool and returns the first observation.

        Returns:
            Observation for the first step of the episode.
        """
        pool = self._all_examples.copy()
        random.shuffle(pool)
        self._episode_examples = pool[:15]
        self._current_index = 0

        logger.debug("Task 2 reset: %d examples for this episode.", len(self._episode_examples))
        return self._make_observation()

    def current_observation(self) -> Observation:
        """Return the current observation without advancing the step counter."""
        return self._make_observation()

    def step(self, action: Any) -> dict[str, Any]:
        """
        Process a multi-dimensional action and return grading info.

        Args:
            action: An Action or MultidimAction containing scores and decision.

        Returns:
            Dict with keys: reward, done, info, next_observation.
        """
        if self._current_index >= len(self._episode_examples):
            return {
                "reward": 0.0001,
                "done": True,
                "info": {"error": "Episode already complete."},
                "next_observation": self._make_observation(),
            }

        current_example = self._episode_examples[self._current_index]
        gold_scores: dict[str, float] = current_example.get("gold_scores", {})

        # Import grader inline to avoid circular imports
        from graders.grader2 import grade  # noqa: PLC0415

        grader_result = grade(action=action, gold_scores=gold_scores)

        self._current_index += 1
        done = self._current_index >= len(self._episode_examples)

        return {
            "reward": grader_result["score"],
            "done": done,
            "info": grader_result,
            "next_observation": self._make_observation(),
        }

    def _make_observation(self) -> Observation:
        """Build an Observation for the current step index."""
        if self._current_index >= len(self._episode_examples):
            last = self._episode_examples[-1] if self._episode_examples else {}
            return Observation(
                llm_output=last.get("text", ""),
                original_prompt="Episode complete.",
                conversation_context=[],
                task_id=TASK_ID,
                step_number=self._current_index,
            )

        example = self._episode_examples[self._current_index]
        context = example.get("conversation_context", []) or []

        return Observation(
            llm_output=example["text"],
            original_prompt=example.get("original_prompt", ""),
            conversation_context=list(context),
            task_id=TASK_ID,
            step_number=self._current_index,
        )

    @property
    def is_done(self) -> bool:
        """Return True if the current episode has been fully completed."""
        return self._current_index >= len(self._episode_examples)

    @staticmethod
    def metadata() -> dict[str, Any]:
        """Return static metadata about this task."""
        return {
            "id": TASK_ID,
            "name": TASK_NAME,
            "difficulty": DIFFICULTY,
            "max_steps": MAX_STEPS,
            "score_range": [0.01, 0.99],
            "description": DESCRIPTION,
            "valid_decisions": ["PASS", "REWRITE", "BLOCK"],
            "dimensions": list(DIMENSION_WEIGHTS.keys()),
            "dimension_weights": DIMENSION_WEIGHTS,
        }
