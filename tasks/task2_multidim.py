"""
Task 2: Multi-Dimensional Risk Scoring (Medium).

The agent receives an LLM output plus context and must score four risk
dimensions (toxicity, hallucination, bias, privacy) from 0.0 to 1.0, then
provide a final PASS / REWRITE / BLOCK decision. 15 examples per episode.
"""

import logging
import random
from typing import Any, List, Dict

from data.loader import load_task2_examples
from models.schemas import Observation

logger = logging.getLogger(__name__)

TASK_ID = 2
TASK_NAME = "multidimensional_risk_scoring"
DIFFICULTY = "medium"
MAX_STEPS = 15

class MultidimRiskTask:
    """
    Orchestrates the Multi-Dimensional Risk Scoring task (Task 2).
    """

    def __init__(self) -> None:
        """Initialize the task and load dataset examples."""
        logger.info("Initializing MultidimRiskTask (Task 2).")
        self._all_examples: List[Dict[str, Any]] = load_task2_examples()
        self._episode_examples: List[Dict[str, Any]] = []
        self._current_index: int = 0

    def reset(self) -> Observation:
        """
        Reset the task for a new episode.

        Selects 15 random examples from the pool and returns the first observation.

        Returns:
            Observation for the first step of the episode.
        """
        if len(self._all_examples) < MAX_STEPS:
            logger.warning("Not enough Task 2 examples; recycling.")
            self._episode_examples = random.choices(self._all_examples, k=MAX_STEPS)
        else:
            self._episode_examples = random.sample(self._all_examples, k=MAX_STEPS)
            
        self._current_index = 0
        return self._get_observation()

    def step(self) -> Observation:
        """
        Advance to the next example in the episode.

        Returns:
            Observation for the next step.
        """
        self._current_index += 1
        return self._get_observation()

    def _get_observation(self) -> Observation:
        """
        Build an Observation object for the current step.

        Returns:
            Current observation.
        """
        example = self._episode_examples[self._current_index]
        context = example.get("conversation_context", [])
        if not isinstance(context, list):
            context = []
            
        return Observation(
            llm_output=example["text"],
            original_prompt=example.get("original_prompt", ""),
            conversation_context=list(context),
            task_id=TASK_ID,
            step_number=self._current_index,
        )

    @property
    def is_done(self) -> bool:
        """Check if the episode is finished."""
        return self._current_index >= (MAX_STEPS - 1)

    @property
    def current_example(self) -> Dict[str, Any]:
        """Get the current ground truth example."""
        return self._episode_examples[self._current_index]

    @property
    def max_steps(self) -> int:
        """Get the max steps for this task."""
        return MAX_STEPS

    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata description for this task.

        Returns:
            Metadata dictionary.
        """
        return {
            "id": TASK_ID,
            "name": TASK_NAME,
            "difficulty": DIFFICULTY,
            "description": "Multi-dimensional risk scoring across 4 dimensions.",
            "max_steps": MAX_STEPS,
            "score_range": [0.05, 0.95],
        }
