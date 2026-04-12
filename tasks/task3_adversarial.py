"""
Task 3: Adversarial Jailbreak Detection (Hard).
"""

import logging
import random
from typing import Any, List, Dict

from data.loader import load_task3_examples
from models.schemas import Observation

logger = logging.getLogger(__name__)

TASK_ID = 3
TASK_NAME = "adversarial_jailbreak_detection"
DIFFICULTY = "hard"
MAX_STEPS = 10

class AdversarialJailbreakTask:
    """
    Orchestrates the Adversarial Jailbreak Detection task (Task 3).
    """

    def __init__(self) -> None:
        """Initialize the task and load adversarial examples."""
        logger.info("Initializing AdversarialJailbreakTask (Task 3).")
        self._all_examples: List[Dict[str, Any]] = load_task3_examples()
        self._episode_examples: List[Dict[str, Any]] = []
        self._current_index: int = 0

    def reset(self) -> Observation:
        """
        Reset the task for a new episode.
        """
        pool = self._all_examples.copy()
        random.shuffle(pool)
        self._episode_examples = pool[:10]
        self._current_index = 0
        return self._get_observation()

    def step(self) -> Observation:
        """
        Advance to the next example in the episode.
        """
        self._current_index += 1
        return self._get_observation()

    def _get_observation(self) -> Observation:
        """
        Build an Observation object for the current step.
        """
        example = self._episode_examples[self._current_index]
        return Observation(
            llm_output=example["text"],
            original_prompt=example.get("original_prompt", ""),
            conversation_context=[],
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
        """
        return {
            "id": TASK_ID,
            "name": TASK_NAME,
            "difficulty": DIFFICULTY,
            "description": "Detection of adversarial jailbreaks.",
            "max_steps": MAX_STEPS,
            "score_range": [0.1, 0.9],
        }
