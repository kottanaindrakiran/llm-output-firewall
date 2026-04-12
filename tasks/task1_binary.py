"""
Task 1: Binary Toxicity Classification (Easy).
"""

import logging
import random
from typing import Any, List, Dict

from data.loader import load_task1_examples
from models.schemas import Observation

logger = logging.getLogger(__name__)

TASK_ID = 1
TASK_NAME = "binary_toxicity_classification"
DIFFICULTY = "easy"
MAX_STEPS = 20

class BinaryToxicityTask:
    """
    Orchestrates the Binary Toxicity Classification task (Task 1).
    """

    def __init__(self) -> None:
        """Initialize the task and load dataset examples."""
        logger.info("Initializing BinaryToxicityTask (Task 1).")
        self._all_examples: List[Dict[str, Any]] = load_task1_examples()
        self._episode_examples: List[Dict[str, Any]] = []
        self._current_index: int = 0

    def reset(self) -> Observation:
        """
        Reset the task for a new episode.
        """
        toxic = [e for e in self._all_examples if e["label"] == "BLOCK"]
        safe = [e for e in self._all_examples if e["label"] == "PASS"]
        
        self._episode_examples = random.sample(toxic, 10) + random.sample(safe, 10)
        random.shuffle(self._episode_examples)
        self._current_index = 0
        return self._get_observation()

    def step(self) -> Observation:
        """Advance to next example."""
        self._current_index += 1
        return self._get_observation()

    def _get_observation(self) -> Observation:
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
        return self._current_index >= (MAX_STEPS - 1)

    @property
    def current_example(self) -> Dict[str, Any]:
        return self._episode_examples[self._current_index]

    @property
    def max_steps(self) -> int:
        return MAX_STEPS

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": TASK_ID,
            "name": TASK_NAME,
            "difficulty": DIFFICULTY,
            "description": "Binary toxicity classification.",
            "max_steps": MAX_STEPS,
            "score_range": [0.1, 0.9],
        }
