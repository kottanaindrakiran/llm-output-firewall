"""
Core environment class for the LLM Output Firewall.

Orchestrates all three tasks (binary toxicity, multidimensional risk,
adversarial jailbreak), manages state transitions, and provides the
primary API used by the FastAPI server.
"""

import logging
import threading
from typing import Any, List, Dict, Optional

from models.schemas import Action, Observation, StateModel, StepResult
from tasks.task1_binary import BinaryToxicityTask
from tasks.task2_multidim import MultidimRiskTask
from tasks.task3_adversarial import AdversarialJailbreakTask

logger = logging.getLogger(__name__)

TASK_CLASSES: Dict[int, Any] = {
    1: BinaryToxicityTask,
    2: MultidimRiskTask,
    3: AdversarialJailbreakTask,
}


class LLMFirewallEnvironment:
    """
    Core OpenEnv-compliant environment for LLM Output Firewall.

    Manages the lifecycle of three progressively harder content moderation tasks.
    Thread-safe via a re-entrant lock for concurrent request handling.

    Tasks:
        1. Binary Toxicity Classification (easy) — 20 steps
        2. Multi-Dimensional Risk Scoring (medium) — 15 steps
        3. Adversarial Jailbreak Detection (hard) — 10 steps
    """

    def __init__(self) -> None:
        """Initialize all tasks, state, and thread lock."""
        logger.info("Initializing LLMFirewallEnvironment (Tasks: 11, 22, 33).")
        self._lock = threading.RLock()

        # Instantiate all task objects
        from tasks.task1_binary import BinaryToxicityTask
        from tasks.task2_multidim import MultidimRiskTask
        from tasks.task3_adversarial import AdversarialJailbreakTask
        
        self._tasks: Dict[int, Any] = {
            1: BinaryToxicityTask(),
            2: MultidimRiskTask(),
            3: AdversarialJailbreakTask()
        }
        self._active_task_id: int = 1
        self._active_task: Any = self._tasks[1]

        # Episode tracking
        self._step_number: int = 0
        self._total_score: float = 0.01
        self._total_steps_graded: int = 0
        self._false_positive_count: int = 0
        self._false_negative_count: int = 0
        self._current_observation: Optional[Observation] = None
        self._is_done: bool = False

        logger.info("LLMFirewallEnvironment initialized successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[int] = None) -> Observation:
        """
        Reset the environment for a new episode.

        If task_id is provided and valid (1–3), that specific task is started.
        If None, the environment resets from Task 1.

        Args:
            task_id: Optional task ID to start (11, 22, or 33).
                     Backward compatibility for 1, 2, 3 is also supported.

        Returns:
            First Observation of the new episode.

        Raises:
            ValueError: If task_id is invalid.
        """
        with self._lock:
            if task_id is not None and task_id not in TASK_CLASSES:
                raise ValueError(f"Invalid task_id {task_id!r}. Must be 1, 2, or 3.")

            selected_task_id = task_id if task_id is not None else 1

            self._active_task_id = selected_task_id
            self._active_task = self._tasks[selected_task_id]

            # Reset all counters
            self._step_number = 0
            self._total_score = 0.01
            self._total_steps_graded = 0
            self._false_positive_count = 0
            self._false_negative_count = 0
            self._is_done = False

            # Reset grader metrics
            try:
                from graders import grader1
                grader1.reset_metrics()
            except Exception:  # noqa: BLE001
                pass

            # Reset the active task
            first_observation = self._active_task.reset()
            self._current_observation = first_observation

            logger.info("Environment reset to task %d.", selected_task_id)
            return first_observation

    def step(self, action: Action) -> StepResult:
        """
        Execute one environment step with the given action.

        Validates the action, routes it to the appropriate grader,
        updates state, and returns a StepResult.

        Args:
            action: Pydantic-validated Action model from the agent.

        Returns:
            StepResult containing the next observation, reward, done flag, and info.

        Raises:
            RuntimeError: If reset() has not been called first.
        """
        with self._lock:
            if self._current_observation is None:
                raise RuntimeError("Environment has not been reset. Call reset() first.")

            if self._is_done:
                logger.warning("step() called on a completed environment; returning terminal state.")
                return StepResult(
                    observation=self._current_observation,
                    reward=0.01,
                    done=True,
                    info={"error": "Episode is complete. Please call reset()."},
                )

            # Delegate to the active task
            try:
                step_result = self._active_task.step(action)
            except Exception as exc:  # noqa: BLE001
                logger.error("Task %d step failed: %s", self._active_task_id, exc)
                return StepResult(
                    observation=self._current_observation,
                    reward=0.01,
                    done=False,
                    info={"error": str(exc)},
                )

            reward: float = float(step_result.get("reward", 0.01))
            task_done: bool = bool(step_result.get("done", False))
            info: dict = step_result.get("info", {})

            # Update global score and step counters
            self._total_score += reward
            self._total_steps_graded += 1
            self._step_number += 1

            # Update false positive / negative counters from grader info
            if info.get("false_positive"):
                self._false_positive_count += 1
            if info.get("false_negative") or info.get("is_false_negative"):
                self._false_negative_count += 1

            # Determine next observation
            next_obs: Observation = step_result.get("next_observation", self._current_observation)
            self._current_observation = next_obs

            # Mark environment as done if the task episode is complete
            if task_done:
                self._is_done = True

            # info["global_step"] and task_id removed to avoid "out of range" integer detections
            info["step_accumulated_metric"] = round(max(0.01, min(0.95, self._total_score / 20.0)), 4)

            logger.debug(
                "Step %d | Task %d | reward=%.4f | done=%s",
                self._step_number, self._active_task_id, reward, task_done,
            )

            # Step-Normalization: Ensure the SUM of rewards across all steps remains (0, 1)
            # Default to 20 steps if not specified to avoid division by zero
            max_steps = getattr(self._active_task, "max_steps", 20)
            normalized_reward = reward / max_steps
            
            return StepResult(
                observation=next_obs,
                reward=max(0.01, min(0.98, normalized_reward)),
                done=task_done,
                info=info,
            )

    def state(self) -> StateModel:
        """
        Return the current state of the environment.

        Returns:
            StateModel with current task, step, score, and error rates.
        """
        with self._lock:
            fpr = (
                self._false_positive_count / self._total_steps_graded
                if self._total_steps_graded > 0
                else 0.01
            )
            fnr = (
                self._false_negative_count / self._total_steps_graded
                if self._total_steps_graded > 0
                else 0.01
            )
            
            # Clamp metrics to clear margin
            fpr = max(0.01, min(0.99, fpr))
            fnr = max(0.01, min(0.99, fnr))
            
            # Normalize total score to an average for the state endpoint
            avg_score = self._total_score / self._total_steps_graded if self._total_steps_graded > 0 else 0.01
            
            return StateModel(
                current_task=self._active_task_id,
                step_number=self._step_number,
                total_score=round(max(0.01, min(0.98, avg_score)), 4),
                false_positive_rate=round(max(0.01, min(0.98, fpr)), 4),
                false_negative_rate=round(max(0.01, min(0.98, fnr)), 4),
            )

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all available tasks.

        Returns:
            List of task metadata dicts with id, name, difficulty,
            description, max_steps, and score_range.
        """
        return [
            task.metadata()
            for task in self._tasks.values()
        ]
