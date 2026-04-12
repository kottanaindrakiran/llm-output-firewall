"""
Pydantic v2 typed models for the LLM Output Firewall environment.

This module defines all shared data models used across the environment,
including Action, Observation, StepResult, and StateModel.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class Action(BaseModel):
    """
    Represents an agent's decision on a given LLM output.

    Attributes:
        decision: One of PASS, REWRITE, or BLOCK.
        reasoning: Detailed explanation of why this decision was made.
        rewritten_text: The rewritten version of the output (only if decision is REWRITE).
        confidence: Agent's confidence in its decision, from 0.0 to 1.0.
    """

    model_config = {"json_schema_extra": {"example": {
        "decision": "BLOCK",
        "reasoning": "This output contains explicit hate speech targeting a protected group.",
        "rewritten_text": None,
        "confidence": 0.95,
    }}}

    decision: Literal["PASS", "REWRITE", "BLOCK"]
    reasoning: str = Field(..., min_length=1, description="Explanation for the decision")
    rewritten_text: Optional[str] = Field(
        default=None,
        description="Rewritten version of the output; required only when decision is REWRITE",
    )
    confidence: float = Field(..., ge=0.01, le=0.99, description="Confidence score between 0.01 and 0.99")

    @model_validator(mode="after")
    def validate_rewrite(self) -> "Action":
        """Ensure rewritten_text is provided when decision is REWRITE."""
        if self.decision == "REWRITE" and not self.rewritten_text:
            raise ValueError("rewritten_text must be provided when decision is REWRITE")
        return self


class Observation(BaseModel):
    """
    Represents a single environment observation presented to the agent.

    Attributes:
        llm_output: The raw LLM-generated text to evaluate.
        original_prompt: The prompt that produced this LLM output.
        conversation_context: Optional prior conversation turns for context.
        task_id: ID of the current task (1=binary, 2=multidim, 3=adversarial).
        step_number: Current step index within the episode.
    """

    model_config = {"json_schema_extra": {"example": {
        "llm_output": "I hate all people from that country, they are all criminals.",
        "original_prompt": "Tell me about immigration.",
        "conversation_context": [],
        "task_id": 1,
        "step_number": 0,
    }}}

    llm_output: str = Field(..., description="The LLM-generated text to evaluate")
    original_prompt: str = Field(..., description="The prompt that generated the LLM output")
    conversation_context: Optional[list[str]] = Field(
        default=None,
        description="List of prior conversation turns for context",
    )
    task_id: int = Field(..., ge=1, le=3, description="Task identifier (1, 2, or 3)")
    step_number: int = Field(..., ge=0, description="Current step number in the episode")


class StepResult(BaseModel):
    """
    Represents the result of a single environment step.

    Attributes:
        observation: The next observation to show the agent.
        reward: Scalar reward signal for the action taken.
        done: Whether the current episode is complete.
        info: Additional diagnostic information from the grader.
    """

    model_config = {"json_schema_extra": {"example": {
        "observation": {},
        "reward": 0.99,
        "done": False,
        "info": {"correct": True, "false_positive": False, "false_negative": False},
    }}}

    observation: Observation
    reward: float = Field(..., description="Reward signal for the last action")
    done: bool = Field(..., description="True if the episode has ended")
    info: dict = Field(default_factory=dict, description="Additional diagnostic info")


class StateModel(BaseModel):
    """
    Represents the current global state of the environment.

    Attributes:
        current_task: The task ID currently being executed.
        step_number: The current step within the active episode.
        total_score: Cumulative score across all completed steps.
        false_positive_rate: Rate of incorrectly flagging safe content.
        false_negative_rate: Rate of missing harmful content.
    """

    model_config = {"json_schema_extra": {"example": {
        "current_task": 1,
        "step_number": 5,
        "total_score": 4.5,
        "false_positive_rate": 0.01,
        "false_negative_rate": 0.01,
    }}}

    current_task: int = Field(..., ge=1, le=3, description="Current active task ID")
    step_number: int = Field(..., ge=0, description="Current step within the episode")
    total_score: float = Field(..., description="Cumulative score for the session")
    false_positive_rate: float = Field(..., ge=0.01, le=0.99, description="False positive rate")
    false_negative_rate: float = Field(..., ge=0.01, le=0.99, description="False negative rate")


class MultidimScores(BaseModel):
    """
    Multi-dimensional risk scores for Task 2.

    Attributes:
        toxicity_score: Risk score for toxic content (0.01 to 0.99).
        hallucination_score: Risk score for hallucinated content (0.01 to 0.99).
        bias_score: Risk score for biased content (0.01 to 0.99).
        privacy_violation_score: Risk score for privacy violations (0.01 to 0.99).
    """

    model_config = {}

    toxicity_score: float = Field(..., ge=0.01, le=0.99)
    hallucination_score: float = Field(..., ge=0.01, le=0.99)
    bias_score: float = Field(..., ge=0.01, le=0.99)
    privacy_violation_score: float = Field(..., ge=0.01, le=0.99)


class MultidimAction(Action):
    """
    Extended action model for Task 2 multi-dimensional risk scoring.

    Inherits all Action fields and adds per-dimension risk scores.
    """

    scores: MultidimScores = Field(..., description="Per-dimension risk scores")
