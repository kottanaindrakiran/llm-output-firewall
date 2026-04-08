"""
FastAPI application for the LLM Output Firewall environment.

Exposes REST endpoints for interacting with the reinforcement learning
environment, including resetting tasks, submitting agent actions, querying
state, and retrieving task metadata.

Runs on host 0.0.0.0 port 7860 (HuggingFace Spaces default).
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import LLMFirewallEnvironment
from models.schemas import Action, Observation, StateModel, StepResult

# Load .env file for local development
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App lifecycle — single shared environment instance
# ---------------------------------------------------------------------------
_env: LLMFirewallEnvironment


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Initialise the shared environment on startup."""
    global _env
    logger.info("Starting LLM Output Firewall API server.")
    _env = LLMFirewallEnvironment()
    yield
    logger.info("Shutting down LLM Output Firewall API server.")


app = FastAPI(
    title="LLM Output Firewall",
    description=(
        "An OpenEnv-compliant reinforcement learning environment where an AI agent "
        "learns to detect and filter toxic, hallucinated, biased, and adversarial "
        "outputs from LLM responses in real-time."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware — allow all origins
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[return]
    """Log every incoming request and its response status/duration."""
    start_time = time.perf_counter()
    logger.info("➜ %s %s", request.method, request.url.path)

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "← %s %s | status=%d | %.1fms",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""

    task_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "LLM Output Firewall",
        "version": "1.0.0",
        "description": "An OpenEnv-compliant RL environment for detecting toxic, hallucinated, biased, and adversarial LLM outputs.",
        "status": "running",
        "tasks": 3,
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "docs": "GET /docs"
        }
    }



@app.get(
    "/health",
    summary="Health check",
    status_code=status.HTTP_200_OK,
)
async def health() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        JSON with status=ok and environment name.
    """
    return {"status": "ok", "environment": "llm-output-firewall"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = Body(None)):
    try:
        task_id = request.task_id if request else None
        observation = _env.reset(task_id=task_id)
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/step",
    response_model=StepResult,
    summary="Submit an agent action",
    status_code=status.HTTP_200_OK,
)
async def step(action: Action) -> StepResult:
    """
    Submit an agent action and receive the step result.

    The action should include:
    - decision: PASS, REWRITE, or BLOCK
    - reasoning: explanation for the decision
    - rewritten_text: required if decision is REWRITE
    - confidence: float between 0.0 and 1.0

    Args:
        action: Validated Action model from request body.

    Returns:
        StepResult with next observation, reward, done flag, and info.
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during step: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process step: {exc}",
        ) from exc


@app.get(
    "/state",
    response_model=StateModel,
    summary="Get current environment state",
    status_code=status.HTTP_200_OK,
)
async def state() -> StateModel:
    """
    Return the current state of the environment.

    Returns:
        StateModel with current_task, step_number, total_score,
        false_positive_rate, and false_negative_rate.
    """
    return _env.state()


@app.get(
    "/tasks",
    summary="List all available tasks",
    status_code=status.HTTP_200_OK,
)
async def get_tasks() -> list[dict[str, Any]]:
    """
    Return metadata for all three available tasks.

    Returns:
        List of task metadata dicts.
    """
    return _env.get_tasks()


# ---------------------------------------------------------------------------
# Application entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
