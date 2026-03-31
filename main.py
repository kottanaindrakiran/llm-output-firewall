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
from fastapi import FastAPI, HTTPException, Request, status
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

# ---------------------------------------------------------------------------
# Dashboard Template (Vanilla CSS + High Aesthetics)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Output Firewall | Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: hsl(220, 30%, 3%);
            --bg-card: hsla(220, 25%, 10%, 0.7);
            --border: hsla(220, 20%, 20%, 1);
            --accent-primary: hsl(0, 100%, 60%); /* Firewall Red */
            --accent-secondary: hsl(30, 100%, 50%); /* Orange */
            --accent-green: hsl(140, 100%, 50%);
            --text-main: hsl(220, 10%, 90%);
            --text-dim: hsl(220, 10%, 60%);
            --glass-bg: hsla(220, 25%, 10%, 0.4);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-deep);
            background-image: 
                radial-gradient(circle at 10% 20%, hsla(0, 100%, 50%, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, hsla(220, 100%, 50%, 0.05) 0%, transparent 40%);
            color: var(--text-main);
            line-height: 1.6;
            min-height: 100vh;
            overflow-x: hidden;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 3rem 1.5rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4rem;
            position: relative;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge {
            background: hsla(140, 100%, 50%, 0.1);
            color: var(--accent-green);
            padding: 0.4rem 0.8rem;
            border-radius: 2rem;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid hsla(140, 100%, 50%, 0.2);
            animation: pulse-green 2s infinite;
        }

        @keyframes pulse-green {
            0% { box-shadow: 0 0 0 0 hsla(140, 100%, 50%, 0.4); }
            70% { box-shadow: 0 0 0 10px hsla(140, 100%, 50%, 0); }
            100% { box-shadow: 0 0 0 0 hsla(140, 100%, 50%, 0); }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 4rem;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 1.5rem;
            padding: 2rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            border-color: hsla(0, 100%, 60%, 0.5);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(to right, transparent, var(--accent-primary), transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .card:hover::before {
            opacity: 1;
        }

        .card-title {
            color: var(--text-dim);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .card-value {
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            margin-bottom: 1rem;
        }

        .card-footer {
            font-size: 0.875rem;
            color: var(--text-dim);
        }

        .task-list {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .task-item {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 1.5rem;
            padding: 1.25rem;
            background: var(--glass-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            transition: background 0.2s;
        }

        .task-item.active {
            border-color: var(--accent-secondary);
            background: hsla(30, 100%, 50%, 0.05);
        }

        .task-id {
            width: 40px;
            height: 40px;
            border-radius: 0.75rem;
            background: var(--bg-deep);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            border: 1px solid var(--border);
        }

        .task-info h3 {
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
        }

        .task-info p {
            font-size: 0.875rem;
            color: var(--text-dim);
        }

        .task-difficulty {
            font-size: 0.75rem;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .diff-easy { color: #4ade80; background: rgba(74, 222, 128, 0.1); }
        .diff-medium { color: #fbbf24; background: rgba(251, 191, 36, 0.1); }
        .diff-hard { color: #f87171; background: rgba(248, 113, 113, 0.1); }

        .api-explorer {
            margin-top: 4rem;
        }

        .api-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
        }

        .api-tag {
            font-family: monospace;
            padding: 1rem;
            background: #000;
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }

        .method {
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 0.3rem;
            font-size: 0.7rem;
        }

        .method-get { color: #60a5fa; background: rgba(96, 165, 250, 0.1); }
        .method-post { color: #c084fc; background: rgba(192, 132, 252, 0.1); }

        .btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: var(--text-main);
            color: var(--bg-deep);
            text-decoration: none;
            border-radius: 0.75rem;
            font-weight: 600;
            transition: all 0.2s;
            margin-top: 2rem;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 255, 255, 0.2);
        }

        footer {
            margin-top: 6rem;
            text-align: center;
            color: var(--text-dim);
            font-size: 0.875rem;
            padding-bottom: 2rem;
        }

        @media (max-width: 768px) {
            header { flex-direction: column; gap: 1.5rem; text-align: center; }
            .card-value { font-size: 2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                🛡️ LLM OUTPUT FIREWALL
            </div>
            <div class="badge">SYSTEM ONLINE</div>
        </header>

        <div class="grid">
            <div class="card">
                <p class="card-title">Environment Health</p>
                <div class="card-value" style="color: var(--accent-green);">PASSED</div>
                <p class="card-footer">Uptime monitoring active</p>
            </div>
            <div class="card">
                <p class="card-title">Cumulative Reward</p>
                <div class="card-value">{total_score}</div>
                <p class="card-footer">Total performance score across all steps</p>
            </div>
            <div class="card">
                <p class="card-title">Error Rates</p>
                <div class="card-footer" style="margin-top: 1rem; display: flex; flex-direction: column; gap: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>False Positive Rate</span>
                        <span style="color: var(--accent-secondary);">{fpr}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>False Negative Rate</span>
                        <span style="color: var(--accent-primary);">{fnr}%</span>
                    </div>
                </div>
            </div>
        </div>

        <h2 style="margin-bottom: 2rem;">Firewall Evaluation Tasks</h2>
        <div class="task-list">
            {task_items}
        </div>

        <div class="api-explorer">
            <h2 style="margin-bottom: 2rem;">API Control Interface</h2>
            <div class="api-grid">
                <div class="api-tag"><span class="method method-get">GET</span> /health<span>200 OK</span></div>
                <div class="api-tag"><span class="method method-post">POST</span> /reset<span>INIT</span></div>
                <div class="api-tag"><span class="method method-post">POST</span> /step<span>ACTION</span></div>
                <div class="api-tag"><span class="method method-get">GET</span> /state<span>STATE</span></div>
                <div class="api-tag"><span class="method method-get">GET</span> /tasks<span>META</span></div>
            </div>
            <a href="/docs" class="btn">View Interactive API Docs</a>
        </div>

        <footer>
            <p>&copy; 2026 LLM Output Firewall &bull; OpenEnv Compliant</p>
        </footer>
    </div>
</body>
</html>
"""

@app.get("/", summary="Environment Dashboard")
async def root(request: Request) -> Any:
    """
    Return the high-end dashboard OR environment info depending on Accept header.
    """
    accept = request.headers.get("accept", "")
    
    # If the request explicitly asks for JSON, or is from a tool like curl (often defaults)
    if "text/html" not in accept and "application/json" in accept:
        return {
            "name": "LLM Output Firewall",
            "version": "1.0.0",
            "description": "An OpenEnv-compliant RL environment for training AI agents.",
            "tasks": 3,
            "status": "online",
            "documentation": "/docs",
        }

    # Otherwise, serve the beautiful dashboard
    from fastapi.responses import HTMLResponse
    
    state = _env.state()
    tasks = _env.get_tasks()
    
    task_html_template = """
    <div class="task-item {active_class}">
        <div class="task-id">{id}</div>
        <div class="task-info">
            <h3>{name}</h3>
            <p>{description}</p>
        </div>
        <div class="task-difficulty diff-{diff_lower}">{difficulty}</div>
    </div>
    """
    
    task_items = []
    for t in tasks:
        active_class = "active" if state.current_task == t["id"] else ""
        task_items.append(task_html_template.format(
            active_class=active_class,
            id=t["id"],
            name=t["name"],
            description=t["description"],
            difficulty=t["difficulty"],
            diff_lower=t["difficulty"].lower()
        ))
    
    content = DASHBOARD_HTML.format(
        total_score=state.total_score,
        fpr=round(state.false_positive_rate * 100, 2),
        fnr=round(state.false_negative_rate * 100, 2),
        task_items="\\n".join(task_items)
    )
    
    return HTMLResponse(content=content)



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


@app.post(
    "/reset",
    response_model=Observation,
    summary="Reset the environment",
    status_code=status.HTTP_200_OK,
)
async def reset(body: ResetRequest) -> Observation:
    """
    Reset the environment and return the first observation.

    - If task_id is provided (1, 2, or 3), that specific task is started.
    - If task_id is omitted or null, Task 1 is started by default.

    Args:
        body: ResetRequest with optional task_id.

    Returns:
        First Observation of the new episode.
    """
    try:
        observation = _env.reset(task_id=body.task_id)
        logger.info("Environment reset to task %s.", body.task_id)
        return observation
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during reset: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset environment: {exc}",
        ) from exc


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
