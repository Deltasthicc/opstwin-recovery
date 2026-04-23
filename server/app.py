"""
OpsTwin Recovery Arena -- FastAPI Server
==========================================
"""
from openenv.core.env_server import create_fastapi_app

from models import OpsAction, OpsObservation
from server.environment import OpsTwinEnvironment

# Single shared environment instance so HTTP callers see a persistent episode.
# The OpenEnv SIMULATION-mode handlers create a fresh env per request by default,
# which makes multi-step interactive demos (HF Space, smoke tests) misbehave
# because every /step sees total_issues=0 and flips done=True on step 1.
# True per-client concurrency is provided via Docker mode or the MCP WebSocket.
_shared_env = OpsTwinEnvironment()

app = create_fastapi_app(lambda: _shared_env, OpsAction, OpsObservation)


def main():
    """Entry point for `uv run server` / pyproject [project.scripts]."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
