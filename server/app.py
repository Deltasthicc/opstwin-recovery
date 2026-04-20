"""
OpsTwin Recovery Arena -- FastAPI Server
==========================================
"""
from openenv.core.env_server import create_fastapi_app

from models import OpsAction, OpsObservation
from server.environment import OpsTwinEnvironment

app = create_fastapi_app(OpsTwinEnvironment, OpsAction, OpsObservation)


def main():
    """Entry point for `uv run server` / pyproject [project.scripts]."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
