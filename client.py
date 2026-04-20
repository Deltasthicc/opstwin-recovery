"""
OpsTwin Recovery Arena -- EnvClient
=====================================
Thin subclass of openenv.core.env_client.EnvClient wired for OpsTwin types.
"""
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import OpsAction, OpsObservation, OpsState


class OpsTwinRecoveryEnv(EnvClient[OpsAction, OpsObservation, OpsState]):
    """Client for the OpsTwin server."""

    def _step_payload(self, action: OpsAction) -> dict:
        return {"command": action.command}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {}) or {}
        return StepResult(
            observation=OpsObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                **{k: v for k, v in obs_data.items()
                   if k in OpsObservation.model_fields
                   and k not in ("done", "reward")}
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> OpsState:
        return OpsState(**{k: v for k, v in payload.items()
                           if k in OpsState.model_fields})
