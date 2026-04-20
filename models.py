"""
OpsTwin Recovery Arena -- Pydantic Type Definitions
====================================================
Action, Observation, and State inherit from openenv.core.env_server base
classes. Observation already provides `done: bool` and `reward: Optional[float]`;
State already provides `episode_id` and `step_count`.
"""
from typing import Dict, List, Optional
from openenv.core.env_server import Action, Observation, State


class OpsAction(Action):
    """Agent sends a single text command each step."""
    command: str


class OpsObservation(Observation):
    """Full OpsTwin state returned after reset() or step()."""
    current_time: str = ""
    incident_severity: int = 1
    incident_description: str = ""
    ops_status: str = ""
    services: List[Dict] = []
    tickets: List[Dict] = []
    pipelines: List[Dict] = []
    alerts: List[Dict] = []
    triage_queue: List[Dict] = []
    graph_alerts: List[Dict] = []
    uncertainty_alerts: List[Dict] = []
    active_desk: str = ""
    desk_messages: List[Dict] = []
    pending_issues_count: int = 0
    resolved_issues_count: int = 0
    total_issues_count: int = 0
    message: str = ""
    available_commands: List[str] = []
    multi_objective_scores: Dict[str, float] = {}
    score: float = 0.0
    hint: str = ""
    memory_hints: List[str] = []


class OpsState(State):
    """Episode metadata. Inherits episode_id + step_count from base State."""
    task_name: str = ""
    disruption_type: str = ""
    total_services: int = 0
    resolved_issues: int = 0
    total_issues: int = 0
    max_steps: int = 0
