"""
Service Dependency Graph
========================
Replaces the airport NetworkTracker. Tracks how service-level issues
propagate through a directed dependency graph.

Visible edges come from scenario["services"][*]["dependencies"]. Hidden
edges live in the HiddenStateLayer and are revealed by ASSESS_BLAST_RADIUS;
once revealed, those edges join the graph here.

Cascade alert types (stable across scenarios):
    cascade_risk         : a FAILED/DEGRADED service will propagate to a dependent
    shared_dependency    : two services both depend on the failing component
    policy_block         : taking action X on A triggers review for B
    slo_breach_imminent  : SLO trending toward breach under current load
"""
from typing import Dict, List, Optional, Set, Tuple


class ServiceDependencyGraph:
    """Directed graph. Edge (A, B) means 'A depends on B' (A calls B)."""

    def __init__(self):
        # Adjacency: service -> set of services it depends on
        self._edges: Dict[str, Set[str]] = {}
        self._hidden_reveals: List[Tuple[str, str]] = []

    def reset(self, services: Dict[str, dict]):
        """Build visible graph from scenario services."""
        self._edges = {}
        self._hidden_reveals = []
        for sid, svc in services.items():
            deps = set(svc.get("dependencies", []) or [])
            self._edges[sid] = deps

    def add_hidden_edge(self, a: str, b: str):
        """Called when HiddenStateLayer.assess_blast_radius reveals an edge."""
        if a not in self._edges:
            self._edges[a] = set()
        self._edges[a].add(b)
        self._hidden_reveals.append((a, b))

    def dependents_of(self, sid: str) -> List[str]:
        """Services that depend on sid (reverse edges)."""
        return [s for s, deps in self._edges.items() if sid in deps]

    def dependencies_of(self, sid: str) -> List[str]:
        return list(self._edges.get(sid, set()))

    def compute_cascade_alerts(self, services: Dict[str, dict]) -> List[Dict]:
        """Compute current cascade risks based on service statuses."""
        alerts: List[Dict] = []

        # Cascade risk: an unhealthy service with dependents
        for sid, svc in services.items():
            status = svc.get("status", "HEALTHY")
            if status in ("FAILED", "DEGRADED"):
                deps = self.dependents_of(sid)
                for d in deps:
                    d_status = services.get(d, {}).get("status", "HEALTHY")
                    if d_status == "HEALTHY":
                        # Healthy service depending on an unhealthy one
                        alerts.append({
                            "type": "cascade_risk",
                            "severity": "high" if status == "FAILED" else "medium",
                            "source_service": sid,
                            "affected_service": d,
                            "reason": f"{d} depends on {sid} ({status}). Cascade within minutes.",
                        })

        # Shared dependency: two HEALTHY services both depend on an unhealthy one
        # (Already surfaced by cascade_risk above. Skip for clarity.)

        # SLO breach imminent
        for sid, svc in services.items():
            slo_target = svc.get("slo_target", 1.0)
            slo_curr = svc.get("current_slo", 1.0)
            if slo_curr < slo_target - 0.001 and svc.get("status", "HEALTHY") != "FAILED":
                alerts.append({
                    "type": "slo_breach_imminent",
                    "severity": "high" if (slo_target - slo_curr) > 0.01 else "medium",
                    "service": sid,
                    "reason": f"{sid} SLO {slo_curr:.4f} < target {slo_target:.4f}",
                })

        return alerts

    def render_topology(self, services: Dict[str, dict]) -> str:
        """Human-readable text view of the current graph."""
        if not services:
            return "No services registered."
        lines = ["Services:"]
        for sid, svc in sorted(services.items()):
            status = svc.get("status", "HEALTHY")
            slo = svc.get("current_slo")
            slo_str = f" slo={slo:.4f}" if isinstance(slo, float) else ""
            lines.append(f"  {sid} [{status}]{slo_str}")
        lines.append("Visible dependencies:")
        any_edge = False
        for sid in sorted(self._edges):
            for dep in sorted(self._edges[sid]):
                marker = " (hidden)" if (sid, dep) in self._hidden_reveals else ""
                lines.append(f"  {sid} -> {dep}{marker}")
                any_edge = True
        if not any_edge:
            lines.append("  (none)")
        return "\n".join(lines)

    @property
    def edges(self) -> Dict[str, Set[str]]:
        return self._edges
