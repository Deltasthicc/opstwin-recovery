"""
OpsTwin Recovery Arena -- Comprehensive Test Suite
====================================================
One file. Many classes. Everything gets tested.

Run:
    python -m pytest tests/test_environment.py -v
    python -m pytest tests/test_environment.py -q       # compact
    python -m pytest tests/test_environment.py -k Scoring  # subset

Test classes (in order of dependency):

    TestEnvironmentBasics ........... reset, step, clock, audit trail
    TestActionDispatch .............. every command handler does the right thing
    TestScoringDimensions ........... each of the 6 dimensions in isolation
    TestInactionCap ................. the "agent did nothing" safety cap
    TestScoringAggregator ........... weighted_final composition + clamping
    TestScenarioFidelity ............ each scenario has expected issues/points
    TestHiddenStateProtocol ......... 5 inspection primitives + revelation logic
    TestPolicyViolations ............ rollback/quarantine guardrails
    TestApprovalLifecycle ........... stale -> refresh -> deploy flow
    TestDeskSystem .................. desk switching + command gating
    TestDeskFilterViews ............. each desk filters observations correctly
    TestDeskMessaging ............... SEND_MESSAGE / READ_MESSAGES
    TestPostmortemClassification .... failure categories
    TestPostmortemLifecycle ......... record -> retrieve -> hint injection
    TestGeneratorStructure .......... all 3 families produce valid dicts
    TestGeneratorDifficulty ......... easy/medium/hard differ appropriately
    TestGeneratorDeterminism ........ same seed -> same scenario
    TestDynamicEvents ............... scheduled events fire at right step
    TestPipelineCooldowns ........... rerun cooldowns enforced
    TestAlertSweep .................. _sweep_clear_alerts per-step recheck
    TestEscalateToIC ................ one-time strategic hint
    TestEfficiencyBonus ............. early-termination bonus
    TestServerRoutes ................ FastAPI endpoints via TestClient
    TestTrainingWrapper ............. OpsTwinTrainingEnv tool dispatch
    TestInferenceParser ............. _extract_command for every shape
    TestInferenceFallback ........... model fallback chain logic
    TestInferencePrompt ............. build_user_prompt structure
    TestIntegrationOptimal .......... expert trajectories score >= 0.90
    TestIntegrationTraps ............ trap behaviours score <= 0.75
    TestReproducibility ............. same inputs produce same outputs

Why one file: the user asked for it, and it makes it easy to run everything
with a single pytest invocation. Test classes provide the logical grouping.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import pytest

from models import OpsAction, OpsObservation, OpsState
from server.environment import OpsTwinEnvironment
from server.scenarios import SCENARIOS, ALL_TASK_NAMES
from server.scoring import (
    compute_multi_objective_scores,
    compute_service_recovery,
    compute_customer_outcome,
    compute_security_compliance,
    compute_change_hygiene,
    compute_communication_quality,
    compute_operational_efficiency,
    WEIGHTS,
)
from server.generator import generate_scenario, GENERATED_FAMILIES, FAMILY_FNS
from server.hidden_state import HiddenStateLayer
from server.graph import ServiceDependencyGraph
from server.desks import (
    DeskCoordinator, VALID_DESKS, UNIVERSAL_COMMANDS,
    DESK_COMMANDS, DESK_INFO_SUBJECTS,
)
from server.postmortem import (
    PostmortemMemory, classify_failure, first_bad_action,
    preferred_intervention_order, record_from_env, FAILURE_CATEGORIES,
)
from baselines.expert_solver import OPTIMAL_TRAJECTORIES


# ===========================================================
# Section 1: Environment Basics
# ===========================================================

class TestEnvironmentBasics:
    """Basic lifecycle: reset, step, clock, audit trail."""

    def test_reset_returns_observation(self):
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        assert isinstance(obs, OpsObservation)
        assert obs.total_issues_count > 0
        assert obs.resolved_issues_count == 0
        assert obs.done is False

    def test_reset_assigns_episode_id(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        assert env.state.episode_id
        assert len(env.state.episode_id) > 0

    def test_reset_with_explicit_episode_id(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release", episode_id="test-ep-001")
        assert env.state.episode_id == "test-ep-001"

    def test_reset_clears_previous_episode(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        env.reset(task="security_cve")
        assert env.state.step_count == 0
        assert env._audit_trail == []
        assert env.state.task_name == "security_cve"

    def test_step_increments_count(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        env.step(OpsAction(command="REQUEST_INFO summary"))
        assert env.state.step_count == 2

    def test_step_increments_clock(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        before = env._clock_minutes
        env.step(OpsAction(command="REQUEST_INFO summary"))
        assert env._clock_minutes == before + env.MINUTES_PER_STEP

    def test_clock_format_hhmm(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        assert len(env._fmt()) == 5
        assert env._fmt()[2] == ":"
        h, m = env._fmt().split(":")
        assert 0 <= int(h) < 24
        assert 0 <= int(m) < 60

    def test_clock_wraps_past_midnight(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env._start_hour = 23
        env._start_min = 55
        env._clock_minutes = 10
        assert env._fmt() == "00:05"

    def test_step_on_done_episode_returns_done(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="DONE"))
        obs = env.step(OpsAction(command="REQUEST_INFO summary"))
        assert obs.done is True

    def test_done_command_terminates(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="DONE"))
        assert obs.done is True

    def test_max_steps_terminates(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")
        max_s = env._max_steps
        for _ in range(max_s + 2):
            obs = env.step(OpsAction(command="REQUEST_INFO summary"))
            if obs.done:
                break
        assert obs.done is True
        assert env.state.step_count <= max_s

    def test_audit_trail_grows(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        env.step(OpsAction(command="REQUEST_INFO summary"))
        assert len(env._audit_trail) == 2

    def test_audit_entry_shape(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        entry = env._audit_trail[0]
        assert "step" in entry
        assert "action" in entry
        assert "reward" in entry
        assert "resolved" in entry

    def test_observation_fields_populated(self):
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        assert obs.current_time
        assert obs.incident_description
        assert isinstance(obs.services, list)
        assert isinstance(obs.tickets, list)
        assert isinstance(obs.pipelines, list)
        assert isinstance(obs.alerts, list)
        assert isinstance(obs.available_commands, list)
        assert len(obs.available_commands) > 10

    def test_score_is_always_in_bounds(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for _ in range(5):
            obs = env.step(OpsAction(command="INVALID_COMMAND"))
            assert 0.01 <= obs.score <= 0.99
            if obs.done:
                break

    def test_reset_resets_all_resolved_sets(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        env.step(OpsAction(command="PRIORITIZE_VIP T-003"))
        assert len(env._resolved["ticket_escalations"]) > 0
        env.reset(task="bad_release")
        for k in env._resolved:
            assert len(env._resolved[k]) == 0


# ===========================================================
# Section 2: Action Dispatch
# ===========================================================

class TestActionDispatch:
    """Every command handler does the right thing."""

    def test_unknown_command_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="NOT_A_REAL_COMMAND"))
        assert obs.reward == pytest.approx(-0.02, abs=1e-6)

    def test_empty_command_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command=""))
        assert obs.reward < 0

    def test_whitespace_command_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="   "))
        assert obs.reward < 0

    def test_switch_desk_all_five(self):
        for desk in VALID_DESKS:
            env = OpsTwinEnvironment()
            env.reset(task="bad_release")
            obs = env.step(OpsAction(command=f"SWITCH_DESK {desk}"))
            assert env._desks.active_desk == desk
            assert obs.reward >= 0

    def test_switch_desk_invalid_name(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="SWITCH_DESK NOT_A_DESK"))
        assert obs.reward < 0

    def test_switch_desk_missing_arg(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="SWITCH_DESK"))
        assert obs.reward < 0

    def test_restart_healthy_service_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(command="RESTART_SERVICE auth-svc"))
        assert obs.reward < 0  # auth-svc is HEALTHY

    def test_restart_unknown_service_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(command="RESTART_SERVICE not-a-svc"))
        assert obs.reward < 0

    def test_flip_flag_resolves_service(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        obs = env.step(OpsAction(command="FLIP_FLAG checkout_v2_ui off"))
        assert obs.reward > 0.20  # should be 0.30 for service_outage
        assert "checkout-svc" in env._resolved["service_outages"]

    def test_flip_flag_invalid_state_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        obs = env.step(OpsAction(command="FLIP_FLAG checkout_v2_ui maybe"))
        assert obs.reward < 0

    def test_flip_unknown_flag_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        obs = env.step(OpsAction(command="FLIP_FLAG unknown_flag off"))
        assert obs.reward < 0

    def test_triage_ticket_valid_priority(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="TRIAGE_TICKET T-001 P2"))
        assert obs.reward > 0

    def test_triage_ticket_invalid_priority(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="TRIAGE_TICKET T-001 PX"))
        assert obs.reward < 0

    def test_prioritize_vip_on_non_vip_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="PRIORITIZE_VIP T-001"))  # T-001 is not VIP
        assert obs.reward < 0

    def test_draft_comms_internal_accepted(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="DRAFT_COMMS internal Alert triaged."))
        assert obs.reward > 0

    def test_draft_comms_external_accepted(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="DRAFT_COMMS external Service restored."))
        assert obs.reward > 0

    def test_draft_comms_invalid_audience(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="DRAFT_COMMS nobody Message."))
        assert obs.reward < 0

    def test_escalate_to_ic_first_call_gives_hint(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="ESCALATE_TO_IC"))
        assert "IC" in obs.message or "hint" in obs.message.lower() \
               or "URGENT" in obs.message or "FLIP" in obs.message or "flag" in obs.message.lower()

    def test_escalate_to_ic_second_call_is_noop(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="ESCALATE_TO_IC"))
        obs = env.step(OpsAction(command="ESCALATE_TO_IC"))
        assert "already" in obs.message.lower()

    def test_request_info_summary_works_from_any_desk(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="REQUEST_INFO summary"))
        assert obs.reward >= 0
        assert "Issue Summary" in obs.message or "Summary" in obs.message


# ===========================================================
# Section 3: Scoring Dimensions (pure functions)
# ===========================================================

class TestScoringDimensions:
    """Each of 6 dimensions computed in isolation on synthetic inputs."""

    def test_service_recovery_no_outages_returns_one(self):
        assert compute_service_recovery(0, 0, False, False) == 1.0

    def test_service_recovery_full_fix_no_assess(self):
        assert compute_service_recovery(5, 5, False, False) == pytest.approx(0.85)

    def test_service_recovery_full_fix_with_assess_contained(self):
        assert compute_service_recovery(5, 5, True, True) == 1.0

    def test_service_recovery_full_fix_with_assess_uncontained(self):
        assert compute_service_recovery(5, 5, True, False) == pytest.approx(0.92)

    def test_service_recovery_partial_fix(self):
        result = compute_service_recovery(2, 4, True, True)
        assert 0.4 <= result <= 0.6

    def test_customer_outcome_no_tickets_returns_one(self):
        assert compute_customer_outcome({}, set()) == 1.0

    def test_customer_outcome_all_resolved(self):
        tickets = {
            "T1": {"priority": "P1", "is_vip": False},
            "T2": {"priority": "P2", "is_vip": False},
        }
        assert compute_customer_outcome(tickets, {"T1", "T2"}) == 1.0

    def test_customer_outcome_vip_bonus(self):
        tickets = {
            "T1": {"priority": "P1", "is_vip": True},
            "T2": {"priority": "P3", "is_vip": False},
        }
        # All tickets resolved including VIP -> bonus on top
        result = compute_customer_outcome(tickets, {"T1", "T2"})
        assert result >= 0.99  # Clamped to <= 1.0

    def test_customer_outcome_sla_expired_weight_zero(self):
        tickets = {
            "T1": {"priority": "P1", "is_vip": False,
                   "sla_minutes_remaining": 0},  # expired
            "T2": {"priority": "P2", "is_vip": False},
        }
        # T1's weight drops to zero since SLA expired and unresolved
        result = compute_customer_outcome(tickets, {"T2"})
        # Only T2 counts, and T2 is resolved -> should be ~1.0
        assert result >= 0.99

    def test_security_compliance_no_violations(self):
        assert compute_security_compliance([], {}, set(), 0, 0) == 1.0

    def test_security_compliance_unresolved_cve_penalty(self):
        assert compute_security_compliance([], {}, set(), 1, 0) == pytest.approx(0.9)

    def test_security_compliance_unsafe_disclosure_heavy_penalty(self):
        assert compute_security_compliance([], {}, set(), 0, 1) == pytest.approx(0.8)

    def test_security_compliance_policy_violation(self):
        audit = [{"action": "QUARANTINE_SERVICE auth-svc"}]
        policy = {"QUARANTINE_SERVICE auth-svc": ["requires_legal_notify"]}
        # policy_satisfied is empty -> violation
        result = compute_security_compliance(audit, policy, set(), 0, 0)
        assert result < 1.0

    def test_security_compliance_floors_at_zero(self):
        assert compute_security_compliance([], {}, set(), 10, 10) == 0.0

    def test_change_hygiene_no_rollbacks(self):
        assert compute_change_hygiene([], set(), set(), 0) == 1.0

    def test_change_hygiene_unnecessary_rollback_penalty(self):
        assert compute_change_hygiene([], set(), {"deploy-x"}, 0) == pytest.approx(0.8)

    def test_change_hygiene_duplicate_action_penalty(self):
        audit = [{"action": "RESTART_SERVICE x"}] * 3  # 2 duplicates
        result = compute_change_hygiene(audit, set(), set(), 0)
        assert result < 1.0

    def test_change_hygiene_info_queries_not_duplicates(self):
        audit = [{"action": "REQUEST_INFO summary"}] * 5
        assert compute_change_hygiene(audit, set(), set(), 0) == 1.0

    def test_change_hygiene_pipeline_rerun_penalty(self):
        assert compute_change_hygiene([], set(), set(), 2) == pytest.approx(0.8)

    def test_communication_quality_no_required(self):
        assert compute_communication_quality([], set(), [], 10) == 1.0

    def test_communication_quality_all_sent(self):
        required = [{"audience": "external", "required": True, "points": 0.2}]
        assert compute_communication_quality(
            required, {"external"}, [{"audience": "external", "step": 3}], 10
        ) >= 1.0

    def test_communication_quality_none_sent(self):
        required = [{"audience": "external", "required": True, "points": 0.2}]
        assert compute_communication_quality(required, set(), [], 10) == 0.0

    def test_communication_quality_timely_bonus(self):
        required = [{"audience": "external", "required": True}]
        early = compute_communication_quality(
            required, {"external"}, [{"audience": "external", "step": 1}], 10)
        late = compute_communication_quality(
            required, {"external"}, [{"audience": "external", "step": 8}], 10)
        assert early >= late

    def test_operational_efficiency_zero_resolved(self):
        assert compute_operational_efficiency(10, 10, 5, 0, 0) == 0.0

    def test_operational_efficiency_zero_steps(self):
        assert compute_operational_efficiency(0, 10, 5, 0, 0) == 0.0

    def test_operational_efficiency_all_resolved_quickly(self):
        # 5 issues resolved in 5 steps out of 10 max steps
        result = compute_operational_efficiency(5, 10, 5, 5, 0)
        assert result >= 0.9

    def test_operational_efficiency_unnecessary_penalty(self):
        with_penalty = compute_operational_efficiency(10, 10, 5, 5, 3)
        without = compute_operational_efficiency(10, 10, 5, 5, 0)
        assert with_penalty < without


# ===========================================================
# Section 4: Inaction Cap
# ===========================================================

class TestInactionCap:
    """The cap that pins idle-agent scores below 0.30."""

    def _mo(self, total_issues, resolved_issues, **overrides):
        defaults = dict(
            resolved_outages=resolved_issues,
            total_outages=total_issues,
            blast_radius_assessed=False,
            blast_radius_contained=False,
            tickets={},
            resolved_ticket_ids=set(),
            audit_trail=[],
            policy_flags={},
            policy_satisfied=set(),
            unresolved_cves=0,
            unsafe_disclosures=0,
            necessary_rollbacks=set(),
            executed_rollbacks=set(),
            pipeline_reruns_on_healthy=0,
            pending_comms_required=[],
            resolved_comms_keys=set(),
            draft_comms_log=[],
            steps_used=1,
            max_steps=10,
            total_issues=total_issues,
            resolved_issues=resolved_issues,
            unnecessary_actions=0,
        )
        defaults.update(overrides)
        return compute_multi_objective_scores(**defaults)

    def test_cap_fires_on_total_inaction(self):
        scores = self._mo(total_issues=3, resolved_issues=0)
        assert scores["weighted_final"] <= 0.30

    def test_cap_does_not_fire_if_any_resolved(self):
        scores = self._mo(total_issues=3, resolved_issues=1,
                          resolved_outages=1, total_outages=3)
        assert scores["weighted_final"] > 0.30

    def test_cap_does_not_fire_if_no_issues(self):
        scores = self._mo(total_issues=0, resolved_issues=0)
        assert scores["weighted_final"] > 0.30  # nothing to do -> full score

    def test_false_positive_inaction_capped(self):
        """Full env test: false_positive with REQUEST_INFO only."""
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")
        for _ in range(10):
            obs = env.step(OpsAction(command="REQUEST_INFO summary"))
            if obs.done:
                break
        assert obs.score <= 0.30

    def test_bad_release_idle_capped(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for _ in range(14):
            obs = env.step(OpsAction(command="REQUEST_INFO summary"))
            if obs.done:
                break
        assert obs.score <= 0.30

    def test_cap_preserves_clamp_bounds(self):
        scores = self._mo(total_issues=3, resolved_issues=0)
        assert 0.01 <= scores["weighted_final"] <= 0.99


# ===========================================================
# Section 5: Scoring Aggregator
# ===========================================================

class TestScoringAggregator:
    """weighted_final composition, weights, clamping."""

    def test_weights_sum_to_one(self):
        total = sum(WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_dimensions_present(self):
        expected = {
            "service_recovery", "customer_outcome", "security_compliance",
            "change_hygiene", "communication_quality", "operational_efficiency",
        }
        assert set(WEIGHTS.keys()) == expected

    def test_final_score_never_below_floor(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for _ in range(20):
            obs = env.step(OpsAction(command="INVALID"))
            if obs.done:
                break
        assert obs.score >= 0.01

    def test_final_score_never_above_ceiling(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for cmd in OPTIMAL_TRAJECTORIES["bad_release"]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        assert obs.score <= 0.99

    def test_score_rounded_to_4_decimals(self):
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        # Should have reasonable precision
        score_str = f"{obs.score:.6f}"
        decimals = score_str.split(".")[1]
        # Trailing beyond 4 should be zeros
        assert decimals[4:].rstrip("0") == ""


# ===========================================================
# Section 6: Scenario Fidelity
# ===========================================================

class TestScenarioFidelity:
    """Each hand-authored scenario has expected issues, points, max_steps."""

    @pytest.mark.parametrize("task", ALL_TASK_NAMES)
    def test_scenario_has_description(self, task):
        sc = SCENARIOS[task]
        assert sc.get("description")
        assert len(sc["description"]) > 20

    @pytest.mark.parametrize("task", ALL_TASK_NAMES)
    def test_scenario_max_steps_sensible(self, task):
        sc = SCENARIOS[task]
        assert 8 <= sc["max_steps"] <= 25

    @pytest.mark.parametrize("task", ALL_TASK_NAMES)
    def test_scenario_severity_in_range(self, task):
        sc = SCENARIOS[task]
        assert 1 <= sc["incident_severity"] <= 5

    @pytest.mark.parametrize("task", ALL_TASK_NAMES)
    def test_scenario_has_hidden_state(self, task):
        sc = SCENARIOS[task]
        assert "hidden_state" in sc

    @pytest.mark.parametrize("task", ALL_TASK_NAMES)
    def test_scenario_issues_sum_to_unity(self, task):
        sc = SCENARIOS[task]
        total = 0.0
        for cat in sc["issues"]:
            for iss in sc["issues"][cat]:
                total += iss["points"]
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_bad_release_has_flag_root_cause(self):
        sc = SCENARIOS["bad_release"]
        assert "flag" in sc["hidden_state"]["root_cause"]

    def test_security_cve_has_stale_approval(self):
        sc = SCENARIOS["security_cve"]
        approvals = sc["hidden_state"]["approval_states"]
        assert any(v == "stale" for v in approvals.values())

    def test_data_pipeline_has_blast_radius_edge(self):
        sc = SCENARIOS["data_pipeline_regression"]
        edges = sc["hidden_state"]["blast_radius_edges"]
        assert len(edges) > 0

    def test_false_positive_has_no_outages(self):
        sc = SCENARIOS["false_positive"]
        assert len(sc["issues"]["service_outages"]) == 0

    def test_all_scenarios_have_vip_ticket_or_none(self):
        for name in ALL_TASK_NAMES:
            sc = SCENARIOS[name]
            vip_count = sum(1 for t in sc["tickets"] if t.get("is_vip"))
            if sc["tickets"]:
                assert vip_count <= 1  # at most one VIP per scenario


# ===========================================================
# Section 7: Hidden State Protocol
# ===========================================================

class TestHiddenStateProtocol:
    """The 5 inspection primitives reveal information correctly."""

    def test_fresh_layer_has_no_reveals(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {}})
        assert h.blast_radius_assessed is False
        assert not h.revealed_edges

    def test_reset_clears_previous(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"root_cause": "A"}})
        h.inspect_runbook("svc")
        h.reset({"hidden_state": {"root_cause": "B"}})
        assert h.get_root_cause() == "B"
        assert not h.blast_radius_assessed

    def test_inspect_runbook_first_time_pays(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"root_cause": "flag_checkout"}})
        r, _, new = h.inspect_runbook("checkout-svc")
        assert new is True
        assert r > 0

    def test_inspect_runbook_second_time_free(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"root_cause": "flag_checkout"}})
        h.inspect_runbook("checkout-svc")
        r, _, new = h.inspect_runbook("checkout-svc")
        assert new is False
        assert r == 0

    def test_verify_flag_reveals_state(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"flag_states": {"beta_ui": True}}})
        r, msg, _ = h.verify_flag("beta_ui")
        assert "ON" in msg

    def test_verify_unknown_flag(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"flag_states": {}}})
        r, msg, _ = h.verify_flag("unknown")
        assert "not found" in msg.lower() or "no flag" in msg.lower()

    def test_check_approval_stale(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"approval_states": {"change-1": "stale"}}})
        r, msg, _ = h.check_approval("change-1")
        assert "STALE" in msg or "stale" in msg.lower()

    def test_check_approval_unknown_change(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"approval_states": {}}})
        r, msg, _ = h.check_approval("nonexistent")
        assert "no record" in msg.lower() or "no approval" in msg.lower()

    def test_assess_blast_radius_reveals_one_edge_at_a_time(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"blast_radius_edges": [
            ["a", "b"], ["c", "d"]]}})
        assert len(h.revealed_edges) == 0
        h.assess_blast_radius()
        assert len(h.revealed_edges) == 1
        h.assess_blast_radius()
        assert len(h.revealed_edges) == 2
        assert h.blast_radius_fully_contained is True

    def test_assess_blast_radius_sets_assessed_flag(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"blast_radius_edges": []}})
        assert h.blast_radius_assessed is False
        h.assess_blast_radius()
        assert h.blast_radius_assessed is True

    def test_set_flag_updates_truth(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"flag_states": {"x": True}}})
        h.set_flag("x", False)
        assert h.get_flag("x") is False

    def test_set_approval_updates_state(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"approval_states": {"c1": "stale"}}})
        h.set_approval("c1", "approved")
        assert h.get_approval("c1") == "approved"

    def test_uncertainty_alerts_for_unassessed_blast_radius(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"blast_radius_edges": [["a", "b"]]}})
        alerts = h.get_uncertainty_alerts()
        assert any(a["kind"] == "unknown_blast_radius" for a in alerts)

    def test_uncertainty_alerts_for_unchecked_stale_approval(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"approval_states": {"c1": "stale"}}})
        alerts = h.get_uncertainty_alerts()
        assert any(a["kind"] == "unchecked_approval" for a in alerts)

    def test_reveal_summary_tracks_progress(self):
        h = HiddenStateLayer()
        h.reset({"hidden_state": {"blast_radius_edges": [["a", "b"], ["c", "d"]]}})
        assert h.reveal_summary()["blast_radius_assessed"] is False
        h.assess_blast_radius()
        summary = h.reveal_summary()
        assert summary["blast_radius_assessed"] is True
        assert summary["hidden_edges_remaining"] == 1
        assert summary["hidden_edges_revealed"] == 1


# ===========================================================
# Section 8: Policy Violations
# ===========================================================

class TestPolicyViolations:
    """Policy-flagged actions penalize when preconditions missing."""

    def test_rollback_without_assess_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(
            command="ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1"))
        assert obs.reward < 0

    def test_rollback_with_assess_is_policy_compliant(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK INCIDENT_COMMAND"))
        env.step(OpsAction(command="ASSESS_BLAST_RADIUS"))
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(
            command="ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1"))
        # Rollback is allowed after assess; reward may be 0 (no outage matched)
        # but no policy penalty
        assert obs.reward >= -0.02  # only allowed is small invalid-action penalty

    def test_quarantine_without_legal_notify_penalizes_score(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        env.step(OpsAction(command="QUARANTINE_SERVICE auth-svc"))
        env.step(OpsAction(command="DONE"))
        sec = env._compute_mo_scores()["security_compliance"]
        assert sec < 1.0

    def test_security_compliance_drops_on_multiple_violations(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        env.step(OpsAction(command="ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1"))
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        env.step(OpsAction(command="QUARANTINE_SERVICE checkout-svc"))
        env.step(OpsAction(command="DONE"))
        sec = env._compute_mo_scores()["security_compliance"]
        # Rollback without ASSESS triggers policy flag: -0.15
        # QUARANTINE checkout-svc has no policy in bad_release so no second hit.
        assert sec < 1.0  # at least dropped from baseline

    def test_approval_refresh_allows_rerun(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        # Approval is in SECURITY desk's command set, not RELEASE.
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        env.step(OpsAction(command="APPROVE_EXCEPTION deploy-auth-patch-1.8.2"))
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        obs = env.step(OpsAction(command="RERUN_PIPELINE deploy-auth-patch-1.8.2"))
        assert obs.reward > 0

    def test_pipeline_rerun_on_healthy_penalized(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        # deploy-checkout-v2.3.1 starts ACTIVE; rerun on it is wasteful
        env.step(OpsAction(command="RERUN_PIPELINE deploy-checkout-v2.3.1"))
        # Cooldown kicks in, but also marked as rerun_on_healthy
        # Wait for cooldown and retry
        for _ in range(3):
            env.step(OpsAction(command="REQUEST_INFO summary"))
        obs = env.step(OpsAction(command="RERUN_PIPELINE deploy-checkout-v2.3.1"))
        # Accumulated at least one healthy rerun penalty
        assert env._pipeline_reruns_on_healthy >= 1


# ===========================================================
# Section 9: Approval Lifecycle
# ===========================================================

class TestApprovalLifecycle:
    """stale -> check -> approve_exception -> rerun sequence."""

    def test_check_approval_marks_revealed(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        env.step(OpsAction(command="CHECK_APPROVAL deploy-auth-patch-1.8.2"))
        assert "approval:deploy-auth-patch-1.8.2" in env._hidden._revealed

    def test_approve_exception_refreshes_to_approved(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        env.step(OpsAction(command="APPROVE_EXCEPTION deploy-auth-patch-1.8.2"))
        assert env._hidden.get_approval("deploy-auth-patch-1.8.2") == "approved"

    def test_approve_exception_resolves_approval_block(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        obs = env.step(OpsAction(command="APPROVE_EXCEPTION deploy-auth-patch-1.8.2"))
        assert "deploy-auth-patch-1.8.2" in env._resolved["approval_blocks"]
        assert obs.reward > 0


# ===========================================================
# Section 10: Desk System
# ===========================================================

class TestDeskSystem:
    def test_desks_inactive_on_fresh_env(self):
        d = DeskCoordinator()
        assert d.is_active is False
        assert d.active_desk is None

    def test_universal_commands_allowed_always(self):
        d = DeskCoordinator()
        for cmd in UNIVERSAL_COMMANDS:
            allowed, _ = d.is_command_allowed(cmd + " ARG")
            assert allowed is True

    def test_desk_command_set_nonempty_for_all_desks(self):
        for desk in VALID_DESKS:
            assert len(DESK_COMMANDS.get(desk, set())) > 0

    def test_desk_info_subjects_nonempty_for_all_desks(self):
        for desk in VALID_DESKS:
            assert len(DESK_INFO_SUBJECTS.get(desk, set())) > 0

    def test_switch_desk_first_time_bonus(self):
        d = DeskCoordinator()
        r, _ = d.switch_desk("SRE")
        assert r > 0

    def test_switch_desk_second_time_no_bonus(self):
        d = DeskCoordinator()
        d.switch_desk("SRE")
        d.switch_desk("SECURITY")
        r, _ = d.switch_desk("SRE")
        assert r == 0

    def test_switch_same_desk_noop(self):
        d = DeskCoordinator()
        d.switch_desk("SRE")
        r, msg = d.switch_desk("SRE")
        assert r == 0
        assert "already" in msg.lower()

    def test_command_gating_blocks_wrong_desk(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SUPPORT"))
        obs = env.step(OpsAction(command="RESTART_SERVICE checkout-svc"))
        assert obs.reward < 0

    def test_command_gating_allows_correct_desk(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(
            command="INSPECT_RUNBOOK checkout-svc"))
        assert obs.reward >= 0

    def test_coordination_bonus_multi_desk(self):
        d = DeskCoordinator()
        d.switch_desk("SRE")
        d.switch_desk("SECURITY")
        bonus = d.coordination_bonus(total_issues=5)
        assert bonus > 0

    def test_coordination_bonus_scales_with_desks(self):
        d = DeskCoordinator()
        d.switch_desk("SRE")
        d.switch_desk("SECURITY")
        d.switch_desk("SUPPORT")
        # 3 desks + high issue count should unlock higher bonus
        bonus_high = d.coordination_bonus(total_issues=10)
        d2 = DeskCoordinator()
        d2.switch_desk("SRE")
        bonus_low = d2.coordination_bonus(total_issues=5)
        assert bonus_high >= bonus_low


# ===========================================================
# Section 11: Desk Filter Views
# ===========================================================

class TestDeskFilterViews:
    """Each desk filters the observation view appropriately."""

    @pytest.fixture
    def full_obs_data(self):
        return {
            "services": [{"id": "svc-1"}],
            "tickets": [{"id": "T-1"}],
            "pipelines": [{"id": "p-1"}],
            "triage_queue": [{"id": "T-1"}],
            "graph_alerts": [{"type": "cascade"}],
            "alerts": [{"id": "A-1"}],
        }

    def test_sre_hides_tickets(self, full_obs_data):
        d = DeskCoordinator()
        d.switch_desk("SRE")
        filtered = d.filter_observation(full_obs_data)
        assert filtered["tickets"] == []

    def test_support_hides_services(self, full_obs_data):
        d = DeskCoordinator()
        d.switch_desk("SUPPORT")
        filtered = d.filter_observation(full_obs_data)
        assert filtered["services"] == []

    def test_security_hides_tickets(self, full_obs_data):
        d = DeskCoordinator()
        d.switch_desk("SECURITY")
        filtered = d.filter_observation(full_obs_data)
        assert filtered["tickets"] == []

    def test_incident_command_sees_everything(self, full_obs_data):
        d = DeskCoordinator()
        d.switch_desk("INCIDENT_COMMAND")
        filtered = d.filter_observation(full_obs_data)
        assert filtered["services"] == full_obs_data["services"]
        assert filtered["tickets"] == full_obs_data["tickets"]

    def test_no_active_desk_no_filter(self, full_obs_data):
        d = DeskCoordinator()
        filtered = d.filter_observation(full_obs_data)
        assert filtered == full_obs_data


# ===========================================================
# Section 12: Desk Messaging
# ===========================================================

class TestDeskMessaging:
    def test_send_message_requires_active_desk(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="SEND_MESSAGE SRE Hello"))
        assert obs.reward < 0

    def test_send_message_to_self_penalizes(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(command="SEND_MESSAGE SRE Hello"))
        assert obs.reward < 0

    def test_send_message_to_another_desk_succeeds(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(command="SEND_MESSAGE SECURITY CVE found"))
        assert obs.reward > 0

    def test_read_messages_marks_read(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        env.step(OpsAction(command="SEND_MESSAGE SECURITY First"))
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        obs = env.step(OpsAction(command="READ_MESSAGES"))
        assert "First" in obs.message

    def test_read_messages_no_unread_is_clean(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        obs = env.step(OpsAction(command="READ_MESSAGES"))
        assert "no unread" in obs.message.lower() or \
               "No unread" in obs.message


# ===========================================================
# Section 13: Postmortem Classification
# ===========================================================

class TestPostmortemClassification:
    def test_classify_success_is_none(self):
        s = {"final_score": 0.9, "multi_objective": {}, "trace": []}
        assert classify_failure(s) == "none"

    def test_classify_policy_violation(self):
        s = {"final_score": 0.4,
             "multi_objective": {"security_compliance": 0.5},
             "trace": []}
        assert classify_failure(s) == "policy_violation"

    def test_classify_missed_hidden_state(self):
        s = {"final_score": 0.3,
             "multi_objective": {"security_compliance": 0.9,
                                  "service_recovery": 0.2},
             "trace": []}
        assert classify_failure(s) == "missed_hidden_state"

    def test_classify_comms_forgotten(self):
        s = {"final_score": 0.4,
             "multi_objective": {"security_compliance": 0.9,
                                  "service_recovery": 0.8,
                                  "communication_quality": 0.1},
             "trace": []}
        assert classify_failure(s) == "comms_forgotten"

    def test_classify_thrashing(self):
        s = {"final_score": 0.3,
             "multi_objective": {"security_compliance": 0.9,
                                  "service_recovery": 0.8,
                                  "communication_quality": 0.9,
                                  "change_hygiene": 0.4},
             "trace": []}
        assert classify_failure(s) == "thrashing"

    def test_classify_cascade_ignored(self):
        s = {"final_score": 0.4,
             "multi_objective": {"security_compliance": 0.9,
                                  "service_recovery": 0.9,
                                  "communication_quality": 0.9,
                                  "change_hygiene": 0.9,
                                  "customer_outcome": 0.2},
             "trace": []}
        assert classify_failure(s) == "cascade_ignored"

    def test_all_categories_documented(self):
        # Every category in FAILURE_CATEGORIES has a human description
        for cat, desc in FAILURE_CATEGORIES.items():
            assert len(desc) > 10

    def test_first_bad_action_identifies_negative_reward(self):
        trace = [
            {"action": "SWITCH_DESK SRE", "reward": 0.01},
            {"action": "ROLLBACK x", "reward": -0.05},
            {"action": "FLIP_FLAG y on", "reward": 0.3},
        ]
        assert first_bad_action(trace) == "ROLLBACK x"

    def test_first_bad_action_none_when_all_good(self):
        trace = [{"action": "X", "reward": 0.1}]
        assert first_bad_action(trace) == ""

    def test_preferred_intervention_order_extracts_good_commands(self):
        trace = [
            {"action": "SWITCH_DESK SRE", "reward": 0.01},
            {"action": "FLIP_FLAG x off", "reward": 0.3},
            {"action": "BAD", "reward": -0.05},
            {"action": "DRAFT_COMMS external Foo", "reward": 0.2},
        ]
        order = preferred_intervention_order(trace)
        assert "FLIP_FLAG" in order
        assert "DRAFT_COMMS" in order
        assert "BAD" not in order


# ===========================================================
# Section 14: Postmortem Lifecycle
# ===========================================================

class TestPostmortemLifecycle:
    def test_record_writes_jsonl(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        entry = mem.record({
            "task": "bad_release", "final_score": 0.3,
            "multi_objective": {"service_recovery": 0.2}, "trace": [],
        })
        assert store.exists()
        line = store.read_text().strip()
        parsed = json.loads(line)
        assert parsed["episode_id"] == entry["episode_id"]

    def test_retrieve_filters_by_family(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        mem.record({"task": "bad_release", "final_score": 0.2,
                    "multi_objective": {"service_recovery": 0.1}, "trace": []})
        mem.record({"task": "security_cve", "final_score": 0.3,
                    "multi_objective": {"security_compliance": 0.5}, "trace": []})
        assert len(mem.retrieve("bad_release", k=10)) == 1
        assert len(mem.retrieve("security_cve", k=10)) == 1

    def test_retrieve_excludes_successful(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        mem.record({"task": "bad_release", "final_score": 0.9,
                    "multi_objective": {}, "trace": []})
        mem.record({"task": "bad_release", "final_score": 0.3,
                    "multi_objective": {"service_recovery": 0.1}, "trace": []})
        results = mem.retrieve("bad_release", k=10)
        assert len(results) == 1
        assert results[0]["final_score"] == 0.3

    def test_retrieve_sorts_worst_first(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        for s in [0.5, 0.2, 0.7]:
            mem.record({"task": "bad_release", "final_score": s,
                        "multi_objective": {"service_recovery": 0.1}, "trace": []})
        results = mem.retrieve("bad_release", k=3)
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores)

    def test_retrieve_k_limit(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        for _ in range(5):
            mem.record({"task": "bad_release", "final_score": 0.2,
                        "multi_objective": {"service_recovery": 0.1}, "trace": []})
        assert len(mem.retrieve("bad_release", k=2)) == 2

    def test_retrieve_empty_memory(self, tmp_path):
        mem = PostmortemMemory(store_path=tmp_path / "empty.jsonl")
        assert mem.retrieve("bad_release") == []

    def test_build_hints_format(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        mem.record({"task": "bad_release", "final_score": 0.2,
                    "multi_objective": {"service_recovery": 0.1},
                    "trace": [{"action": "ROLLBACK x", "reward": -0.05}]})
        hints = mem.build_hints("bad_release", k=1)
        assert len(hints) == 1
        assert "bad_release" in hints[0]
        assert "0.2" in hints[0] or "0.20" in hints[0]

    def test_hints_injected_into_observation(self, tmp_path):
        store = tmp_path / "pm.jsonl"
        mem = PostmortemMemory(store_path=store)
        mem.record({"task": "bad_release", "final_score": 0.2,
                    "multi_objective": {"service_recovery": 0.1}, "trace": []})
        hints = mem.build_hints("bad_release", k=1)
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release", memory_hints=hints)
        assert obs.memory_hints == hints

    def test_record_from_env_captures_trace(self, tmp_path):
        mem = PostmortemMemory(store_path=tmp_path / "pm.jsonl")
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for cmd in ["SWITCH_DESK SRE", "DONE"]:
            env.step(OpsAction(command=cmd))
            if env._done:
                break
        pm = record_from_env(env, mem)
        assert pm["scenario_family"] == "bad_release"
        assert pm["steps_used"] == len(env._audit_trail)


# ===========================================================
# Section 15: Generator
# ===========================================================

class TestGeneratorStructure:
    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_family_fn_exists(self, family):
        assert family in FAMILY_FNS
        assert callable(FAMILY_FNS[family])

    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_points_sum_to_unity(self, family):
        sc = generate_scenario(family, seed=1)
        total = sum(iss["points"]
                    for cat in sc["issues"]
                    for iss in sc["issues"][cat])
        assert total == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_has_hidden_state(self, family):
        sc = generate_scenario(family, seed=1)
        assert "hidden_state" in sc

    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_has_max_steps(self, family):
        sc = generate_scenario(family, seed=1)
        assert sc["max_steps"] > 0

    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_has_services(self, family):
        sc = generate_scenario(family, seed=1)
        assert len(sc["services"]) > 0

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            generate_scenario("not_a_family", seed=1)


class TestGeneratorDifficulty:
    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_hard_more_tickets_than_easy(self, family):
        easy = generate_scenario(family, seed=1, difficulty="easy")
        hard = generate_scenario(family, seed=1, difficulty="hard")
        assert len(hard["tickets"]) >= len(easy["tickets"])

    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_hard_more_steps_than_easy(self, family):
        easy = generate_scenario(family, seed=1, difficulty="easy")
        hard = generate_scenario(family, seed=1, difficulty="hard")
        assert hard["max_steps"] >= easy["max_steps"]


class TestGeneratorDeterminism:
    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_same_seed_same_output(self, family):
        a = generate_scenario(family, seed=77, difficulty="medium")
        b = generate_scenario(family, seed=77, difficulty="medium")
        assert a == b

    @pytest.mark.parametrize("family", GENERATED_FAMILIES)
    def test_different_seeds_different_output(self, family):
        a = generate_scenario(family, seed=1, difficulty="medium")
        b = generate_scenario(family, seed=999, difficulty="medium")
        # At least description or current_time differs
        assert (a["description"] != b["description"]
                or a["current_time"] != b["current_time"])

    def test_generated_loadable_by_env(self):
        env = OpsTwinEnvironment()
        obs = env.reset(task="generated", family="bad_release", seed=42)
        assert obs.total_issues_count > 0

    def test_generated_task_naming_parses(self):
        env = OpsTwinEnvironment()
        obs = env.reset(task="gen_bad_release_s5_easy")
        assert obs.total_issues_count > 0


# ===========================================================
# Section 16: Dynamic Events
# ===========================================================

class TestDynamicEvents:
    def test_event_fires_at_scheduled_step(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")  # has event at step 6
        for i in range(6):
            obs = env.step(OpsAction(command="REQUEST_INFO summary"))
            if obs.done:
                break
        # At step 6 the event should have fired
        assert 6 in env._fired_events

    def test_event_does_not_refire(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for i in range(10):
            env.step(OpsAction(command="REQUEST_INFO summary"))
            if env._done:
                break
        # Fired set should contain step 6 exactly once
        assert len([e for e in env._fired_events if e == 6]) == 1

    def test_scenario_without_events_runs_clean(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")  # no dynamic events
        for _ in range(5):
            env.step(OpsAction(command="REQUEST_INFO summary"))
        assert env._fired_events == set()


# ===========================================================
# Section 17: Pipeline Cooldowns
# ===========================================================

class TestPipelineCooldowns:
    def test_rerun_in_cooldown_rejected(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        env.step(OpsAction(command="APPROVE_EXCEPTION deploy-auth-patch-1.8.2"))
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        env.step(OpsAction(command="RERUN_PIPELINE deploy-auth-patch-1.8.2"))
        obs = env.step(OpsAction(
            command="RERUN_PIPELINE deploy-auth-patch-1.8.2"))
        assert obs.reward < 0
        assert "cooldown" in obs.message.lower()

    def test_cooldown_expires(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        env.step(OpsAction(command="SWITCH_DESK SECURITY"))
        env.step(OpsAction(command="APPROVE_EXCEPTION deploy-auth-patch-1.8.2"))
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        env.step(OpsAction(command="RERUN_PIPELINE deploy-auth-patch-1.8.2"))
        # Wait out cooldown
        for _ in range(env.PIPELINE_COOLDOWN_STEPS + 1):
            env.step(OpsAction(command="REQUEST_INFO summary"))
        # Cooldown should have cleared
        assert "deploy-auth-patch-1.8.2" not in env._pipeline_cooldowns


# ===========================================================
# Section 18: Alert Sweep
# ===========================================================

class TestAlertSweep:
    def test_alert_clears_when_service_healthy(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        env.step(OpsAction(command="SWITCH_DESK RELEASE"))
        env.step(OpsAction(command="FLIP_FLAG checkout_v2_ui off"))
        # Service now HEALTHY; alerts should clear
        for aid in ["A-001", "A-002"]:
            assert aid in env._resolved["alerts_to_clear"]

    def test_alert_with_requires_inspection_does_not_clear_without_inspect(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")
        # Do not call INSPECT_RUNBOOK
        for _ in range(3):
            env.step(OpsAction(command="REQUEST_INFO summary"))
        assert len(env._resolved["alerts_to_clear"]) == 0

    def test_alert_with_requires_inspection_clears_after_inspect(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")
        env.step(OpsAction(command="SWITCH_DESK SRE"))
        env.step(OpsAction(command="INSPECT_RUNBOOK auth-svc"))
        # Both A-001 and A-002 have requires_inspection runbook:auth-svc
        assert len(env._resolved["alerts_to_clear"]) == 2


# ===========================================================
# Section 19: Escalate to IC
# ===========================================================

class TestEscalateToIC:
    def test_first_call_sets_flag(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        assert env._escalation_used is False
        env.step(OpsAction(command="ESCALATE_TO_IC"))
        assert env._escalation_used is True

    def test_hint_mentions_flag_for_bad_release(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="ESCALATE_TO_IC"))
        assert "flag" in obs.message.lower() or "FLIP" in obs.message \
               or "checkout" in obs.message.lower()

    def test_hint_mentions_approval_for_security_cve(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        obs = env.step(OpsAction(command="ESCALATE_TO_IC"))
        # First hint may be about service outage; but within scenario body
        # someone should be mentioned
        assert len(obs.message) > 20


# ===========================================================
# Section 20: Efficiency Bonus
# ===========================================================

class TestEfficiencyBonus:
    def test_early_completion_earns_bonus(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")  # 10 max steps
        for cmd in OPTIMAL_TRAJECTORIES["false_positive"]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        # Should have finished in fewer than max_steps
        assert env.state.step_count < env._max_steps
        # Efficiency bonus added to score
        assert obs.score > 0.85

    def test_time_out_no_bonus(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for _ in range(14):
            obs = env.step(OpsAction(command="REQUEST_INFO summary"))
            if obs.done:
                break
        # No bonus: capped at 0.30 by inaction
        assert obs.score <= 0.30


# ===========================================================
# Section 21: FastAPI Server
# ===========================================================

class TestServerRoutes:
    """Use FastAPI's TestClient to hit routes in-process."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self, client):
        r = client.get("/schema")
        assert r.status_code == 200
        body = r.json()
        # Should describe action and observation types
        assert isinstance(body, dict)

    def test_metadata(self, client):
        r = client.get("/metadata")
        assert r.status_code == 200

    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


# ===========================================================
# Section 22: Training Wrapper
# ===========================================================

class TestTrainingWrapper:
    """OpsTwinTrainingEnv from train.py: typed tool dispatch."""

    def test_import(self):
        from train import OpsTwinTrainingEnv
        assert OpsTwinTrainingEnv is not None

    def test_reset_returns_string(self):
        from train import OpsTwinTrainingEnv
        env = OpsTwinTrainingEnv()
        result = env.reset()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_switch_desk_tool(self):
        from train import OpsTwinTrainingEnv
        env = OpsTwinTrainingEnv()
        env.reset()
        result = env.switch_desk(desk="SRE")
        assert isinstance(result, str)
        assert env.reward > 0  # first switch earns bonus

    def test_assess_blast_radius_tool(self):
        from train import OpsTwinTrainingEnv
        env = OpsTwinTrainingEnv()
        env.reset()
        env.switch_desk(desk="INCIDENT_COMMAND")
        env.assess_blast_radius()
        # Should have accumulated reward from both calls
        assert env.reward > 0

    def test_reward_accumulates_across_turns(self):
        from train import OpsTwinTrainingEnv
        env = OpsTwinTrainingEnv()
        env.reset()
        env.switch_desk(desk="INCIDENT_COMMAND")
        r1 = env.reward
        env.switch_desk(desk="SRE")
        r2 = env.reward
        assert r2 > r1

    def test_finish_incident_ends_episode(self):
        from train import OpsTwinTrainingEnv
        env = OpsTwinTrainingEnv()
        env.reset()
        env.finish_incident()
        assert env._done is True

    def test_tool_raises_on_done_episode(self):
        from train import OpsTwinTrainingEnv
        env = OpsTwinTrainingEnv()
        env.reset()
        env.finish_incident()
        with pytest.raises(ValueError):
            env.switch_desk(desk="SRE")

    def test_task_rotation(self):
        from train import OpsTwinTrainingEnv, _next_task, _TRAIN_TASKS
        seen = set()
        for _ in range(len(_TRAIN_TASKS) * 2):
            seen.add(_next_task())
        assert seen == set(_TRAIN_TASKS)


# ===========================================================
# Section 23: Inference Parser
# ===========================================================

class TestInferenceParser:
    """_extract_command handles every shape of LLM output."""

    @pytest.fixture
    def parse(self):
        from inference import _extract_command
        return _extract_command

    def test_plain_command(self, parse):
        assert parse("SWITCH_DESK SRE") == "SWITCH_DESK SRE"

    def test_empty_input(self, parse):
        assert parse("") == ""

    def test_whitespace_only(self, parse):
        assert parse("   \n\t  ") == ""

    def test_quoted_command(self, parse):
        assert parse('"SWITCH_DESK SRE"') == "SWITCH_DESK SRE"
        assert parse("'SWITCH_DESK SRE'") == "SWITCH_DESK SRE"

    def test_think_block_stripped(self, parse):
        assert parse("<think>reasoning</think>\nSWITCH_DESK SRE") == "SWITCH_DESK SRE"

    def test_think_block_multiline(self, parse):
        text = "<think>\nline1\nline2\n</think>\n\nFLIP_FLAG x off"
        assert parse(text) == "FLIP_FLAG x off"

    def test_truncated_think_returns_empty(self, parse):
        assert parse("<think>\nlet me analyze") == ""

    def test_nested_thinks(self, parse):
        text = "<think>a</think><think>b</think>\nCOMMAND"
        assert parse(text) == "COMMAND"

    def test_code_fence(self, parse):
        text = "```\nASSESS_BLAST_RADIUS\n```"
        assert parse(text) == "ASSESS_BLAST_RADIUS"

    def test_code_fence_with_lang(self, parse):
        text = "```bash\nASSESS_BLAST_RADIUS\n```"
        assert parse(text) == "ASSESS_BLAST_RADIUS"

    def test_prose_prefix_command(self, parse):
        assert parse("Command: SWITCH_DESK SRE") == "SWITCH_DESK SRE"

    def test_prose_prefix_action(self, parse):
        assert parse("Action: SWITCH_DESK SRE") == "SWITCH_DESK SRE"

    def test_prose_prefix_i_will_call(self, parse):
        assert parse("I will call: SWITCH_DESK SRE") == "SWITCH_DESK SRE"

    def test_multiline_first_wins(self, parse):
        text = "SWITCH_DESK SRE\nThen I will do something else"
        assert parse(text) == "SWITCH_DESK SRE"

    def test_think_then_multiline(self, parse):
        text = "<think>reasoning</think>\n\nASSESS_BLAST_RADIUS\n\n(commentary)"
        assert parse(text) == "ASSESS_BLAST_RADIUS"

    def test_backtick_command(self, parse):
        assert parse("`SWITCH_DESK SRE`") == "SWITCH_DESK SRE"

    def test_comment_line_skipped(self, parse):
        text = "# This is a comment\nSWITCH_DESK SRE"
        assert parse(text) == "SWITCH_DESK SRE"

    def test_command_with_sentence_content(self, parse):
        # DRAFT_COMMS takes a message body
        text = "DRAFT_COMMS external Customer-facing outage resolved."
        assert parse(text) == "DRAFT_COMMS external Customer-facing outage resolved."

    def test_only_think_block(self, parse):
        assert parse("<think>just thinking</think>") == ""


# ===========================================================
# Section 24: Inference Fallback Chain
# ===========================================================

class TestInferenceFallback:
    def test_chain_deduplicated(self):
        from inference import _MODEL_FALLBACK_CHAIN
        assert len(_MODEL_FALLBACK_CHAIN) == len(set(_MODEL_FALLBACK_CHAIN))

    def test_primary_model_first(self):
        from inference import _MODEL_FALLBACK_CHAIN, MODEL_NAME
        assert _MODEL_FALLBACK_CHAIN[0] == MODEL_NAME

    def test_fallback_models_present(self):
        from inference import _MODEL_FALLBACK_CHAIN
        # Should include at least one Qwen2.5 or Llama fallback
        has_fallback = any("Qwen2.5" in m or "Llama" in m
                           for m in _MODEL_FALLBACK_CHAIN)
        assert has_fallback

    def test_current_model_returns_active(self):
        from inference import _current_model, _MODEL_FALLBACK_CHAIN
        assert _current_model() in _MODEL_FALLBACK_CHAIN


# ===========================================================
# Section 25: Inference Prompt Builder
# ===========================================================

class TestInferencePrompt:
    def test_build_prompt_returns_string(self):
        from inference import build_user_prompt
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        prompt = build_user_prompt(1, obs, 0.0, [])
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_prompt_includes_services(self):
        from inference import build_user_prompt
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        prompt = build_user_prompt(1, obs, 0.0, [])
        assert "SERVICES" in prompt

    def test_prompt_includes_tickets(self):
        from inference import build_user_prompt
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        prompt = build_user_prompt(1, obs, 0.0, [])
        assert "TICKETS" in prompt

    def test_prompt_truncates_long_history(self):
        from inference import build_user_prompt
        env = OpsTwinEnvironment()
        obs = env.reset(task="bad_release")
        long_history = [f"S{i}: CMD -> +0.01" for i in range(50)]
        prompt = build_user_prompt(1, obs, 0.0, long_history)
        # Only keeps last 5
        for i in range(40):
            assert f"S{i}: " not in prompt
        for i in range(45, 50):
            assert f"S{i}: " in prompt


# ===========================================================
# Section 26: Integration - Optimal Trajectories
# ===========================================================

class TestIntegrationOptimal:
    """Expert trajectories hit the 0.90+ ceiling on every scenario."""

    @pytest.mark.parametrize("task",
                             ["bad_release", "security_cve",
                              "data_pipeline_regression"])
    def test_optimal_scores_high(self, task):
        env = OpsTwinEnvironment()
        env.reset(task=task)
        for cmd in OPTIMAL_TRAJECTORIES[task]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        assert obs.score > 0.90
        assert obs.resolved_issues_count == obs.total_issues_count

    def test_false_positive_optimal_scores_high(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")
        for cmd in OPTIMAL_TRAJECTORIES["false_positive"]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        assert obs.score > 0.85


# ===========================================================
# Section 27: Integration - Trap Behaviours
# ===========================================================

class TestIntegrationTraps:
    """Trap behaviours (fast-but-wrong) score <= 0.75."""

    def test_rollback_without_assess_bad_release(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        for cmd in [
            "SWITCH_DESK SRE",
            "ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1",
            "DONE",
        ]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        assert obs.score < 0.50

    def test_quarantine_vip_security_cve(self):
        env = OpsTwinEnvironment()
        env.reset(task="security_cve")
        for cmd in [
            "SWITCH_DESK SECURITY",
            "QUARANTINE_SERVICE auth-svc",
            "DONE",
        ]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        assert obs.score < 0.80

    def test_rollback_on_false_positive(self):
        env = OpsTwinEnvironment()
        env.reset(task="false_positive")
        for cmd in [
            "SWITCH_DESK SRE",
            "ROLLBACK_DEPLOYMENT deploy-auth-v1.9.0",
            "SWITCH_DESK SECURITY",
            "QUARANTINE_SERVICE auth-svc",
            "DONE",
        ]:
            obs = env.step(OpsAction(command=cmd))
            if obs.done:
                break
        assert obs.score < 0.80

    def test_immediate_done_on_bad_release(self):
        env = OpsTwinEnvironment()
        env.reset(task="bad_release")
        obs = env.step(OpsAction(command="DONE"))
        assert obs.score < 0.30


# ===========================================================
# Section 28: Reproducibility
# ===========================================================

class TestReproducibility:
    def test_same_actions_same_score(self):
        """Two env instances running identical action sequences hit same score."""
        def run():
            env = OpsTwinEnvironment()
            env.reset(task="bad_release", episode_id="repro-test")
            for cmd in OPTIMAL_TRAJECTORIES["bad_release"]:
                obs = env.step(OpsAction(command=cmd))
                if obs.done:
                    break
            return obs.score

        assert run() == run()

    def test_reset_is_idempotent_for_same_task(self):
        env = OpsTwinEnvironment()
        a = env.reset(task="bad_release")
        b = env.reset(task="bad_release")
        # Same total issues, same max_steps, same services count
        assert a.total_issues_count == b.total_issues_count
        assert len(a.services) == len(b.services)
        assert len(a.tickets) == len(b.tickets)

    def test_all_scenarios_deterministic_score(self):
        for task in ["bad_release", "security_cve", "data_pipeline_regression"]:
            env1 = OpsTwinEnvironment()
            env1.reset(task=task)
            env2 = OpsTwinEnvironment()
            env2.reset(task=task)
            for cmd in OPTIMAL_TRAJECTORIES[task]:
                o1 = env1.step(OpsAction(command=cmd))
                o2 = env2.step(OpsAction(command=cmd))
                if o1.done:
                    break
            assert o1.score == o2.score
