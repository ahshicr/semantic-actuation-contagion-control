"""Explicit v2 terminal controllers, not replacements for frozen CAGMC.

An affordable public physical-target floor is installed before generation and
charged to the SAME budget. Network derivatives are conditioned on this fixed
floor. The path prior supplies an optional candidate, never an obligation to
buy an action or a reason to remove a guard. Its candidate must improve J and
not increase predicted execution exposure relative to the matched no-path
controller. These are finite current-state forecast guarantees, not empirical
closed-loop superiority or semantic detection of an incoming instruction.
"""
import math
import time

import torch

from hardware_static_guard import static_actuator_guard
from sim_core import (action_cost, actuator_metrics, allocate_discrete,
                      forecast_objective, gather, path_pressure, propagate)


VERSION = "terminal_protected_v2_20261005"
POLICIES = ("cagmc_protected_v2", "gradient_protected_v2")
OBJECTIVE_IMPROVEMENT_MARGIN = 1e-12
BUDGET_TOLERANCE = 1e-6


def conditional_network_benefits(z, adjacency, types, tables, params, floor):
    """Differentiate the complete forecast at (x=0, g=floor), not (0,0)."""
    network = torch.zeros_like(floor, requires_grad=True)
    value = forecast_objective(z, adjacency, types, tables, network, floor,
                               params, params.horizon)
    derivative, = torch.autograd.grad(value.sum(), (network,))
    return (-derivative).clamp_min(0.).detach()


def forecast_execution_exposure(z, adjacency, types, tables, network, guard, params):
    """Same discounted successful-exposure term as J, without its weight."""
    state = z
    exposure = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    for step in range(params.horizon):
        state = propagate(adjacency, state, types, tables, network, params)
        _, _, successful = actuator_metrics(state, types, tables, guard, params)
        exposure = exposure + params.discount ** step * successful.mean(dim=1)
    return exposure


def improves_without_execution_increase(candidate, reference):
    """Strict J improvement, scaled numerical margin, and E <= reference E.

    There is no positive tolerance permitting an execution-exposure increase.
    Ties retain the reference action, even when the path prior has a different
    score. The small J margin only makes accepting a new action MORE strict.
    """
    margin = OBJECTIVE_IMPROVEMENT_MARGIN * max(1., abs(reference["forecast_objective"]))
    return (candidate["forecast_objective"] < reference["forecast_objective"] - margin
            and candidate["forecast_execution_exposure"] <= reference["forecast_execution_exposure"])


def _validate(nodes, z, adjacency, types, tables, params):
    n = len(nodes)
    if not n or z.shape != (1, n, 5) or adjacency.shape != (1, n, n) or types.shape != (1, n):
        raise ValueError("one matching nonempty ordered network required")
    if (z.device.type != "cpu" or adjacency.device != z.device or types.device != z.device
            or not z.is_floating_point() or not adjacency.is_floating_point()
            or types.dtype != torch.long):
        raise ValueError("CPU floating observations and int64 types required")
    if (not torch.isfinite(z).all() or (z < 0).any() or (z > 1).any()
            or not torch.isfinite(adjacency).all() or (adjacency < 0).any()):
        raise ValueError("invalid observed state/contact weights")
    if not isinstance(params.horizon, int) or params.horizon < 1:
        raise ValueError("positive integer forecast horizon required")
    if not 0 <= params.guard_effectiveness <= 1 or not 0 <= params.discount <= 1:
        raise ValueError("invalid exposure-effectiveness or discount")
    if not 0 <= params.gate_strength <= 1 or not 0 <= params.cleansing_strength <= 1:
        raise ValueError("invalid network effectiveness")
    if not math.isfinite(params.eps) or params.eps <= 0 or not 0 <= params.ppr_restart <= 1:
        raise ValueError("invalid path-normalization parameters")
    for key, value in tables.items():
        if not isinstance(value, torch.Tensor) or value.device != z.device:
            raise ValueError("matching CPU tensor tables required: " + key)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("nonfinite table: " + key)
    # Preserve the hardware baseline's represented float32 prices. The audit
    # independently rounds the same input prices, not idealized real prices.
    if any(tables[key].dtype != torch.float32 for key in ("network_cost", "guard_cost_factor")):
        raise ValueError("declared float32 represented action-price tables required")
    if (types < 0).any() or (types >= len(tables["network_cost"])).any():
        raise ValueError("invalid endpoint type")
    if any(type(node.get("serial_target")) is not bool for node in nodes):
        raise ValueError("explicit public physical-target flags required")


def protected_action(policy, nodes, z, adjacency, types, tables, params,
                     budget_fraction, guard_cost_ratio):
    started = time.perf_counter()
    if policy not in POLICIES:
        raise ValueError("unknown protected-v2 policy")
    _validate(nodes, z, adjacency, types, tables, params)
    n = len(nodes)
    floor_network, floor, _, floor_selection = static_actuator_guard(
        nodes, types, tables, budget_fraction, guard_cost_ratio)
    cn = gather(tables["network_cost"], types)
    cg = guard_cost_ratio * gather(tables["guard_cost_factor"], types)
    if not torch.isfinite(cn).all() or (cn <= 0).any():
        raise ValueError("finite positive represented network costs required")
    floor_cost = action_cost(floor_network, floor, cn, cg)
    budget = n * budget_fraction
    if not math.isfinite(budget) or bool((floor_cost > budget + BUDGET_TOLERANCE).any()):
        raise ValueError("invalid shared budget")
    residual = (budget - floor_cost).clamp_min(0.)
    oz, oa, og = z.double(), adjacency.double(), floor.double()
    ot = {key: value.double() if value.is_floating_point() else value for key, value in tables.items()}
    benefits = conditional_network_benefits(oz, oa, types, ot, params, og)
    weights = torch.ones_like(benefits)
    if policy == "cagmc_protected_v2":
        pressure = path_pressure(oa, types, ot, restart=params.ppr_restart, eps=params.eps)
        weights = .75 + .25 * pressure / pressure.mean(dim=1, keepdim=True).clamp_min(params.eps)

    def allocate(scores):
        network, unused_guard = allocate_discrete(
            scores, torch.full_like(scores, -torch.inf), cn.double(), cg.double(), residual)
        if bool(unused_guard.any()):
            raise AssertionError("the immutable floor cannot be replaced or double charged")
        return network

    raw = allocate(benefits)
    candidates = [("floor_only", floor_network.double()), ("conditional_no_path", raw)]
    if policy == "cagmc_protected_v2":
        candidates.append(("conditional_path_prior", allocate(benefits * weights)))
    rows = []
    with torch.no_grad():
        for name, network in candidates:
            cost = float(action_cost(network, og, cn, cg)[0])
            value = float(forecast_objective(oz, oa, types, ot, network, og, params, params.horizon)[0])
            exposure = float(forecast_execution_exposure(oz, oa, types, ot, network, og, params)[0])
            if not math.isfinite(value) or not math.isfinite(exposure) or exposure < 0:
                raise ValueError("invalid candidate forecast")
            if not 0 <= cost <= budget + BUDGET_TOLERANCE:
                raise AssertionError("candidate exceeds the joint budget")
            rows.append({"name": name, "network": network.tolist(), "guard": floor.tolist(),
                         "forecast_objective": value, "forecast_execution_exposure": exposure, "cost": cost})
    # First compute precisely the same baseline action in BOTH policies. Then
    # the path candidate has to beat that action without worsening execution.
    matched = 1 if improves_without_execution_increase(rows[1], rows[0]) else 0
    selected = matched
    prior_changes = False
    prior_reason = "not_requested_in_matched_no_path_controller"
    if len(rows) == 3:
        prior_changes = rows[2]["network"] != rows[1]["network"]
        if not prior_changes:
            prior_reason = "same_feasible_network_action"
        elif improves_without_execution_increase(rows[2], rows[matched]):
            selected = 2
            prior_reason = "strict_forecast_improvement_without_execution_increase"
        elif rows[2]["forecast_execution_exposure"] > rows[matched]["forecast_execution_exposure"]:
            prior_reason = "rejected_execution_exposure_increase"
        else:
            prior_reason = "rejected_no_strict_forecast_improvement"
    network, guard = candidates[selected][1].to(z.dtype), floor.to(z.dtype)
    protected = [i for i, node in enumerate(nodes) if node["serial_target"] and bool(guard[0, i])]
    uncovered = [i for i, node in enumerate(nodes) if node["serial_target"] and not bool(guard[0, i])]
    affordable = [i for i in range(n) if float(cn[0, i]) <= float(residual[0]) + 1e-7]
    diagnostic = {
        "controller_version": VERSION, "controller_variant": policy, "forecast_horizon": params.horizon,
        "derivative_precision": "float64", "candidate_precision": "float64",
        "floor": {"network": floor_network.tolist(), "guard": floor.tolist(),
                  "realized_cost": float(floor_cost[0]), "selection": floor_selection},
        "protected_public_target_indices": protected, "uncovered_public_target_indices": uncovered,
        "budget": budget, "budget_tolerance": BUDGET_TOLERANCE, "remaining_budget": float(residual[0]),
        "network_affordable_after_floor_indices": affordable,
        "positive_benefit_affordable_indices": [i for i in affordable if float(benefits[0, i]) > 0],
        "conditional_network_benefits": benefits.tolist(), "network_prior_weights": weights.tolist(),
        "path_candidate_changes_no_path_action": prior_changes, "path_candidate_disposition": prior_reason,
        "candidates": rows, "matched_no_path_selected_index": matched, "selected_index": selected,
        "selected_objective": rows[selected]["forecast_objective"],
        "selected_execution_exposure": rows[selected]["forecast_execution_exposure"],
        "objective_improvement_margin": OBJECTIVE_IMPROVEMENT_MARGIN,
        "execution_increase_tolerance": 0.,
        "residual_action_space": "network_only_with_immutable_budgeted_public_target_floor",
        "scope": "affordable-target first-generation guard; J and E do not exceed the floor or matched no-path forecast",
        "not_claimed": "instruction detection, protection of unaffordable targets, global optimality, a stability certificate, or empirical closed-loop superiority"}
    return network, guard, time.perf_counter() - started, diagnostic
