"""Common-weight certificates checked on the actual selected action.

Passing on A_bar certifies every next-step A <= A_bar (componentwise), with
the same type/model envelope and positive weight. A snapshot certificate is
NOT labelled a guarantee for unknown future contacts. Infeasible/unknown
searches retain an explicit failure state instead of asserting stability.
"""
import torch


def ratio(adjacency_upper, types, tables, network, params, weight=None):
    from sim_core import linear_comparison_step
    if weight is None:
        weight = torch.ones(*types.shape, 5, device=types.device)
    if (weight <= 0).any():
        raise ValueError("certificate requires a strictly positive, fixed weight")
    image = linear_comparison_step(adjacency_upper, weight, types, tables, network, params)
    return (image / weight).amax(dim=(1, 2))


def trial_tables(tables, index):
    return {key: value[index:index + 1] if key.endswith(("_by_trial", "_by_node")) else value
            for key, value in tables.items()}


def repair_action(network, guard, adjacency_upper, types, tables, params,
                  network_cost, guard_cost, budget, weight=None):
    """Certify proposal or replace it by a budget-feasible network repair.

    N<=12: enumerate all network masks, therefore prove certificate feasibility
    or infeasibility for the supplied common weight. Larger N: a deterministic
    violation-priority candidate plus the affordable full-network fallback;
    failure of that search is reported as unknown, never as proven infeasible.
    Guards do not enter this propagation certificate and are filled only from
    budget left after a certified network mask has been chosen.
    """
    from sim_core import action_cost, allocate_discrete, linear_comparison_step
    batch, n = types.shape
    weight = torch.ones(batch, n, 5, device=types.device) if weight is None else weight
    before = ratio(adjacency_upper, types, tables, network, params, weight)
    repaired_network, repaired_guard = network.clone(), guard.clone()
    diagnostics = []
    for b in range(batch):
        item = {"proposal_ratio": float(before[b]), "weight_fixed": True,
                "envelope_scope": "supplied_adjacency_upper", "search": "not_needed"}
        if before[b] <= params.certificate_mu:
            item.update(status="proposal_certified", selected_ratio=float(before[b]))
            diagnostics.append(item)
            continue
        table = trial_tables(tables, b)
        adj = adjacency_upper[b:b + 1]
        tp = types[b:b + 1]
        w = weight[b:b + 1]
        chosen = None
        if n <= 12:
            bits = torch.arange(1 << n, device=types.device).unsqueeze(1)
            masks = ((bits >> torch.arange(n, device=types.device)) & 1).float()
            spend = (masks.double() * network_cost[b].double()).sum(-1)
            masks = masks[spend <= budget[b].double() + 1e-7]
            candidate_tables = {key: value.expand(len(masks), *value.shape[1:])
                                if key.endswith(("_by_trial", "_by_node")) else value
                                for key, value in table.items()}
            bounds = ratio(adj.expand(len(masks), -1, -1), tp.expand(len(masks), -1),
                           candidate_tables, masks, params, w.expand(len(masks), -1, -1))
            valid = bounds <= params.certificate_mu
            item["search"] = "exhaustive_network_masks"
            if valid.any():
                valid_masks = masks[valid]
                costs = (valid_masks.double() * network_cost[b].double()).sum(-1)
                chosen = valid_masks[costs.argmin()].unsqueeze(0)
            else:
                item["status"] = "proven_infeasible_for_supplied_weight_budget_envelope"
        else:
            influence = linear_comparison_step(adj, w, tp, table, torch.zeros_like(network[b:b + 1]), params)
            score = (influence / w).amax(-1)
            candidate, _ = allocate_discrete(score, torch.full_like(score, -torch.inf),
                                             network_cost[b:b + 1], guard_cost[b:b + 1], budget[b:b + 1])
            item["search"] = "violation_priority_with_full_network_fallback"
            if ratio(adj, tp, table, candidate, params, w)[0] <= params.certificate_mu:
                chosen = candidate
            elif network_cost[b].double().sum() <= budget[b].double() + 1e-7:
                full = torch.ones_like(candidate)
                if ratio(adj, tp, table, full, params, w)[0] <= params.certificate_mu:
                    chosen = full
                else:
                    item["status"] = "proven_infeasible_full_network_fails_for_supplied_weight"
            else:
                item["status"] = "uncertified_repair_search_incomplete"
        if chosen is not None:
            remainder = (budget[b:b + 1].double() - (chosen.double() * network_cost[b:b + 1].double()).sum(-1)).clamp_min(0)
            _, remaining_guard = allocate_discrete(torch.full_like(chosen, -torch.inf),
                                                   guard[b:b + 1], network_cost[b:b + 1],
                                                   guard_cost[b:b + 1], remainder)
            repaired_network[b:b + 1] = chosen
            repaired_guard[b:b + 1] = remaining_guard
            item["status"] = "repaired_and_certified"
        item["selected_ratio"] = float(ratio(adj, tp, table, repaired_network[b:b + 1], params, w)[0])
        diagnostics.append(item)
    spent = action_cost(repaired_network, repaired_guard, network_cost, guard_cost)
    if (spent > budget.double() + 1e-5).any():
        raise AssertionError("certificate repair exceeded the shared budget")
    return repaired_network, repaired_guard, diagnostics


def forced_comparison_bound(initial_norm, block_mu, block_input_bound, blocks):
    if not 0 <= block_mu < 1 or block_input_bound < 0:
        raise ValueError("contractive common block factor and nonnegative input bound required")
    return block_mu ** blocks * initial_norm + block_input_bound * (1 - block_mu ** blocks) / (1 - block_mu)


def prospective_adjacency_upper(domains, scenario, admission=None):
    """Position-independent envelope fixed before any future motion is drawn.

    Gaussian motion has unbounded increments, so a snapshot plus a finite
    distance margin cannot be a deterministic bound. The unrestricted
    envelope permits every pair. An optional predeclared admission mask is an
    additional enforced network constraint, not a prediction of future edges.
    """
    if scenario not in ("warehouse", "coverage", "formation", "allocation"):
        raise ValueError("unknown scenario")
    n = domains.shape[1]
    same = (domains.unsqueeze(1) == domains.unsqueeze(2)).float()
    factor = {"warehouse": 1., "coverage": .92, "formation": 1.08, "allocation": 1.}[scenario]
    upper = factor * (.42 + .58 * same) * (1. - torch.eye(n, device=domains.device))
    if admission is not None:
        if admission.shape != upper.shape or not torch.isfinite(admission).all() or not ((admission == 0) | (admission == 1)).all():
            raise ValueError("admission must be an explicit binary pair mask")
        upper = upper * admission
    return upper


def minimum_common_weight_cost(adjacency_upper, types, tables, params, network_cost, weight=None):
    """Exact N<=12 feasibility frontier; never interpret unknown as infeasible."""
    if types.shape[0] != 1 or types.shape[1] > 12:
        raise ValueError("exact network-mask frontier requires one case with N<=12")
    n = types.shape[1]
    bits = torch.arange(1 << n, device=types.device).unsqueeze(1)
    masks = ((bits >> torch.arange(n, device=types.device)) & 1).float()
    expanded = {key: value.expand(len(masks), *value.shape[1:]) if key.endswith(("_by_trial", "_by_node")) else value
                for key, value in tables.items()}
    weights = None if weight is None else weight.expand(len(masks), -1, -1)
    bounds = ratio(adjacency_upper.expand(len(masks), -1, -1), types.expand(len(masks), -1),
                   expanded, masks, params, weights)
    valid = bounds <= params.certificate_mu
    costs = (masks.double() * network_cost.double()).sum(-1)
    return {"search": "all_network_masks", "evaluated_masks": len(masks),
            "certified_masks": int(valid.sum()),
            "minimum_certification_cost": float(costs[valid].min()) if valid.any() else None,
            "full_network_ratio": float(bounds[-1]), "weight": "fixed_all_ones" if weight is None else "supplied_fixed_positive"}


def receiving_role_envelope(tables, calibration, types):
    """Entrywise comparison envelope over all declared receiving-task roles.

    This positive upper matrix need not be row-substochastic. It is a bound,
    not a fabricated measured transition-probability matrix, and must never
    replace the actual role-specific matrix in nonlinear plant dynamics.
    """
    if calibration is None or "receiver_type_role_matrices" not in calibration:
        raise ValueError("a role-switching certificate requires all measured role kernels")
    array = torch.as_tensor(calibration["receiver_type_role_matrices"], dtype=tables["mutation"].dtype, device=types.device)
    if array.shape != (4, 5, 5, 5) or not torch.isfinite(array).all() or (array < 0).any() or (array.sum(-1) > 1.000001).any():
        raise ValueError("invalid actual receiving-role kernels")
    result = dict(tables)
    result["mutation_by_node"] = array.amax(dim=1)[types]
    return result
