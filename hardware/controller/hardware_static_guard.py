"""Hardware-only static actuator guard baseline, independent of contamination.

Protect the maximum affordable number of public physical targets, using the
lowest represented cost then public node ID. No network actions, truth labels,
model outputs, observed contamination or attacker state are inputs. This is not
a modification of CAGMC or the frozen formal simulation policies.
"""
import math
import time

import torch


def static_actuator_guard(nodes, types, tables, budget_fraction, guard_cost_ratio):
    started = time.perf_counter()
    n = len(nodes)
    if types.shape != (1, n) or not n or len({node["node_id"] for node in nodes}) != n:
        raise ValueError("one ordered unique physical inventory required")
    if not math.isfinite(budget_fraction) or budget_fraction < 0 or not math.isfinite(guard_cost_ratio) or guard_cost_ratio <= 0:
        raise ValueError("finite nonnegative budget and positive guard cost required")
    if types.tolist()[0] != [node["type"] for node in nodes]:
        raise ValueError("type order differs from actual endpoint inventory")
    costs = (guard_cost_ratio * tables["guard_cost_factor"][types])[0].tolist()
    if any(not math.isfinite(value) or value <= 0 for value in costs):
        raise ValueError("finite positive represented guard costs required")
    active = [index for index, node in enumerate(nodes) if node["serial_target"]]
    order = sorted(active, key=lambda index: (costs[index], nodes[index]["node_id"]))
    network = torch.zeros((1, n), dtype=torch.float32, device=types.device)
    guard = network.clone()
    budget, spend = budget_fraction * n, 0.
    for index in order:
        if spend + costs[index] <= budget + 1e-6:
            guard[0, index] = 1.
            spend += costs[index]
    diagnostics = {"controller_variant": "static_actuator_guard", "public_actuator_indices": active,
                   "ranked_target_indices": order, "represented_guard_costs": costs, "budget": budget,
                   "budget_tolerance": 1e-6, "static_selection_cost": spend,
                   "scope": "maximum affordable count of public physical targets, then minimum cost and public node ID; no risk/observation input",
                   "not_claimed": "global risk optimality, reactive guard_only equivalence, or new CAGMC method"}
    return network, guard, time.perf_counter() - started, diagnostics
