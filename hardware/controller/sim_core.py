"""Cost-matched semantic-actuation contagion simulation primitives.

The controller sees an observation that may be noisy or delayed, while the
true nonlinear dynamics remain separate.  Network actions and execution
guards consume the same explicit per-step budget.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
import torch


SEMANTICS = ("prompt", "memory", "map", "tool", "actuation")
TYPES = ("legacy", "general", "edge_ai", "actuator_host")
INFORMATION_ABLATIONS = ("cagmc_diagonal_semantics", "cagmc_pooled_semantics", "cagmc_uniform_reachability")


@dataclass
class Params:
    world: float = 30.0
    contact_radius: float = 7.0
    contact_scale: float = 3.8
    speed: float = 0.45
    beta_scale: float = 0.115
    control_interval: int = 4
    horizon: int = 5
    discount: float = 0.82
    ppr_restart: float = 0.25
    execution_weight: float = 1.65
    gate_strength: float = 0.78
    guard_effectiveness: float = 0.98
    guard_false_block: float = 0.018
    network_task_penalty: float = 0.14
    guard_task_penalty: float = 0.018
    contamination_task_penalty: float = 0.16
    semantic_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    objective_network_penalty: float = 0.10
    objective_guard_penalty: float = 0.025
    cleansing_strength: float = 0.16
    attack_interval: int = 12
    role_switch_interval: int = 0
    mpc_iterations: int = 300
    mpc_step_size: float = 1.0
    mpc_tolerance: float = 1e-4
    mpc_precision: str = "float64"
    certificate_mu: float = 0.995
    eps: float = 1e-8


def default_mutation(device):
    return torch.tensor(
        [
            [0.62, 0.18, 0.08, 0.08, 0.04],
            [0.11, 0.56, 0.15, 0.11, 0.07],
            [0.07, 0.14, 0.55, 0.14, 0.10],
            [0.05, 0.08, 0.11, 0.55, 0.21],
            [0.03, 0.05, 0.07, 0.19, 0.66],
        ],
        dtype=torch.float32,
        device=device,
    )


def build_tables(device, mutation=None, transfer_shift=False):
    susceptibility = torch.tensor(
        [
            [1.20, 1.12, 1.05, 1.08, 0.95],
            [1.00, 1.00, 0.95, 1.00, 0.90],
            [0.82, 0.78, 0.75, 0.80, 0.72],
            [1.05, 0.98, 0.85, 1.12, 1.28],
        ], dtype=torch.float32, device=device,
    )
    detector = torch.tensor(
        [
            [0.17, 0.14, 0.12, 0.13, 0.10],
            [0.34, 0.31, 0.27, 0.30, 0.24],
            [0.58, 0.55, 0.50, 0.53, 0.46],
            [0.26, 0.22, 0.18, 0.21, 0.58],
        ], dtype=torch.float32, device=device,
    )
    recovery = torch.tensor(
        [
            [0.030, 0.026, 0.023, 0.022, 0.019],
            [0.050, 0.045, 0.040, 0.038, 0.033],
            [0.073, 0.068, 0.060, 0.056, 0.050],
            [0.045, 0.040, 0.036, 0.044, 0.070],
        ], dtype=torch.float32, device=device,
    )
    reachability = torch.tensor(
        [
            [0.08, 0.10, 0.14, 0.22, 0.30],
            [0.12, 0.16, 0.20, 0.28, 0.38],
            [0.16, 0.20, 0.24, 0.32, 0.42],
            [0.20, 0.28, 0.34, 0.58, 0.90],
        ], dtype=torch.float32, device=device,
    )
    beta = torch.tensor([0.95, 0.82, 0.75, 0.90, 1.15], dtype=torch.float32, device=device)
    base = default_mutation(device) if mutation is None else torch.as_tensor(mutation, dtype=torch.float32, device=device)
    if base.shape == (5, 5):
        base = base.unsqueeze(0).expand(len(TYPES), -1, -1).clone()
    if base.shape != (len(TYPES), 5, 5):
        raise ValueError("mutation must have shape (5,5) or (4,5,5); no implicit receiver transformations")
    if not torch.isfinite(base).all() or (base < 0).any() or (base.sum(-1) > 1.000001).any():
        raise ValueError("mutation must be finite, nonnegative and row-substochastic")
    if transfer_shift:
        # Explicit stress transform, preserving observed row survival mass.
        survival = base.sum(dim=-1, keepdim=True)
        base = base.clone()
        base[:, :3, 3] *= 1.45
        base[:, :3, 4] *= 1.85
        base[:, 3, 4] *= 1.35
        base = base / base.sum(dim=-1, keepdim=True).clamp_min(1e-8) * survival
    return {
        "susceptibility": susceptibility,
        "detector": detector,
        "recovery": recovery,
        "reachability": reachability,
        "beta": beta,
        "mutation": base,
        "network_cost": torch.tensor([0.95, 1.00, 1.25, 1.10], dtype=torch.float32, device=device),
        "guard_cost_factor": torch.tensor([0.90, 1.00, 1.20, 1.05], dtype=torch.float32, device=device),
    }


def gather(table, types):
    flat = table[types.reshape(-1)]
    return flat.reshape(types.shape[0], types.shape[1], *table.shape[1:])


def typed(tables, key, types):
    """Gather shared type tables or explicitly independent per-trial tables."""
    if key + "_by_node" in tables:
        return tables[key + "_by_node"]
    if key + "_by_trial" not in tables:
        return gather(tables[key], types)
    table = tables[key + "_by_trial"]
    rows = torch.arange(types.shape[0], device=types.device).unsqueeze(1)
    return table[rows, types]


def slice_tables(tables, index):
    return {key: value[index:index + 1] if key.endswith(("_by_trial", "_by_node")) else value
            for key, value in tables.items()}


def attach_receiver_roles(tables, calibration, types, roles, transfer_shift=False):
    """Use measured model x receiving-task-role kernels without synthetic scaling."""
    if calibration is None or "receiver_type_role_matrices" not in calibration:
        return tables
    result = dict(tables)
    array = torch.as_tensor(calibration["receiver_type_role_matrices"], device=types.device, dtype=torch.float32)
    if array.shape != (4, 5, 5, 5) or not torch.isfinite(array).all() or (array < 0).any() or (array.sum(-1) > 1.000001).any():
        raise ValueError("receiver role matrices must be finite nonnegative substochastic 4x5x5x5")
    if roles.shape != types.shape or (roles < 0).any() or (roles >= 5).any():
        raise ValueError("one valid task role is required for every receiver")
    matrix = array[types, roles]
    if transfer_shift:
        survival = matrix.sum(-1, keepdim=True)
        matrix = matrix.clone()
        matrix[..., :3, 3] *= 1.45
        matrix[..., :3, 4] *= 1.85
        matrix[..., 3, 4] *= 1.35
        matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-8) * survival
    result["mutation_by_node"] = matrix
    if "receiver_type_role_benign_success" in calibration:
        quality = torch.as_tensor(calibration["receiver_type_role_benign_success"], device=types.device, dtype=torch.float32)
        if quality.shape != (4, 5) or not torch.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any():
            raise ValueError("benign task success must be a measured finite 4x5 probability table")
        result["benign_task_success_by_node"] = quality[types, roles]
    if "receiver_type_role_injection_matrices" in calibration:
        injection = torch.as_tensor(calibration["receiver_type_role_injection_matrices"], device=types.device, dtype=torch.float32)
        if injection.shape != (4, 5, 5, 5) or not torch.isfinite(injection).all() or (injection < 0).any() or (injection.sum(-1) > 1.000001).any():
            raise ValueError("injection role matrices must be finite nonnegative substochastic 4x5x5x5")
        result["injection_by_node"] = injection[types, roles]
    return result


def init_batch(batch, n, device):
    probs = torch.tensor([0.22, 0.45, 0.23, 0.10], dtype=torch.float32, device=device)
    types = torch.multinomial(probs, batch * n, replacement=True).reshape(batch, n)
    # Ensure every trial contains at least one actuator host.
    no_host = ~(types == 3).any(dim=1)
    if no_host.any():
        types[no_host, -1] = 3
    pos = torch.rand(batch, n, 2, device=device) * 30.0
    goals = torch.rand(batch, n, 2, device=device) * 30.0
    domains = torch.randint(0, 4, (batch, n), device=device)
    z = torch.zeros(batch, n, len(SEMANTICS), device=device)
    seed_count = max(2, int(round(0.025 * n)))
    for b in range(batch):
        index = torch.randperm(n, device=device)[:seed_count]
        z[b, index, 0] = 0.72
        z[b, index[: max(1, seed_count // 2)], 1] = 0.48
        host = torch.where(types[b] == 3)[0][0]
        z[b, host, 3] = 0.22
    return types, pos, goals, domains, z


def update_positions(pos, goals, scenario, params):
    direction = goals - pos
    distance = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(params.eps)
    unit = direction / distance
    speed = params.speed
    if scenario == "warehouse":
        corridor = torch.exp(-((pos[..., 0:1] - params.world * 0.5) ** 2) / 8.0)
        unit = unit + torch.cat([torch.zeros_like(corridor), 0.35 * corridor * torch.sign(params.world * 0.5 - pos[..., 1:2])], dim=-1)
    elif scenario == "formation":
        center = pos.mean(dim=1, keepdim=True)
        relative = pos - center
        relative_norm = torch.linalg.norm(relative, dim=-1, keepdim=True).clamp_min(params.eps)
        unit = 0.55 * unit + 0.45 * (relative / relative_norm) * torch.sign(8.0 - relative_norm)
    elif scenario == "allocation":
        speed = speed * (1.0 + 0.15 * torch.sin(pos[..., 0:1] / 2.5))
    pos = (pos + 0.25 * speed * unit + 0.08 * torch.randn_like(pos)).clamp(0.0, params.world)
    reached = distance.squeeze(-1) < 0.9
    if reached.any():
        replacement = torch.rand_like(goals) * params.world
        goals = torch.where(reached.unsqueeze(-1), replacement, goals)
    return pos, goals


def contact_graph(pos, domains, scenario, params):
    distance = torch.cdist(pos, pos)
    adjacency = torch.exp(-((distance / params.contact_scale) ** 2)) * (distance < params.contact_radius).float()
    eye = torch.eye(pos.shape[1], device=pos.device).unsqueeze(0)
    adjacency = adjacency * (1.0 - eye)
    same = (domains.unsqueeze(1) == domains.unsqueeze(2)).float()
    adjacency = adjacency * (0.42 + 0.58 * same)
    if scenario == "warehouse":
        left = pos[..., 0] < params.world * 0.5
        cross = (left.unsqueeze(1) != left.unsqueeze(2)).float()
        passage = torch.exp(-((pos[..., 1] - params.world * 0.5) ** 2) / 11.0)
        pair = torch.minimum(passage.unsqueeze(1), passage.unsqueeze(2))
        adjacency = adjacency * (1.0 - cross * (0.78 - 0.58 * pair))
    elif scenario == "coverage":
        adjacency = adjacency * 0.92
    elif scenario == "formation":
        adjacency = adjacency * 1.08
    elif scenario == "allocation":
        # Deterministic position-dependent modulation, not policy-dependent noise.
        modulation = 0.85 + 0.15 * torch.sin(pos[..., 0].unsqueeze(1) + pos[..., 1].unsqueeze(2))
        adjacency = adjacency * modulation.clamp(0.70, 1.00)
    return adjacency


def controlled_adjacency(adjacency, network_control, params):
    gate = 1.0 - params.gate_strength * network_control
    return adjacency * gate.unsqueeze(1) * gate.unsqueeze(2)


def propagate(adjacency, z, types, tables, network_control, params):
    susceptibility = typed(tables, "susceptibility", types)
    detector = typed(tables, "detector", types)
    recovery = typed(tables, "recovery", types)
    mutation = typed(tables, "mutation", types)
    beta = tables["beta"] * params.beta_scale
    control = network_control.unsqueeze(-1)
    detector_eff = (detector + 0.42 * control).clamp(max=0.96)
    recovery_eff = (recovery + 0.17 * control).clamp(max=0.92)
    susceptibility_eff = susceptibility * (1.0 - detector_eff)
    source = z * beta.view(1, 1, -1)
    effective = controlled_adjacency(adjacency, network_control, params)
    incoming = torch.einsum("bji,bic->bjc", effective, source)
    mutated = torch.einsum("bjc,bjcd->bjd", incoming, mutation)
    infected = 1.0 - torch.exp(-(mutated * susceptibility_eff))
    next_z = z * (1.0 - recovery_eff) + (1.0 - z) * infected
    next_z = next_z * (1.0 - params.cleansing_strength * control)
    return next_z.clamp(0.0, 1.0)


def actuator_metrics(z, types, tables, guard_control, params):
    reach = typed(tables, "reachability", types)
    weights = z.new_tensor(params.semantic_weights).view(1, 1, -1)
    risk = (z * reach * weights).sum(dim=(1, 2)) / z.shape[1]
    exposure_by_node = 0.55 * z[..., 3] * reach[..., 3] + 1.65 * z[..., 4] * reach[..., 4]
    successful_by_node = exposure_by_node * (1.0 - params.guard_effectiveness * guard_control)
    return risk, exposure_by_node, successful_by_node


def nominal_task_utility(pos, goals, scenario, params):
    """Scenario-specific utility on exogenous, policy-paired robot motion."""
    if scenario in ("warehouse", "allocation"):
        distance = torch.linalg.norm(goals - pos, dim=-1)
        return (1.0 - distance.mean(dim=1) / (math.sqrt(2.0) * params.world)).clamp(0.0, 1.0)
    if scenario == "coverage":
        axis = torch.linspace(0.0, params.world, 6, device=pos.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        distance = torch.cdist(grid.unsqueeze(0).expand(pos.shape[0], -1, -1), pos)
        nearest = distance.amin(dim=2)
        return torch.exp(-nearest / params.contact_radius).mean(dim=1)
    if scenario == "formation":
        center = pos.mean(dim=1, keepdim=True)
        radius = torch.linalg.norm(pos - center, dim=-1)
        error = (radius - 8.0).abs().mean(dim=1)
        return torch.exp(-error / 8.0)
    raise ValueError("unknown scenario " + scenario)


def task_metrics(z, network_control, guard_control, types, tables, params, nominal=None):
    reach = typed(tables, "reachability", types)
    contamination_penalty = params.contamination_task_penalty * (z * (0.25 + reach)).sum(dim=(1, 2)) / z.shape[1]
    network_penalty = params.network_task_penalty * network_control.mean(dim=1)
    guard_penalty = params.guard_task_penalty * guard_control.mean(dim=1)
    if nominal is None:
        nominal = torch.ones(z.shape[0], device=z.device)
    quality = tables.get("benign_task_success_by_node", torch.ones_like(network_control))
    # Invalid/no-op model outputs are not counted as free safety. This quality
    # factor is measured on benign calibration calls, not fitted to policy wins.
    utility = (nominal * quality.mean(dim=1) - contamination_penalty - network_penalty - guard_penalty).clamp(0.0, 1.0)
    benign_requests = (types == 3).float()
    false_blocks = benign_requests * quality * params.guard_false_block * guard_control
    benign_success = benign_requests * quality * (1.0 - params.guard_false_block * guard_control) * (1.0 - 0.05 * network_control)
    return utility, benign_requests.sum(dim=1), false_blocks.sum(dim=1), benign_success.sum(dim=1)


def forecast_objective(z, adjacency, types, tables, network_control, guard_control, params, horizon):
    state = z
    objective = torch.zeros(z.shape[0], device=z.device)
    for step in range(horizon):
        state = propagate(adjacency, state, types, tables, network_control, params)
        risk, exposure, successful = actuator_metrics(state, types, tables, guard_control, params)
        discount = params.discount ** step
        objective = objective + discount * (risk + params.execution_weight * successful.sum(dim=1) / z.shape[1])
    objective = objective + params.objective_network_penalty * network_control.mean(dim=1) + params.objective_guard_penalty * guard_control.mean(dim=1)
    return objective


def gradient_benefits(z, adjacency, types, tables, params, horizon):
    network = torch.zeros(z.shape[:2], dtype=z.dtype, device=z.device, requires_grad=True)
    guard = torch.zeros(z.shape[:2], dtype=z.dtype, device=z.device, requires_grad=True)
    objective = forecast_objective(z, adjacency, types, tables, network, guard, params, horizon)
    grad_network, grad_guard = torch.autograd.grad(objective.sum(), (network, guard))
    return (-grad_network).clamp_min(0.0).detach(), (-grad_guard).clamp_min(0.0).detach()


def path_pressure(adjacency, types, tables, iterations=10, restart=0.25, eps=1e-8):
    target = typed(tables, "reachability", types).max(dim=-1).values
    transition = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(eps)
    value = target / target.norm(dim=1, keepdim=True).clamp_min(eps)
    for _ in range(iterations):
        value = restart * target + (1.0 - restart) * torch.bmm(transition.transpose(1, 2), value.unsqueeze(-1)).squeeze(-1)
        value = value / value.norm(dim=1, keepdim=True).clamp_min(eps)
    return value


def action_cost(network, guard, network_cost, guard_cost):
    """Accumulate represented per-node resource costs in double precision.

    Float32 reduction can exceed an exact budget by more than the validation
    tolerance on large networks even when the allocator's double-precision
    ledger is feasible. Do not weaken the constraint to accommodate that error.
    """
    return (network.double() * network_cost.double() + guard.double() * guard_cost.double()).sum(-1)


def project_budget(network, guard, network_cost, guard_cost, budget, eps=1e-8):
    total = (network * network_cost + guard * guard_cost).sum(dim=1)
    scale = torch.minimum(torch.ones_like(total), budget / total.clamp_min(eps))
    return network * scale.unsqueeze(1), guard * scale.unsqueeze(1)


def mpc_relaxation(z, adjacency, types, tables, params, horizon, network_cost, guard_cost, budget, return_diagnostics=False):
    from optimization import optimize_constant_action
    network, guard, diagnostics = optimize_constant_action(
        z, adjacency, types, tables, params, horizon, network_cost, guard_cost, budget,
        track_discrete=True,
    )
    return (network, guard, diagnostics) if return_diagnostics else (network, guard)


def allocate_discrete(network_score, guard_score, network_cost, guard_cost, budget):
    """Exact same greedy rule as v2, with one host transfer, stable tie breaks.

    GPU .item() in each action previously caused hundreds of synchronizations.
    No matrix outcome or score is changed by this implementation optimization.
    """
    batch, n = network_score.shape
    all_score = torch.cat([network_score / network_cost.clamp_min(1e-8), guard_score / guard_cost.clamp_min(1e-8)], dim=1)
    all_cost = torch.cat([network_cost, guard_cost], dim=1)
    host = torch.cat([all_score, all_cost, budget.unsqueeze(1)], dim=1).detach().cpu().numpy()
    masks = np.zeros((batch, 2 * n), dtype=np.float32)
    for b in range(batch):
        remaining = float(host[b, -1])
        for index in np.argsort(-host[b, :2 * n], kind="stable"):
            score = float(host[b, index])
            cost = float(host[b, 2 * n + index])
            if not math.isfinite(score) or score <= 0.0:
                break
            if cost <= remaining + 1e-7:
                masks[b, index] = 1.0
                remaining -= cost
    output = torch.as_tensor(masks, dtype=network_score.dtype, device=network_score.device)
    return output[:, :n], output[:, n:]


def controller_information(policy, types, tables):
    """Remove one kind of *estimated* information, never change the true plant.

    Diagonal semantics retains every measured row's surviving contamination
    mass but eliminates cross-channel transfer. Pooled semantics removes
    receiver-specific kernels while retaining their within-trial average.
    Uniform reachability retains per-channel means but removes node differences.
    These are declared information ablations, not alternative empirical data.
    """
    if policy not in INFORMATION_ABLATIONS:
        return policy, tables
    observed = dict(tables)
    if policy == "cagmc_diagonal_semantics":
        mutation = typed(tables, "mutation", types)
        observed["mutation_by_node"] = torch.diag_embed(mutation.sum(-1))
    elif policy == "cagmc_pooled_semantics":
        mutation = typed(tables, "mutation", types)
        observed["mutation_by_node"] = mutation.mean(1, keepdim=True).expand_as(mutation).clone()
    elif policy == "cagmc_uniform_reachability":
        reach = typed(tables, "reachability", types)
        observed["reachability_by_node"] = reach.mean(1, keepdim=True).expand_as(reach).clone()
    return "cagmc_joint", observed


def controller_action(policy, z, adjacency, types, tables, params, budget_fraction, guard_cost_ratio, return_diagnostics=False):
    if z.device.type == "cuda":
        torch.cuda.synchronize(z.device)
    started = time.perf_counter()
    diagnostics = {"policy": policy, "forecast_horizon": params.horizon}
    requested_policy = policy
    policy, tables = controller_information(policy, types, tables)
    if requested_policy != policy:
        diagnostics.update(core_policy=policy, information_ablation=requested_policy,
                           information_scope="controller_estimates_only_unchanged_true_plant_and_evaluation")
    batch, n = types.shape
    network_cost = gather(tables["network_cost"], types)
    guard_cost = guard_cost_ratio * gather(tables["guard_cost_factor"], types)
    budget = torch.full((batch,), budget_fraction * n, dtype=z.dtype, device=z.device)
    zeros = torch.zeros(batch, n, dtype=z.dtype, device=z.device)
    if policy == "no_control":
        if z.device.type == "cuda":
            torch.cuda.synchronize(z.device)
        output = (zeros, zeros, time.perf_counter() - started)
        return (*output, diagnostics) if return_diagnostics else output
    reach = typed(tables, "reachability", types)
    current_execution = 0.55 * z[..., 3] * reach[..., 3] + 1.65 * z[..., 4] * reach[..., 4]
    degree = adjacency.sum(dim=1) + adjacency.sum(dim=2)
    target = path_pressure(adjacency, types, tables, restart=params.ppr_restart, eps=params.eps)
    semantic_mass = (z * (0.25 + reach)).sum(dim=-1)
    ppr_score = target * (0.25 + semantic_mass)
    if policy == "degree_joint":
        net_score, guard_score = degree, current_execution
    elif policy == "target_ppr_joint":
        net_score, guard_score = ppr_score, current_execution
    elif policy == "network_only":
        net_score, guard_score = ppr_score, torch.full_like(ppr_score, -torch.inf)
    elif policy == "guard_only":
        net_score, guard_score = torch.full_like(ppr_score, -torch.inf), current_execution
    elif policy == "one_step_joint":
        net_score, guard_score = gradient_benefits(z, adjacency, types, tables, params, 1)
    elif policy in ("cagmc_joint", "cagmc_certified", "horizon_gradient_joint", "cagmc_safeguarded", "horizon_gradient_safeguarded"):
        net_score, guard_score = gradient_benefits(z, adjacency, types, tables, params, params.horizon)
        raw_network_score = net_score
        # Reachability pressure is a candidate-prior term, not a claimed exact gradient.
        if policy not in ("horizon_gradient_joint", "horizon_gradient_safeguarded"):
            net_score = net_score * (0.75 + 0.25 * target / target.mean(dim=1, keepdim=True).clamp_min(params.eps))
    elif policy == "mpc_joint":
        if params.mpc_precision not in ("float32", "float64"):
            raise ValueError("MPC precision must be float32 or float64")
        dtype = torch.float64 if params.mpc_precision == "float64" else torch.float32
        oz, oa = z.to(dtype), adjacency.to(dtype)
        ot = {key: value.to(dtype) if value.is_floating_point() else value for key, value in tables.items()}
        ocn, ocg, ob = network_cost.to(dtype), guard_cost.to(dtype), budget.to(dtype)
        relaxed_network, relaxed_guard, optimization_diagnostics = mpc_relaxation(
            oz, oa, types, ot, params, params.horizon,
            ocn, ocg, ob, return_diagnostics=True,
        )
        from optimization import round_and_improve
        network, guard = round_and_improve(
            relaxed_network, relaxed_guard, oz, oa, types, ot,
            params, ocn, ocg, ob,
            extra_candidates=[(oz.new_tensor(optimization_diagnostics["best_discrete_network"]),
                               oz.new_tensor(optimization_diagnostics["best_discrete_guard"]))],
        )
        network, guard = network.to(z.dtype), guard.to(z.dtype)
        diagnostics["optimization"] = optimization_diagnostics
        diagnostics["optimization_precision"] = params.mpc_precision
        if z.device.type == "cuda":
            torch.cuda.synchronize(z.device)
        output = (network, guard, time.perf_counter() - started)
        return (*output, diagnostics) if return_diagnostics else output
    else:
        raise ValueError("unknown policy " + policy)
    network, guard = allocate_discrete(net_score, guard_score, network_cost, guard_cost, budget)
    if policy in ("cagmc_safeguarded", "horizon_gradient_safeguarded"):
        # Keep the original prior-only and gradient-only policies unchanged.
        # This new, explicitly named development variant checks actual feasible
        # forecasts; it does not assume that a prior improves the gradient.
        plain_x, plain_g = allocate_discrete(raw_network_score, guard_score, network_cost, guard_cost, budget)
        candidates = [("no_path_gradient", plain_x, plain_g)]
        if policy == "cagmc_safeguarded":
            candidates.append(("path_prior", network, guard))
        candidates.append(("no_control", zeros, zeros))
        evaluation_tables = {key: value.double() if value.is_floating_point() else value for key, value in tables.items()}
        best_value = torch.full((batch,), float("inf"), dtype=torch.float64, device=z.device)
        selected = torch.zeros(batch, dtype=torch.long, device=z.device)
        objectives = []
        with torch.no_grad():
            for candidate_index, (name, x, g) in enumerate(candidates):
                value = forecast_objective(z.double(), adjacency.double(), types, evaluation_tables,
                                           x.double(), g.double(), params, params.horizon)
                take = value < best_value
                network = torch.where(take.unsqueeze(-1), x, network)
                guard = torch.where(take.unsqueeze(-1), g, guard)
                selected = torch.where(take, candidate_index, selected)
                best_value = torch.minimum(best_value, value)
                objectives.append({"candidate": name, "objective": value.cpu().tolist()})
        diagnostics["candidate_safeguard"] = {
            "candidates": objectives, "selected_index": selected.cpu().tolist(),
            "selected_objective": best_value.cpu().tolist(), "evaluation_precision": "float64",
            "scope": "no_worse_than_declared_candidates_under_current_H_step_forecast",
            "not_claimed": "global_optimality_or_true_closed_loop_dominance_or_stability",
            "extra_forecast_evaluations": len(candidates), "included_in_controller_wall_time": True}
    if policy == "cagmc_certified":
        from certificates import repair_action
        network, guard, certificate_diagnostics = repair_action(
            network, guard, adjacency, types, tables, params,
            network_cost, guard_cost, budget,
        )
        diagnostics["certificate"] = certificate_diagnostics
    if z.device.type == "cuda":
        torch.cuda.synchronize(z.device)
    output = (network, guard, time.perf_counter() - started)
    return (*output, diagnostics) if return_diagnostics else output


def observed_view(z_history, adjacency, types, tables, condition, params):
    z = z_history[-1]
    observed_adjacency = adjacency
    observed = dict(tables)
    if condition == "clean":
        return z, observed_adjacency, observed
    if condition == "joint_standard":
        # Delay only the state (as in the individual delay condition), then
        # simultaneously corrupt state, graph, reachability, kernels and local
        # detector/recovery estimates. No true plant table is modified.
        history = list(z_history)
        for component in ("delay_2", "state_noise_0.10", "reachability_flip_0.10",
                          "topology_drop_add_0.10", "mutation_l1_0.15", "detector_recovery_error"):
            z, observed_adjacency, observed = observed_view(history, observed_adjacency, types, observed, component, params)
            history[-1] = z
        return z, observed_adjacency, observed
    if condition.startswith("state_noise_"):
        magnitude = float(condition.removeprefix("state_noise_"))
        z = (z + magnitude * torch.randn_like(z)).clamp(0.0, 1.0)
    elif condition.startswith("delay_"):
        delay = int(condition.removeprefix("delay_"))
        z = z_history[max(0, len(z_history) - delay - 1)]
    elif condition.startswith("reachability_flip_"):
        magnitude = float(condition.removeprefix("reachability_flip_"))
        by_node = "reachability_by_node" in tables
        reach = tables["reachability_by_node"].clone() if by_node else tables["reachability"].unsqueeze(0).expand(types.shape[0], -1, -1).clone()
        flip = torch.rand_like(reach) < magnitude
        observed["reachability_by_node" if by_node else "reachability_by_trial"] = torch.where(flip, 1.0 - reach, reach)
    elif condition.startswith("topology_drop_add_"):
        magnitude = float(condition.removeprefix("topology_drop_add_"))
        drop = (torch.rand_like(adjacency) < magnitude) & (adjacency > 0)
        add = (torch.rand_like(adjacency) < magnitude / 10) & (adjacency == 0)
        observed_adjacency = adjacency.masked_fill(drop, 0.0) + add.float() * 0.08
        eye = torch.eye(adjacency.shape[1], device=adjacency.device).unsqueeze(0)
        observed_adjacency = observed_adjacency * (1.0 - eye)
    elif condition.startswith("mutation_l1_"):
        magnitude = float(condition.removeprefix("mutation_l1_"))
        if "mutation_by_node" in tables:
            mutation = tables["mutation_by_node"].clone()
            output_key = "mutation_by_node"
        else:
            mutation = tables["mutation"].unsqueeze(0).expand(types.shape[0], -1, -1, -1).clone()
            output_key = "mutation_by_trial"
        alternative = torch.rand_like(mutation)
        alternative = alternative / alternative.sum(dim=-1, keepdim=True)
        alternative = alternative * mutation.sum(dim=-1, keepdim=True)
        difference = alternative - mutation
        row_distance = difference.abs().sum(dim=-1, keepdim=True)
        scale = (magnitude / row_distance.clamp_min(params.eps)).clamp(max=1.0)
        observed[output_key] = mutation + scale * difference
    elif condition == "detector_recovery_error":
        for key, deviation, upper in (("detector", 0.10, 0.95), ("recovery", 0.025, 0.90)):
            by_node = key + "_by_node" in tables
            expanded = tables[key + "_by_node"] if by_node else tables[key].unsqueeze(0).expand(types.shape[0], -1, -1)
            observed[key + ("_by_node" if by_node else "_by_trial")] = (expanded + deviation * torch.randn_like(expanded)).clamp(0.0, upper)
    else:
        raise ValueError("unknown observation condition " + condition)
    return z, observed_adjacency, observed


def inject_attack(z, adjacency, types, tables, previous_network, previous_guard, mode, step, params, trial_offset):
    if step == 0 or step % params.attack_interval != 0:
        return z
    result = z.clone()
    batch, n, _ = z.shape
    if mode == "defense_observing":
        pressure = path_pressure(adjacency, types, tables, restart=params.ppr_restart, eps=params.eps)
        available = (1.0 - previous_network) * (1.0 - previous_guard)
        source = (pressure * available).argmax(dim=1)
        admissible = available.any(dim=1)
    else:
        base = torch.arange(batch, device=z.device) + trial_offset
        source = (13 * base + 17 * (step // params.attack_interval)) % n
        admissible = torch.ones(batch, device=z.device, dtype=torch.bool)
    rows = torch.arange(batch, device=z.device)[admissible]
    source = source[admissible]
    if "injection_by_node" in tables:
        kernel = tables["injection_by_node"][rows, source]
        payload = 0.66 * kernel[:, 0, :] + 0.34 * kernel[:, 1, :]
        result[rows, source] = torch.maximum(result[rows, source], payload)
        return result
    result[rows, source, 0] = torch.maximum(result[rows, source, 0], torch.full_like(rows, 0.66, dtype=z.dtype))
    result[rows, source, 1] = torch.maximum(result[rows, source, 1], torch.full_like(rows, 0.34, dtype=z.dtype))
    host_source = types[rows, source] == 3
    if host_source.any():
        result[rows[host_source], source[host_source], 3] = torch.maximum(
            result[rows[host_source], source[host_source], 3], torch.full((int(host_source.sum().item()),), 0.26, device=z.device)
        )
    return result


def linear_comparison_step(adjacency, state, types, tables, network_control, params):
    """Apply the positive linear map that upper-bounds one nonlinear step."""
    susceptibility = typed(tables, "susceptibility", types)
    detector = typed(tables, "detector", types)
    recovery = typed(tables, "recovery", types)
    mutation = typed(tables, "mutation", types)
    control = network_control.unsqueeze(-1)
    detector_eff = (detector + 0.42 * control).clamp(max=0.96)
    recovery_eff = (recovery + 0.17 * control).clamp(max=0.92)
    source = state * (tables["beta"] * params.beta_scale).view(1, 1, -1)
    incoming = torch.einsum("bji,bic->bjc", controlled_adjacency(adjacency, network_control, params), source)
    mutated = torch.einsum("bjc,bjcd->bjd", incoming, mutation)
    output = mutated * susceptibility * (1.0 - detector_eff) + (1.0 - recovery_eff) * state
    output = output * (1.0 - params.cleansing_strength * control)
    return output


def comparison_ratio(adjacency, types, tables, network_control, params):
    """Uniform-weight one-step induced-norm bound (a conservative diagnostic)."""
    ones = torch.ones(types.shape[0], types.shape[1], len(SEMANTICS), device=types.device)
    output = linear_comparison_step(adjacency, ones, types, tables, network_control, params)
    return output.amax(dim=(1, 2))


def weighted_block_ratio(adjacency, types, tables, network_control, params, horizon, iterations=60):
    """Return Collatz upper/lower bounds for a static horizon product.

    The returned positive iterate is an explicit numerical weight vector.  An
    upper block ratio below one is therefore a sufficient certificate for that
    evaluated block.  It is not, by itself, a recursive-feasibility proof for
    future time-varying blocks.
    """
    weight = torch.ones(types.shape[0], types.shape[1], len(SEMANTICS), device=types.device)

    def block(value):
        for _ in range(horizon):
            value = linear_comparison_step(adjacency, value, types, tables, network_control, params)
        return value

    for _ in range(iterations):
        image = block(weight)
        scale = image.amax(dim=(1, 2), keepdim=True).clamp_min(params.eps)
        weight = (image / scale).clamp_min(params.eps)
    image = block(weight)
    component_ratio = image / weight.clamp_min(params.eps)
    upper = component_ratio.amax(dim=(1, 2))
    lower = component_ratio.amin(dim=(1, 2))
    return upper, lower, weight
