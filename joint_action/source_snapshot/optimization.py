"""Budget-projected, backtracked multi-step optimization on the actual plant.

This is a constant-action receding-horizon baseline, not an exact global solver
or a full H-action-sequence optimizer. Its convergence diagnostics and rounding
are exposed rather than inferred from an arbitrary iteration count.
"""
import torch


def project_box_budget_bisection(values, costs, budget, iterations=45):
    """Euclidean projection onto 0 <= u <= 1, costs @ u <= budget.

    KKT form: u_i = clip(v_i - multiplier * c_i, 0, 1). The multiplier is
    independently bisected for every trial. Unlike radial scaling this is the
    true Euclidean projection used in the projected-gradient mapping.
    """
    if (costs <= 0).any() or (budget < 0).any():
        raise ValueError("positive costs and nonnegative budgets required")
    clipped = values.clamp(0.0, 1.0)
    active = (clipped * costs).sum(-1) > budget
    low = torch.zeros_like(budget)
    high = (values / costs).amax(-1).clamp_min(0.0)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        proposed = (values - middle.unsqueeze(-1) * costs).clamp(0.0, 1.0)
        excess = (proposed * costs).sum(-1) > budget
        low = torch.where(excess, middle, low)
        high = torch.where(excess, high, middle)
    projected = (values - high.unsqueeze(-1) * costs).clamp(0.0, 1.0)
    return torch.where(active.unsqueeze(-1), projected, clipped)


def project_box_budget(values, costs, budget, iterations=45):
    """Exact piecewise-linear KKT root from sorted breakpoints.

    This removes dozens of sequential GPU kernel launches per projection.
    Float64 cumulative sums avoid cancellation in the breakpoint sweep. The
    retained bisection implementation is an independent numerical reference.
    """
    if not torch.isfinite(values).all() or not torch.isfinite(costs).all() or not torch.isfinite(budget).all():
        raise ValueError("finite projection inputs required")
    if (costs <= 0).any() or (budget < 0).any():
        raise ValueError("positive costs and nonnegative budgets required")
    v, c, b = values.double(), costs.double(), budget.double()
    clipped = v.clamp(0., 1.)
    initial = (clipped * c).sum(-1)
    zero = torch.zeros_like(b).unsqueeze(-1)
    locations = torch.cat([((v - 1) / c).clamp_min(0), (v / c).clamp_min(0), zero], -1)
    deltas = torch.cat([-c.square(), c.square(), zero], -1)
    locations, order = locations.sort(dim=-1, stable=True)
    slopes = deltas.gather(-1, order).cumsum(-1)
    changes = slopes[:, :-1] * (locations[:, 1:] - locations[:, :-1])
    totals = initial.unsqueeze(-1) + torch.cat([zero, changes.cumsum(-1)], -1)
    index = (totals > b.unsqueeze(-1)).sum(-1).sub(1).clamp(0, locations.shape[-1] - 1).unsqueeze(-1)
    location = locations.gather(-1, index).squeeze(-1)
    total = totals.gather(-1, index).squeeze(-1)
    negative_slope = (-slopes.gather(-1, index).squeeze(-1)).clamp_min(1e-15)
    multiplier = (location + (total - b) / negative_slope).clamp_min(0)
    projected = (v - multiplier.unsqueeze(-1) * c).clamp(0., 1.)
    projected = torch.where((initial <= b).unsqueeze(-1), clipped, projected)
    projected = torch.where((b == 0).unsqueeze(-1), torch.zeros_like(projected), projected)
    return projected.to(values.dtype)


def optimize_constant_action(z, adjacency, types, tables, params, horizon,
                             network_cost, guard_cost, budget,
                             starts=("zero", "gradient", "uniform"), record_history=False,
                             track_discrete=False):
    from sim_core import allocate_discrete, forecast_objective, gradient_benefits
    n = z.shape[1]
    if params.mpc_iterations < 1 or not starts:
        raise ValueError("at least one iteration and initialization are required")
    costs = torch.cat([network_cost, guard_cost], -1)

    def objective(action):
        return forecast_objective(z, adjacency, types, tables, action[:, :n],
                                  action[:, n:], params, horizon)

    best_value = torch.full_like(budget, float("inf"))
    best_action = torch.zeros_like(costs)
    best_discrete = torch.zeros_like(costs)
    best_discrete_value = objective(best_discrete).detach()
    discrete_checkpoints = []

    def consider_discrete(action, start, iteration):
        nonlocal best_discrete, best_discrete_value
        with torch.enable_grad():
            nx, ng = round_and_improve(action[:, :n], action[:, n:], z, adjacency, types,
                                       tables, params, network_cost, guard_cost, budget, horizon=horizon)
        with torch.no_grad():
            candidate = torch.cat([nx, ng], -1)
            value = objective(candidate)
            take = value < best_discrete_value
            best_discrete = torch.where(take.unsqueeze(-1), candidate, best_discrete)
            best_discrete_value = torch.minimum(best_discrete_value, value)
        discrete_checkpoints.append({"start": start, "iteration": iteration,
                                     "best_feasible_objective": best_discrete_value.cpu().tolist()})
    starts_diagnostics = []
    for start in starts:
        if start == "zero":
            action = torch.zeros_like(costs)
        elif start == "gradient":
            gn, gg = gradient_benefits(z, adjacency, types, tables, params, horizon)
            an, ag = allocate_discrete(gn, gg, network_cost, guard_cost, budget)
            action = torch.cat([an, ag], -1)
        elif start == "uniform":
            action = project_box_budget(torch.full_like(costs, 0.5), costs, budget)
        else:
            raise ValueError("unknown optimizer initialization " + start)
        if track_discrete:
            consider_discrete(action, start, 0)
        step = torch.full_like(budget, params.mpc_step_size * n)
        history = []
        rejected = torch.zeros_like(budget, dtype=torch.long)
        for iteration in range(params.mpc_iterations):
            variable = action.detach().requires_grad_(True)
            value = objective(variable)
            gradient, = torch.autograd.grad(value.sum(), variable)
            with torch.no_grad():
                projected = project_box_budget(variable - n * gradient, costs, budget)
                # Scale out the per-node objective normalization for the stop rule.
                mapping = (variable - projected).abs().amax(-1)
                if record_history:
                    history.append({"iteration": iteration,
                                    "objective": value.detach().cpu().tolist(),
                                    "scaled_projected_gradient_inf": mapping.cpu().tolist()})
                if bool((mapping <= params.mpc_tolerance).all()):
                    action = variable.detach()
                    break
                # Freeze converged trials individually, so stopping of a trial
                # is not changed by slower-converging neighbours in its batch.
                accepted = mapping <= params.mpc_tolerance
                next_action = variable.detach().clone()
                for _ in range(16):
                    candidate = project_box_budget(variable - step.unsqueeze(-1) * gradient,
                                                   costs, budget)
                    candidate_value = objective(candidate)
                    directional = (gradient * (candidate - variable)).sum(-1)
                    sufficient = candidate_value <= value.detach() + 1e-4 * directional + 1e-8
                    take = (~accepted) & sufficient
                    next_action = torch.where(take.unsqueeze(-1), candidate, next_action)
                    accepted |= sufficient
                    if bool(accepted.all()):
                        break
                    step = torch.where(accepted, step, step * 0.5)
                rejected += (~accepted).long()
                action = next_action
                step = (step * 1.15).clamp(max=n * 32.0)
            if track_discrete and iteration + 1 in (10, 50, 100, 200, 300, 500):
                consider_discrete(action, start, iteration + 1)
        if track_discrete:
            consider_discrete(action, start, iteration + 1)
        variable = action.detach().requires_grad_(True)
        value = objective(variable)
        gradient, = torch.autograd.grad(value.sum(), variable)
        with torch.no_grad():
            mapping = (variable - project_box_budget(variable - n * gradient, costs, budget)).abs().amax(-1)
            take = value < best_value
            best_value = torch.where(take, value, best_value)
            best_action = torch.where(take.unsqueeze(-1), variable.detach(), best_action)
        starts_diagnostics.append({
            "start": start, "iterations": iteration + 1,
            "objective": value.detach().cpu().tolist(),
            "scaled_projected_gradient_inf": mapping.detach().cpu().tolist(),
            "converged": (mapping <= params.mpc_tolerance).detach().cpu().tolist(),
            "line_search_failures": rejected.detach().cpu().tolist(),
            "fractional_cost": (action * costs).sum(-1).detach().cpu().tolist(),
            "history": history,
        })
    return best_action[:, :n], best_action[:, n:], {
        "objective": best_value.detach().cpu().tolist(),
        "starts": starts_diagnostics,
        "objective_scope": "constant_action_H_step_nonlinear_objective",
        "global_optimality_claimed": False,
        "discrete_checkpoint_selection": track_discrete,
        "best_discrete_network": best_discrete[:, :n].cpu().tolist() if track_discrete else None,
        "best_discrete_guard": best_discrete[:, n:].cpu().tolist() if track_discrete else None,
        "best_discrete_objective": best_discrete_value.cpu().tolist() if track_discrete else None,
        "discrete_checkpoints": discrete_checkpoints,
    }


def round_and_improve(network, guard, z, adjacency, types, tables, params,
                      network_cost, guard_cost, budget, extra_candidates=(), horizon=None):
    """Choose the best feasible candidate under the actual objective.

    Candidate construction and objective selection are shared across trials.
    Fractional actions are not interpreted as marginal benefits. Baseline
    zero-action-gradient rounding is retained, so optimization cannot silently
    discard that feasible reference solely due to a poor relaxation rounding.
    """
    from sim_core import allocate_discrete, forecast_objective, gradient_benefits
    horizon = params.horizon if horizon is None else horizon
    candidates = [allocate_discrete(network * network_cost, guard * guard_cost,
                                    network_cost, guard_cost, budget)]
    candidates.extend(extra_candidates)
    x = network.detach().requires_grad_(True)
    g = guard.detach().requires_grad_(True)
    objective = forecast_objective(z, adjacency, types, tables, x, g, params, horizon)
    gx, gg = torch.autograd.grad(objective.sum(), (x, g))
    candidates.append(allocate_discrete((-gx).clamp_min(0), (-gg).clamp_min(0),
                                        network_cost, guard_cost, budget))
    gn, gg = gradient_benefits(z, adjacency, types, tables, params, horizon)
    candidates.append(allocate_discrete(gn, gg, network_cost, guard_cost, budget))
    candidates.append((torch.zeros_like(network), torch.zeros_like(guard)))
    best = torch.full_like(budget, float("inf"))
    selected_x, selected_g = candidates[-1]
    with torch.no_grad():
        for x, g in candidates:
            value = forecast_objective(z, adjacency, types, tables, x, g, params, horizon)
            take = value < best
            selected_x = torch.where(take.unsqueeze(-1), x, selected_x)
            selected_g = torch.where(take.unsqueeze(-1), g, selected_g)
            best = torch.minimum(best, value)
    return selected_x.detach(), selected_g.detach()
