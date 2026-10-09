"""Portable paired simulator with policy-independent exogenous random streams.

Formal calls require a frozen protocol and empirical receiver calibration.
Performance measurements live in batch-call records, never replicated as
independent trial timings. All attempted/failed runs retain manifests/traces.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from evidence_io import append_jsonl, digest, exclusive_gpu, utc_now, write_json
from experiment_protocol import archive_inputs, source_hashes, validate_protocol
from simulation_evidence import (assemble_views, attempts_for, completed_attempt,
                                 planned_units, run_lease, start_attempt, validate_resume)
from sim_core import (INFORMATION_ABLATIONS, Params, action_cost, actuator_metrics, attach_receiver_roles, build_tables, comparison_ratio,
                      contact_graph, controller_action, controller_information, gather, init_batch,
                      inject_attack, nominal_task_utility, observed_view,
                      propagate, slice_tables, task_metrics, typed, update_positions)

POLICIES = ("no_control", "degree_joint", "target_ppr_joint", "one_step_joint",
            "horizon_gradient_joint", "mpc_joint", "network_only", "guard_only",
            "cagmc_joint", "cagmc_certified", "cagmc_safeguarded", "horizon_gradient_safeguarded") + INFORMATION_ABLATIONS
SCENARIOS = ("warehouse", "coverage", "formation", "allocation")


def seed_for(*parts):
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


def environment(scenario, n, trial_ids, steps, base_seed, setting_id, params, device):
    """Generate each trial on CPU using its own seed, independent of batching."""
    episodes = []
    for trial in trial_ids:
        seed = seed_for(base_seed, setting_id, "environment", trial)
        torch.manual_seed(seed)
        types, pos, goals, domains, initial = init_batch(1, n, torch.device("cpu"))
        roles = torch.randint(0, 5, (1, n))
        positions, goal_history, role_history = [], [], []
        for step in range(steps):
            pos, goals = update_positions(pos, goals, scenario, params)
            positions.append(pos.clone())
            goal_history.append(goals.clone())
            role_history.append((roles + (step // params.role_switch_interval if params.role_switch_interval else 0)) % 5)
        episodes.append({"types": types, "roles": roles, "domains": domains, "initial": initial,
                         "positions": torch.stack(positions), "goals": torch.stack(goal_history),
                         "role_schedule": torch.stack(role_history), "seed": seed})
    merged = {key: torch.cat([row[key] for row in episodes], dim=1 if key in ("positions", "goals", "role_schedule") else 0)
              for key in ("types", "roles", "domains", "initial", "positions", "goals", "role_schedule")}
    env_hash = digest(b"".join(merged[key].numpy().tobytes() for key in sorted(merged)))
    return {key: value.to(device) for key, value in merged.items()}, [row["seed"] for row in episodes], env_hash


def observation(history, adjacency, types, tables, condition, params, trial_ids, setting_id, step, base_seed):
    # Independent per-trial streams ensure the same observed view is presented
    # across policies and is unaffected by batch size or earlier policy RNG use.
    if condition == "clean" or condition.startswith("delay_"):
        return observed_view(history, adjacency, types, tables, condition, params)
    observations = []
    for index, trial in enumerate(trial_ids):
        torch.manual_seed(seed_for(base_seed, setting_id, "observation", step, trial))
        view = observed_view([state[index:index+1] for state in history], adjacency[index:index+1],
                             types[index:index+1], slice_tables(tables, index), condition, params)
        observations.append(view)
    observed_tables = dict(tables)
    for key in observations[0][2]:
        if key.endswith(("_by_trial", "_by_node")):
            observed_tables[key] = torch.cat([entry[2][key] for entry in observations], 0)
    return torch.cat([entry[0] for entry in observations], 0), torch.cat([entry[1] for entry in observations], 0), observed_tables


def simulate(policy, config, trial_ids, env, env_seeds, env_hash, mutation,
             params, trace_stream=None, fail_at_step=-1, calibration=None, plant_calibration=None):
    device = env["initial"].device
    types, z = env["types"], env["initial"].clone()
    batch, n, _ = z.shape
    if params.role_switch_interval < 0:
        raise ValueError("role switch interval must be nonnegative")
    if params.role_switch_interval and (calibration is None or "receiver_type_role_matrices" not in calibration):
        raise ValueError("time-varying role kernels require receiver-role calibration")
    actual_calibration = calibration if plant_calibration is None else plant_calibration
    actual_mutation = mutation if plant_calibration is None else plant_calibration["receiver_type_matrices"]
    true_base = build_tables(device, actual_mutation, config["attack_mode"] == "transfer_shift")
    nominal_base = build_tables(device, mutation, False)
    true_tables, nominal_tables = true_base, nominal_base
    network, guard = torch.zeros(batch, n, device=device), torch.zeros(batch, n, device=device)
    sums = {key: torch.zeros(batch, device=device) for key in
            ("risk", "success", "exposure", "utility", "cost", "benign", "false_block", "benign_success", "certificates")}
    sums["cost"] = torch.zeros(batch, dtype=torch.float64, device=device)
    maximum_cost = torch.zeros(batch, dtype=torch.float64, device=device)
    history, calls = [], []
    decisions, controller_failures = 0, 0
    cn = gather(true_tables["network_cost"], types)
    cg = config["guard_cost_ratio"] * gather(true_tables["guard_cost_factor"], types)
    budget = n * config["budget_fraction"]
    previous_roles = None
    for step in range(config["steps"]):
        if step == fail_at_step:
            raise RuntimeError("deliberate development-only failure injection")
        pos, goals = env["positions"][step], env["goals"][step]
        roles = env["role_schedule"][step]
        if previous_roles is None or not torch.equal(roles, previous_roles):
            true_tables = attach_receiver_roles(true_base, actual_calibration, types, roles, config["attack_mode"] == "transfer_shift")
            nominal_tables = attach_receiver_roles(nominal_base, calibration, types, roles, False)
            previous_roles = roles
        adj = contact_graph(pos, env["domains"], config["scenario"], params)
        z = inject_attack(z, adj, types, true_tables, network, guard,
                          config["attack_mode"], step, params, trial_ids[0])
        history.append(z.clone())
        if len(history) > 16:
            history.pop(0)
        if step % params.control_interval == 0:
            oz, oa, ot = observation(history, adj, types, nominal_tables, config["observation_condition"],
                                     params, trial_ids, config.get("observation_stream_id", config["setting_id"]), step, config["base_seed"])
            started = time.perf_counter()
            network, guard, elapsed, controller_diagnostics = controller_action(policy, oz, oa, types, ot, params,
                                                         config["budget_fraction"], config["guard_cost_ratio"], return_diagnostics=True)
            _, controller_tables = controller_information(policy, types, ot)
            certificate = comparison_ratio(adj, types, true_tables, network, params)
            sums["certificates"] += (certificate <= params.certificate_mu).float()
            actual_cost = action_cost(network, guard, cn, cg)
            if bool((actual_cost > budget + 1e-5).any()):
                raise AssertionError("shared budget violated")
            record = {"setting_id": config["setting_id"], "policy": policy, "step": step,
                      "trial_ids": trial_ids, "batch_size": batch, "batch_id": f"{config['setting_id']}/{trial_ids[0]}/{step}/{policy}",
                      "controller_batch_seconds": elapsed, "measurement_unit": "one_batch_controller_call",
                      "certificate_scope": "actual_snapshot_diagnostic_not_future_envelope",
                      "controller_diagnostics": controller_diagnostics,
                      "receiver_roles": roles.cpu().tolist(),
                      "role_switch_interval": params.role_switch_interval,
                      "network_nodes": [torch.where(mask > .5)[0].cpu().tolist() for mask in network],
                      "guard_nodes": [torch.where(mask > .5)[0].cpu().tolist() for mask in guard],
                      "cost": actual_cost.cpu().tolist(), "snapshot_ratio": certificate.cpu().tolist(),
                      "state_observation_rmse": (oz - z).square().mean((1,2)).sqrt().cpu().tolist(),
                      "mutation_observation_row_l1_max": (typed(ot, "mutation", types) - typed(nominal_tables, "mutation", types)).abs().sum(-1).amax(dim=(1,2)).cpu().tolist(),
                      "total_mutation_model_mismatch_row_l1_max": (typed(controller_tables, "mutation", types) - typed(true_tables, "mutation", types)).abs().sum(-1).amax(dim=(1,2)).cpu().tolist(),
                      "controller_information_ablation": controller_diagnostics.get("information_ablation"),
                      "reachability_estimate_rmse": (typed(controller_tables, "reachability", types) - typed(true_tables, "reachability", types)).square().mean((1,2)).sqrt().cpu().tolist(),
                      "complete_call_utc": utc_now()}
            calls.append(record)
            if trace_stream is not None:
                append_jsonl(trace_stream, record)
            decisions += 1
        z = propagate(adj, z, types, true_tables, network, params)
        risk, exposure, success = actuator_metrics(z, types, true_tables, guard, params)
        utility, benign, false_block, benign_success = task_metrics(
            z, network, guard, types, true_tables, params,
            nominal_task_utility(pos, goals, config["scenario"], params))
        cost = action_cost(network, guard, cn, cg)
        values = {"risk": risk, "success": success.sum(-1), "exposure": exposure.sum(-1),
                  "utility": utility, "cost": cost, "benign": benign,
                  "false_block": false_block, "benign_success": benign_success}
        for key, value in values.items():
            sums[key] += value
        maximum_cost = torch.maximum(maximum_cost, cost)
    host = {key: value.detach().cpu().numpy() for key, value in sums.items()}
    maxcost = maximum_cost.cpu().numpy()
    rows = []
    for index, trial in enumerate(trial_ids):
        exposure = float(host["exposure"][index])
        success = float(host["success"][index])
        rows.append({"setting_id": config["setting_id"], "phase": config["phase"],
                     "scenario": config["scenario"], "n": n, "steps": config["steps"],
                     "trial_id": trial, "environment_seed": env_seeds[index], "environment_batch_sha256": env_hash,
                     "policy": policy, "status": "completed", "attack_mode": config["attack_mode"],
                     "observation_condition": config["observation_condition"],
                     "mean_actuator_risk": float(host["risk"][index]) / config["steps"],
                     "successful_exposure_burden": success / (n * config["steps"]),
                     "total_exposure_burden": exposure / (n * config["steps"]),
                     "conditional_pass_through": success / exposure if exposure > params.eps else None,
                     "unsafe_success_mass": success, "unsafe_exposure_mass": exposure,
                     "node_steps": n * config["steps"],
                     "task_utility_mean": float(host["utility"][index]) / config["steps"],
                     "mean_cost": float(host["cost"][index]) / config["steps"], "max_cost": float(maxcost[index]),
                     "budget": budget, "snapshot_certificate_fraction": float(host["certificates"][index]) / decisions,
                     "benign_request_mass": float(host["benign"][index]), "benign_false_block_mass": float(host["false_block"][index]),
                     "benign_success_mass": float(host["benign_success"][index]),
                     "controller_failures": controller_failures})
        rows[-1]["receiver_role_switches"] = (config["steps"] - 1) // params.role_switch_interval if params.role_switch_interval else 0
    return rows, calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--protocol")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inject-failure-step", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    params = Params(**config.get("params", {}))
    if args.inject_failure_step >= 0 and not args.development:
        raise SystemExit("failure injection is development-only")
    if not args.development and (not args.protocol or not args.calibration):
        raise SystemExit("formal experiments require frozen protocol and empirical calibration")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8")) if args.protocol else None
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else None
    mutation = calibration["receiver_type_matrices"] if calibration else None
    if not args.development:
        validate_protocol(protocol, "simulation", config, args.calibration,
                          {"device": args.device, "batch_size": args.batch_size})
    elif config.get("phase") != "development":
        raise SystemExit("development output cannot be labelled formal")
    for policy in config["policies"]:
        if policy not in POLICIES:
            raise ValueError("unknown policy " + policy)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    units = planned_units(config, args.batch_size)
    identity = {"development": args.development, "config": config, "params": asdict(params),
                "config_sha256": digest(Path(args.config)), "code_hashes": source_hashes("simulation"),
                "runtime": {"device": args.device, "batch_size": args.batch_size},
                "numerical_versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
                "protocol_sha256": digest(Path(args.protocol)) if args.protocol else None,
                "calibration_sha256": digest(Path(args.calibration)) if args.calibration else None}
    identity = json.loads(json.dumps(identity))  # Tuple-valued dataclass fields persist as JSON arrays.
    with run_lease(out), exclusive_gpu(device):
        if args.resume:
            if not (out / "manifest.json").exists():
                raise SystemExit("resume requires an existing authoritative run manifest")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            validate_resume(out, manifest, identity)
            if json.loads((out / "planned_batches.json").read_text(encoding="utf-8")) != units:
                raise ValueError("planned unit matrix changed")
            view = assemble_views(out, units, config)
            if manifest["status"] == "completed":
                if any(manifest[key] != value for key, value in view.items()):
                    raise ValueError("completed aggregate views do not match immutable unit evidence")
                print(json.dumps({"status": "already_completed_verified", "rows": view["completed_trial_rows"]}))
                return
            manifest.update(view)
            manifest.setdefault("resume_invocations", []).append({"utc": utc_now(), "prior_status": manifest["status"],
                "prior_error": manifest.get("error"), "rule": "all_unfinished_units_without_outcome_selection"})
            manifest.update(status="running", resumed_utc=utc_now())
            manifest.pop("error", None)
        else:
            if any(path.name != ".execution.lock" for path in out.iterdir()):
                raise SystemExit("refusing to overwrite evidence directory; use --resume with identical inputs")
            archive_inputs(out, "simulation", config, args.protocol, args.calibration)
            manifest = {**identity, "evidence_schema": 2, "status": "running", "started_utc": utc_now(),
                        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                        "expected_trial_rows": config["trials"] * len(config["policies"]), "completed_trial_rows": 0,
                        "aggregate_scope": "derived_views_from_immutable_unit_attempts_with_full_failure_ledger"}
            write_json(out / "planned_units.json", [{"policy": policy, "trial_id": trial} for policy in config["policies"] for trial in range(config["trials"])])
            write_json(out / "planned_batches.json", units)
        write_json(out / "manifest.json", manifest)
        try:
            cached_trials, env, seeds, env_hash = None, None, None, None
            for unit in units:
                history = attempts_for(out, unit, config)
                if completed_attempt(history) is not None:
                    continue
                trial_ids, policy = unit["trial_ids"], unit["policy"]
                if cached_trials != trial_ids:
                    env, seeds, env_hash = environment(config["scenario"], config["n"], trial_ids, config["steps"],
                                                        config["base_seed"], config.get("environment_stream_id", config["setting_id"]), params, device)
                    cached_trials = trial_ids
                attempt, attempt_manifest = start_attempt(out, unit, history)
                with (attempt / "controller_calls.jsonl").open("w", encoding="utf-8") as trace, (attempt / "trials.jsonl").open("w", encoding="utf-8") as stream:
                    try:
                        rows, _ = simulate(policy, config, trial_ids, env, seeds, env_hash, mutation, params,
                                           trace_stream=trace, fail_at_step=args.inject_failure_step, calibration=calibration)
                        for row in rows:
                            append_jsonl(stream, row)
                    except BaseException as error:
                        for trial in trial_ids:
                            append_jsonl(stream, {"setting_id": config["setting_id"], "policy": policy, "trial_id": trial,
                                                  "status": "failed", "error": type(error).__name__ + ": " + str(error), "utc": utc_now()})
                        attempt_manifest.update(status="failed", finished_utc=utc_now(), error=type(error).__name__ + ": " + str(error),
                            trials_sha256=digest(attempt / "trials.jsonl"), calls_sha256=digest(attempt / "controller_calls.jsonl"))
                        write_json(attempt / "manifest.json", attempt_manifest)
                        raise
                attempt_manifest.update(status="completed", finished_utc=utc_now(),
                    trials_sha256=digest(attempt / "trials.jsonl"), calls_sha256=digest(attempt / "controller_calls.jsonl"))
                write_json(attempt / "manifest.json", attempt_manifest)
                manifest["completed_trial_rows"] += len(rows)
                write_json(out / "manifest.json", manifest)
                print(json.dumps({"setting_id": config["setting_id"], "unit_id": unit["unit_id"], "attempt_id": attempt_manifest["attempt_id"],
                                  "completed_rows": manifest["completed_trial_rows"]}), flush=True)
            if source_hashes("simulation") != manifest["code_hashes"]:
                raise RuntimeError("source changed during the experiment")
            manifest.update(assemble_views(out, units, config))
            if manifest["completed_trial_rows"] != manifest["expected_trial_rows"]:
                raise RuntimeError("planned formal denominator is incomplete")
            manifest.update(status="completed", finished_utc=utc_now())
        except BaseException as error:
            manifest.update(assemble_views(out, units, config))
            manifest.update(status="failed", finished_utc=utc_now(), error=type(error).__name__ + ": " + str(error))
            write_json(out / "manifest.json", manifest)
            raise
        write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    main()
