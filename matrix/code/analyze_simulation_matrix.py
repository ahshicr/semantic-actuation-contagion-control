"""Full-grid, paired analysis of the frozen matrix, from immutable attempts.

All means and effects come from raw trial rows. The input audit verifies hashes,
complete attempts, pairing and accounting, but is not a fresh dynamical replay;
that stronger scope must be provided separately and must never be implied here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evidence_io import digest, utc_now, write_json
from simulation_evidence import attempts_for, completed_attempt, planned_units

OUTCOMES = ("mean_actuator_risk", "successful_exposure_burden", "task_utility_mean", "mean_cost_per_node", "benign_success_fraction")
DERIVED = {"successful_exposure_burden": ("unsafe_success_mass", "node_steps"),
           "total_exposure_burden": ("unsafe_exposure_mass", "node_steps"),
           "conditional_pass_through": ("unsafe_success_mass", "unsafe_exposure_mass")}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def require_close(actual, expected, label, atol=2e-7, rtol=2e-6):
    if actual is None or not np.isfinite(actual) or not np.isclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError("raw accounting mismatch: " + label)


def seed_for(*parts):
    value = hashlib.sha256(json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).digest()
    return int.from_bytes(value[:4], "little") % (2 ** 31 - 1)


def validate_trial(row, config, calls, trial_index):
    n, steps, params = config["n"], config["steps"], config["params"]
    for key in ("setting_id", "phase", "scenario", "n", "steps", "attack_mode", "observation_condition"):
        if row[key] != config[key]:
            raise ValueError("trial metadata differs: " + key)
    if row["status"] != "completed" or row["controller_failures"] != 0:
        raise ValueError("failed trial in complete analysis")
    expected_seed = seed_for(config["base_seed"], config.get("environment_stream_id", config["setting_id"]), "environment", row["trial_id"])
    if row["environment_seed"] != expected_seed or row["node_steps"] != n * steps:
        raise ValueError("environment seed or node-step denominator differs")
    budget = n * config["budget_fraction"]
    require_close(row["budget"], budget, "budget")
    interval, mu = params["control_interval"], params["certificate_mu"]
    costs, certificates = [], []
    for call in calls:
        if call["setting_id"] != config["setting_id"] or call["policy"] != row["policy"] or call["trial_ids"][trial_index] != row["trial_id"]:
            raise ValueError("controller-call/trial linkage differs")
        for field in ("network_nodes", "guard_nodes"):
            nodes = call[field][trial_index]
            if len(nodes) != len(set(nodes)) or any(not isinstance(node, int) or not 0 <= node < n for node in nodes):
                raise ValueError("invalid actual action indices")
        cost = call["cost"][trial_index]
        if not np.isfinite(cost) or cost < 0 or cost > budget + 1e-5:
            raise ValueError("controller call violates shared budget")
        duration = min(interval, steps - call["step"])
        costs.extend([cost] * duration)
        certificates.append(call["snapshot_ratio"][trial_index] <= mu)
    if len(costs) != steps:
        raise ValueError("held-action accounting does not cover exactly all steps")
    require_close(row["mean_cost"], np.mean(costs), "mean cost")
    require_close(row["max_cost"], max(costs), "maximum cost")
    require_close(row["snapshot_certificate_fraction"], np.mean(certificates), "snapshot certificate denominator")
    for output, (numerator, denominator) in DERIVED.items():
        if row[denominator] > params["eps"]:
            require_close(row[output], row[numerator] / row[denominator], output)
        elif row[output] is not None:
            raise ValueError("zero conditional denominator must remain undefined")
    for field in ("mean_actuator_risk", "unsafe_success_mass", "unsafe_exposure_mass", "benign_request_mass", "benign_false_block_mass", "benign_success_mass"):
        if not np.isfinite(row[field]) or row[field] < -1e-8:
            raise ValueError("negative or nonfinite simulation quantity: " + field)
    if row["unsafe_success_mass"] > row["unsafe_exposure_mass"] + 1e-5:
        raise ValueError("successful exposure exceeds attempted exposure")
    if row["benign_success_mass"] > row["benign_request_mass"] + 1e-5 or row["benign_false_block_mass"] > row["benign_request_mass"] + 1e-5:
        raise ValueError("benign outcome exceeds its request denominator")
    if not 0 <= row["task_utility_mean"] <= 1.000001:
        raise ValueError("invalid task utility")
    if row["benign_request_mass"] <= 0:
        # The model explicitly forces >=1 actuator host. Do not silently drop
        # such an impossible row or fill its undefined ratio with zero.
        raise ValueError("zero benign request denominator contradicts forced actuator-host model")
    return dict(row, analysis_family=config["analysis_family"], mean_cost_per_node=row["mean_cost"] / n,
                benign_success_fraction=row["benign_success_mass"] / row["benign_request_mass"])


def load_matrix(root):
    root = Path(root)
    matrix, bundle = read_json(root / "matrix_manifest.json"), read_json(root / "execution_bundle_manifest.json")
    protocol, plan = read_json(root / "frozen_protocol.json"), read_json(root / "analysis_plan.json")
    if matrix["status"] != "completed" or bundle["kind"] != "simulation_matrix":
        raise ValueError("complete frozen matrix required")
    if digest(root / "execution_bundle_manifest.json") != matrix["bundle_sha256"] or digest(root / "frozen_protocol.json") != bundle["protocol_sha256"]:
        raise ValueError("matrix bundle/protocol identity mismatch")
    if digest(root / "analysis_plan.json") != bundle["payload_sha256"]["analysis_plan.json"]:
        raise ValueError("declared analysis plan changed")
    configs = {item["setting_id"]: item["config"] for item in protocol["configurations"]}
    states = {item["setting_id"]: item for item in matrix["settings"]}
    if len(configs) != matrix["expected_settings"] or len(states) != len(matrix["settings"]) or set(states) != set(configs):
        raise ValueError("matrix setting grid incomplete or duplicated")
    rows, failures, source_hashes, paired_envs = [], [], {}, {}
    for setting, config in configs.items():
        run = root / "settings" / setting
        child = read_json(run / "manifest.json")
        if child["status"] != "completed" or digest(run / "manifest.json") != states[setting]["child_manifest_sha256"]:
            raise ValueError("child completion/manifest mismatch")
        if child["config"] != config or read_json(run / "configuration.json") != config or child["runtime"] != protocol["runtime"]:
            raise ValueError("child configuration/runtime mismatch")
        for name, expected in protocol["sources"].items():
            if child["code_hashes"].get(name) != expected or digest(run / "source_snapshot" / name) != expected:
                raise ValueError("child frozen source mismatch")
        for filename, field, expected in (("frozen_protocol.json", "protocol_sha256", bundle["protocol_sha256"]),
                                        ("semantic_calibration.json", "calibration_sha256", protocol["calibration_sha256"])):
            if child[field] != expected or digest(run / filename) != expected:
                raise ValueError("child protocol/calibration mismatch")
        for filename, field in (("trials.jsonl", "trials_sha256"), ("controller_calls.jsonl", "calls_sha256"), ("attempt_history.json", "attempt_history_sha256")):
            if digest(run / filename) != child[field]:
                raise ValueError("child aggregate evidence hash mismatch")
        units = planned_units(config, protocol["runtime"]["batch_size"])
        if read_json(run / "planned_batches.json") != units:
            raise ValueError("planned batch grid changed")
        all_trials, all_calls = iter(json_rows(run / "trials.jsonl")), iter(json_rows(run / "controller_calls.jsonl"))
        visited, child_rows, child_calls, history_records = set(), 0, 0, []
        for unit in units:
            history = attempts_for(run, unit, config)
            history_records.extend(history)
            chosen = completed_attempt(history)
            if chosen is None:
                raise ValueError("unfinished planned unit")
            failures.extend(dict(item, setting_id=setting) for item in history if item["status"] != "completed")
            attempt = run / chosen["relative_directory"]
            calls = list(json_rows(attempt / "controller_calls.jsonl"))
            for call in calls:
                expected = dict(call, unit_id=unit["unit_id"], attempt_id=chosen["attempt_id"])
                if next(all_calls, None) != expected:
                    raise ValueError("aggregate call view differs from immutable successful attempt")
            for trial_index, row in enumerate(json_rows(attempt / "trials.jsonl")):
                expected = dict(row, unit_id=unit["unit_id"], attempt_id=chosen["attempt_id"])
                if next(all_trials, None) != expected:
                    raise ValueError("aggregate trial view differs from immutable successful attempt")
                key = row["policy"], row["trial_id"]
                if key in visited:
                    raise ValueError("duplicate policy-trial row")
                visited.add(key)
                signature = (config["scenario"], config["n"], config["base_seed"], config.get("environment_stream_id", setting),
                             config["steps"], config["params"]["speed"], config["params"]["role_switch_interval"], tuple(unit["trial_ids"]))
                if paired_envs.setdefault(signature, row["environment_batch_sha256"]) != row["environment_batch_sha256"]:
                    raise ValueError("policies or comparable settings did not share the same exogenous environment")
                rows.append(validate_trial(expected, config, calls, trial_index))
                child_rows += 1
            child_calls += len(calls)
        if next(all_trials, None) is not None or next(all_calls, None) is not None:
            raise ValueError("extra derived rows outside planned units")
        expected_keys = {(policy, trial) for policy in config["policies"] for trial in range(config["trials"])}
        if visited != expected_keys or child_rows != child["completed_trial_rows"] or child_rows != child["expected_trial_rows"]:
            raise ValueError("complete paired trial denominator mismatch")
        ledger = read_json(run / "attempt_history.json")
        if ledger["attempts"] != history_records or ledger["completed_trial_rows"] != child_rows:
            raise ValueError("attempt ledger differs from preserved attempt manifests")
        source_hashes[setting] = {"manifest": digest(run / "manifest.json"), "trials": child["trials_sha256"], "calls": child["calls_sha256"], "attempt_history": child["attempt_history_sha256"]}
        print(json.dumps({"verified_setting": setting, "trial_rows": child_rows, "batch_calls_not_independent_trials": child_calls}), flush=True)
    if len(rows) != matrix["expected_trial_rows"]:
        raise ValueError("matrix trial-row denominator mismatch")
    return pd.DataFrame(rows), configs, plan, {"scope": "all immutable attempts, complete grids, exact hashes, pairing and arithmetic accounting; not dynamical replay",
        "source_matrix_manifest_sha256": digest(root / "matrix_manifest.json"), "source_settings": source_hashes,
        "preserved_noncompleted_attempts": failures, "verified_policy_trial_rows": len(rows)}


def holm(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    result[order] = np.minimum(1., np.maximum.accumulate(np.asarray(values)[order] * np.arange(len(values), 0, -1)))
    return result


def summarize(frame, configs, plan, out):
    out = Path(out)
    if list(plan["inferential_outcomes"]) != list(OUTCOMES):
        raise ValueError("unsupported change to declared outcomes")
    bank, summaries, comparisons, bank_metadata = {}, [], [], []
    required = {(setting, policy, trial) for setting, config in configs.items() for policy in config["policies"] for trial in range(config["trials"])}
    actual = set(zip(frame.setting_id, frame.policy, frame.trial_id))
    if actual != required or len(frame) != len(required):
        raise ValueError("analysis grid is missing, duplicated or unplanned")
    for setting, config in configs.items():
        block = frame[frame.setting_id == setting]
        ntrials = config["trials"]
        bank_key = (config["scenario"], config["n"], ntrials)
        if bank_key not in bank:
            seed = seed_for(plan["analysis_seed"], *bank_key, "paired_trial_bootstrap")
            rng = np.random.default_rng(seed)
            draws = rng.integers(ntrials, size=(plan["bootstrap_replicates"], ntrials))
            bank[bank_key] = np.stack([(draws == index).sum(1) for index in range(ntrials)], axis=1)
            name = f"weights_{config['scenario']}_n{config['n']}_trials{ntrials}"
            bank_metadata.append({"array_name": name, "scenario": bank_key[0], "n": bank_key[1], "trials": ntrials, "seed": seed})
        weights = bank[bank_key]
        meta = {key: config[key] for key in ("setting_id", "analysis_family", "scenario", "n", "trials", "steps")}
        values = {}
        for policy in config["policies"]:
            rows = block[block.policy == policy].set_index("trial_id").reindex(range(ntrials))
            for metric in OUTCOMES:
                vector = rows[metric].to_numpy(dtype=float)
                if not np.isfinite(vector).all():
                    raise ValueError("nonfinite or missing outcome, no selective complete-case fallback: " + metric)
                values[policy, metric] = vector
                boot = weights @ vector / ntrials
                summaries.append({**meta, "policy": policy, "outcome": metric, "mean": float(vector.mean()),
                    "median": float(np.median(vector)), "minimum": float(vector.min()), "maximum": float(vector.max()),
                    "pointwise_bootstrap_low": float(np.quantile(boot, .025)), "pointwise_bootstrap_high": float(np.quantile(boot, .975))})
        for method, baseline in plan["contrasts_by_setting"][setting]:
            for metric in OUTCOMES:
                delta = values[method, metric] - values[baseline, metric]
                tested = np.where(np.abs(delta) <= 1e-12, 0., delta)
                boot = weights @ delta / ntrials
                p = 1. if np.all(tested == 0) else float(wilcoxon(tested, zero_method="pratt", alternative="two-sided", method="auto").pvalue)
                benefit = tested > 0 if metric in ("task_utility_mean", "benign_success_fraction") else tested < 0
                comparisons.append({**meta, "method": method, "baseline": baseline, "outcome": metric,
                    "holm_family": "primary" if config["analysis_family"] == "primary" else "secondary",
                    "paired_mean_difference_method_minus_baseline": float(delta.mean()), "paired_median_difference": float(np.median(delta)),
                    "pointwise_bootstrap_low": float(np.quantile(boot, .025)), "pointwise_bootstrap_high": float(np.quantile(boot, .975)),
                    "raw_signed_rank_p": p, "favorable_trial_fraction": float(benefit.mean()),
                    "tie_trial_fraction": float((tested == 0).mean()), "negative_differences": int((tested < 0).sum()),
                    "zero_differences": int((tested == 0).sum()), "positive_differences": int((tested > 0).sum())})
    effects = pd.DataFrame(comparisons)
    for family in ("primary", "secondary"):
        if len(effects):
            mask = effects.holm_family == family
            count = int(mask.sum())
        else:
            count = 0
        if count != plan["expected_holm_family_sizes"][family]:
            raise ValueError("declared Holm family denominator differs: " + family)
        if count:
            effects.loc[mask, "holm_p_whole_declared_family"] = holm(effects.loc[mask, "raw_signed_rank_p"].to_numpy())
    pd.DataFrame(summaries).to_csv(out / "all_policy_summaries.csv", index=False)
    effects.to_csv(out / "all_paired_comparisons.csv", index=False)
    frame.to_csv(out / "all_trial_analysis_inputs.csv", index=False)
    descriptive = ["total_exposure_burden", "conditional_pass_through", "snapshot_certificate_fraction", "benign_false_block_mass"]
    frame.groupby(["setting_id", "policy"])[descriptive].agg(["count", "mean", "min", "max"]).to_csv(out / "descriptive_diagnostics.csv")
    np.savez_compressed(out / "paired_environment_bootstrap_weights.npz", **{item["array_name"]: bank[(item["scenario"], item["n"], item["trials"])] for item in bank_metadata})
    write_json(out / "bootstrap_bank_metadata.json", {"banks": bank_metadata, "repetitions": plan["bootstrap_replicates"],
        "overlap": "primary96 and secondary48 share underlying environments; not pooled or independent replications"})
    return {"settings": len(configs), "policy_trial_rows_not_independent_environments": len(frame),
            "unique_initial_environment_seeds": frame[["scenario", "n", "environment_seed"]].drop_duplicates().shape[0],
            "policy_summary_rows": len(summaries), "paired_comparison_rows": len(effects),
            "holm_family_sizes": plan["expected_holm_family_sizes"], "all_declared_comparisons_retained": True,
            "bootstrap_unit": "whole paired environment trial within setting, shared banks per scene/N/trial-count",
            "inference_scope": "fixed declared scenes, point-calibrated simulation, pointwise percentile intervals; signed-rank symmetry test not specifically a mean test",
            "not_claimed": "independent policy/time-step samples, simultaneous confidence bands, measured robot missions, future stability from snapshot ratios, or isolated runtime measurements"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("refusing to overwrite matrix analysis")
    try:
        frame, configs, plan, input_audit = load_matrix(args.run)
        write_json(out / "complete_input_accounting_audit.json", dict(input_audit, status="passed", audited_utc=utc_now()))
        report = summarize(frame, configs, plan, out)
        report.update(status="completed", created_utc=utc_now(), source_matrix_manifest_sha256=digest(Path(args.run) / "matrix_manifest.json"),
            analysis_plan_sha256=digest(Path(args.run) / "analysis_plan.json"),
            source_sha256={name: digest(Path(__file__).parent / name) for name in ("analyze_simulation_matrix.py", "simulation_evidence.py", "evidence_io.py")},
            output_sha256={path.name: digest(path) for path in out.iterdir() if path.is_file()})
        write_json(out / "analysis_manifest.json", report)
        print(json.dumps({key: report[key] for key in ("status", "settings", "paired_comparison_rows")}))
    except BaseException as error:
        write_json(out / "analysis_manifest.json", {"status": "failed", "created_utc": utc_now(), "error": type(error).__name__ + ": " + str(error)})
        raise


if __name__ == "__main__":
    main()
