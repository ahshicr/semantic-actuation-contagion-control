"""Family-clustered held-out chain prediction, with all baselines retained."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evidence_io import digest, utc_now, write_json

BASELINES = ("stationary", "identity", "frequency", "role_only")
FATES = ("prompt", "memory", "map", "tool", "actuation", "tag_absent", "protocol_rejected")


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    corrected = np.minimum(1., np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    answer = np.empty_like(values)
    answer[order] = corrected
    return answer


def summarize(frame, output, repetitions=2000, seed=713902):
    if frame.duplicated(["model_alias", "case_id", "depth"]).any():
        raise ValueError("duplicated prediction outcome")
    rows, depths = [], []
    rng = np.random.default_rng(seed)
    for (model, condition), group in frame.groupby(["model_alias", "condition"]):
        families = sorted(group.content_family.unique())
        weights = np.eye(len(families), dtype=int)[rng.integers(len(families), size=(repetitions, len(families)))].sum(1)
        # Every complete chain has the same prospective depth. Absorbed chains
        # remain in evaluation denominators but are not additional model calls.
        n_depths = group.groupby("case_id").depth.nunique()
        if n_depths.nunique() != 1:
            raise ValueError("incomplete chain prediction horizon")
        for baseline in BASELINES:
            differences = group.contextual_brier - group[baseline + "_brier"]
            family_differences = differences.groupby(group.content_family).mean().reindex(families).to_numpy()
            boot = weights @ family_differences / len(families)
            p = 1. if np.allclose(family_differences, 0) else float(wilcoxon(family_differences, zero_method="pratt", alternative="two-sided", method="auto").pvalue)
            rows.append({"model_alias": model, "condition": condition, "baseline": baseline,
                         "families": len(families), "model_case_chains": group.case_id.nunique(),
                         "outcome": "equal_weight_mean_brier_over_all_prespecified_depths",
                         "contextual_mean": float(group.contextual_brier.mean()),
                         "baseline_mean": float(group[baseline + "_brier"].mean()),
                         "paired_mean_difference": float(family_differences.mean()),
                         "family_bootstrap_low": float(np.quantile(boot, .025)),
                         "family_bootstrap_high": float(np.quantile(boot, .975)), "raw_p": p,
                         "families_contextual_better": int((family_differences < 0).sum())})
        for depth, part in group.groupby("depth"):
            for predictor in ("contextual", *BASELINES):
                probabilities = part[[predictor + "_prob_" + fate for fate in FATES]].to_numpy()
                if not np.allclose(probabilities.sum(1), 1) or (probabilities < -1e-10).any():
                    raise ValueError("invalid predicted probability distribution")
                truths = np.array([[fate == observed for fate in FATES] for observed in part.observed_fate], float)
                values = part[predictor + "_brier"].groupby(part.content_family).mean().reindex(families).to_numpy()
                boot = weights @ values / len(families)
                depths.append({"model_alias": model, "condition": condition, "depth": int(depth), "predictor": predictor,
                               "families": len(families), "chains": len(part), "actual_model_calls": int(part.actual_model_call.sum()),
                               "brier": float(values.mean()), "brier_low": float(np.quantile(boot, .025)),
                               "brier_high": float(np.quantile(boot, .975)),
                               "predicted_survival": float(probabilities[:, :5].sum(1).mean()),
                               "observed_survival": float(part.observed_survival.mean()),
                               "aggregate_distribution_l1_error": float(np.abs(probabilities.mean(0) - truths.mean(0)).sum())})
    effects = pd.DataFrame(rows)
    effects["holm_p_all_model_condition_baseline_comparisons"] = holm(effects.raw_p)
    effects.to_csv(output / "heldout_chain_effects.csv", index=False)
    pd.DataFrame(depths).to_csv(output / "heldout_depth_diagnostics.csv", index=False)
    return {"primary_comparisons": len(effects), "all_comparisons_retained": True,
            "bootstrap_unit": "whole heldout content family conditional on frozen calibration",
            "calibration_uncertainty_separate": True,
            "multiplicity": "Holm across all model-condition-baseline Brier comparisons, not per-depth p values",
            "rows_are_not_independent_calls": True, "families": frame.content_family.nunique()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("refusing to overwrite validation analysis")
    frame = pd.read_csv(args.predictions)
    result = summarize(frame, output)
    write_json(output / "manifest.json", {"status": "completed", "created_utc": utc_now(),
               "predictions_sha256": digest(Path(args.predictions)), "code_sha256": digest(Path(__file__)), **result})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
