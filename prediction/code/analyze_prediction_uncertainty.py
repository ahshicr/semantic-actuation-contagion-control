"""Calibration-family and heldout-family uncertainty in long-chain predictions.

Observed absorbed states remain evaluation outcomes, never extra model calls.
Dirichlet sensitivity ranges are not labelled frequentist confidence intervals.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_io import digest, utc_now, write_json
from semantic_uncertainty import FATES, PREDICTORS, load_draw_sets, prediction_draws


def propagate_uncertainty(frame, point, draws, label, output, seed=871229):
    families = sorted(frame.content_family.unique())
    if set(families) & set(point["content_families"]):
        raise ValueError("calibration and heldout family overlap")
    if frame.duplicated(["model_alias", "case_id", "depth"]).any():
        raise ValueError("duplicate prediction outcome")
    depth = int(frame.depth.max())
    if not frame.groupby(["model_alias", "case_id"]).depth.nunique().eq(depth).all():
        raise ValueError("incomplete common prediction horizon")
    count = len(draws["forward"])
    family_index = {family: index for index, family in enumerate(families)}
    conditions = sorted(frame.condition.unique())
    model_names = list(point["models"])
    # Shape: draw, model, condition, predictor, held-out family.
    scores = np.zeros((count, len(model_names), len(conditions), len(PREDICTORS), len(families)))
    denominators = np.zeros((len(model_names), len(conditions), len(families)))
    probability_ranges = []
    group_keys = ["model_alias", "condition", "receiving_role", "source_channel"]
    for (model_name, condition, role, source_name), group in frame.groupby(group_keys):
        model, condition_index = model_names.index(model_name), conditions.index(condition)
        source = FATES.index(source_name)
        probabilities = prediction_draws(draws["forward"], draws["injection"], model, int(role), source, condition, depth)
        # Before propagating uncertainty, verify the same equations reproduce
        # every archived point prediction, not a more favourable new predictor.
        point_probabilities = prediction_draws(np.asarray(point["forward_fate_probabilities"])[None],
            np.asarray(point["injection_fate_probabilities"])[None], model, int(role), source, condition, depth)
        for level, part in group.groupby("depth"):
            truth_counts = np.zeros((len(families), 7))
            for (family, fate), size in part.groupby(["content_family", "observed_fate"]).size().items():
                truth_counts[family_index[family], FATES.index(fate)] = size
            totals = truth_counts.sum(-1)
            denominators[model, condition_index] += totals
            for predictor_index, predictor in enumerate(PREDICTORS):
                archived = part[[predictor + "_prob_" + fate for fate in FATES]].to_numpy()
                expected = point_probabilities[predictor][0, int(level) - 1]
                if not np.allclose(archived, expected[None], atol=1e-10, rtol=0):
                    raise ValueError("uncertainty equations do not match frozen point predictions")
                p = probabilities[predictor][:, int(level) - 1]
                # Sum over actual outcomes; no random outcomes are invented.
                total_brier = (np.square(p).sum(-1)[:, None] + 1.) * totals[None] - 2 * (p @ truth_counts.T)
                scores[:, model, condition_index, predictor_index] += total_brier
                if predictor == "contextual":
                    low, median, high = np.quantile(p, [.025, .5, .975], axis=0)
                    for fate_index, fate in enumerate(FATES):
                        probability_ranges.append({"uncertainty": label, "model_alias": model_name, "condition": condition,
                            "receiving_role": int(role), "source_channel": source_name, "depth": int(level), "fate": fate,
                            "point_probability": float(expected[fate_index]), "draw_median": float(median[fate_index]),
                            "draw_low": float(low[fate_index]), "draw_high": float(high[fate_index]),
                            "observed_count": int(truth_counts[:, fate_index].sum()), "heldout_outcomes": int(totals.sum()),
                            "actual_model_calls": int(part.actual_model_call.sum()),
                            "interval_scope": "calibration_family_bootstrap" if label == "family_bootstrap" else "model_based_prior_sensitivity_not_frequentist_CI"})
    if (denominators <= 0).any():
        raise ValueError("missing heldout model/condition/family denominator")
    scores /= denominators[None, :, :, None, :]
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(families), size=(count, len(families)))
    weights = np.stack([(selected == index).sum(1) for index in range(len(families))], axis=1)
    calibration_only = scores.mean(-1)
    joint = np.einsum("dmcpf,df->dmcp", scores, weights) / len(families)
    effects = []
    for model, model_name in enumerate(model_names):
        for condition_index, condition in enumerate(conditions):
            for baseline_index, baseline in enumerate(PREDICTORS[1:], start=1):
                for scope, values in (("calibration_only_fixed_heldout_families", calibration_only),
                                      ("joint_independent_calibration_and_heldout_family_resampling", joint)):
                    differences = values[:, model, condition_index, 0] - values[:, model, condition_index, baseline_index]
                    effects.append({"uncertainty": label, "scope": scope, "model_alias": model_name, "condition": condition,
                        "baseline": baseline, "draws_not_independent_experiments": count, "heldout_families": len(families),
                        "difference_draw_median": float(np.median(differences)), "difference_low": float(np.quantile(differences, .025)),
                        "difference_high": float(np.quantile(differences, .975)), "draw_fraction_contextual_better": float((differences < 0).mean()),
                        "fraction_scope": "uncertainty_draw_fraction_not_p_value_or_empirical_success_rate"})
    pd.DataFrame(effects).to_csv(output / (label + "_paired_effect_ranges.csv"), index=False)
    pd.DataFrame(probability_ranges).to_csv(output / (label + "_probability_ranges.csv"), index=False)
    np.savez_compressed(output / (label + "_score_draws.npz"), calibration_only=calibration_only, joint=joint,
                        heldout_family_indices=selected, model_aliases=np.array(model_names), conditions=np.array(conditions),
                        predictors=np.array(PREDICTORS), heldout_families=np.array(families))
    return {"label": label, "draws": count, "families": len(families), "effect_rows": len(effects),
            "probability_rows": len(probability_ranges), "resampling_seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("refusing to overwrite uncertainty analysis")
    point, draw_sets, calibration_manifest = load_draw_sets(args.calibration)
    predictions = Path(args.predictions)
    validation_manifest = json.loads((predictions.parent / "analysis_manifest.json").read_text(encoding="utf-8"))
    if validation_manifest.get("mode") != "evaluate" or validation_manifest.get("status") != "completed":
        raise ValueError("completed heldout prediction validation required")
    if validation_manifest.get("predictions_sha256") != digest(predictions) or validation_manifest.get("calibration_sha256") != digest(Path(args.calibration)):
        raise ValueError("prediction/calibration provenance mismatch")
    frame = pd.read_csv(predictions)
    reports = [propagate_uncertainty(frame, point, draws, label, out) for label, draws in draw_sets.items()]
    write_json(out / "manifest.json", {"status": "completed", "created_utc": utc_now(), "reports": reports,
        "calibration_sha256": digest(Path(args.calibration)), "predictions_sha256": digest(predictions),
        "source_sha256": {name: digest(Path(__file__).parent / name) for name in ("analyze_prediction_uncertainty.py", "semantic_uncertainty.py")},
        "uncertainty_sources": calibration_manifest["uncertainty_artifact_sha256"],
        "point_estimator_unchanged": True, "heldout_used_for_refitting": False,
        "primary_p_values": "retain separate prespecified family-level Holm-adjusted point analysis; draw fractions are not p values"})


if __name__ == "__main__":
    main()
