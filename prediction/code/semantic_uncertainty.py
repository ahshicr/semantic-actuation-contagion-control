"""Propagate jointly resampled measured kernels; never invent observations."""
import copy
import json
from pathlib import Path

import numpy as np

from evidence_io import digest

FATES = ("prompt", "memory", "map", "tool", "actuation", "tag_absent", "protocol_rejected")
PREDICTORS = ("contextual", "stationary", "identity", "frequency", "role_only")


def load_draw_sets(calibration_path):
    calibration_path = Path(calibration_path)
    root = calibration_path.parent
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("mode") != "calibrate" or manifest.get("calibration_sha256") != digest(calibration_path):
        raise ValueError("completed calibrated analysis and exact point-estimate hash required")
    for name, sha in manifest["uncertainty_artifact_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError("uncertainty artifact changed: " + name)
    result = {}
    with np.load(root / "family_bootstrap_kernels.npz", allow_pickle=False) as archive:
        result["family_bootstrap"] = {name: archive[name].copy() for name in ("forward", "injection", "benign_success", "sampled_family_indices")}
        models, families = archive["model_aliases"].tolist(), archive["families"].tolist()
    if models != calibration["models"] or families != calibration["content_families"]:
        raise ValueError("draw coordinates differ from frozen calibration")
    with np.load(root / "dirichlet_sensitivity_kernels.npz", allow_pickle=False) as archive:
        if archive["model_aliases"].tolist() != models or archive["families"].tolist() != families:
            raise ValueError("prior sensitivity coordinates differ")
        for alpha in ("0p1", "0p5", "1p0"):
            result["dirichlet_alpha_" + alpha] = {name: archive[name + "_alpha_" + alpha].copy()
                                                   for name in ("forward", "injection", "benign_success")}
            result["dirichlet_alpha_" + alpha]["sampled_family_indices"] = archive["sampled_family_indices"].copy()
    for name, draws in result.items():
        count = len(draws["forward"])
        for kind in ("forward", "injection"):
            values = draws[kind]
            if values.shape != (count, len(models), 5, 5, 7) or not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(-1), 1., atol=1e-10):
                raise ValueError("invalid kernel draw: " + name + "/" + kind)
        quality = draws["benign_success"]
        if quality.shape != (count, len(models), 5) or not np.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any():
            raise ValueError("invalid jointly resampled benign success")
        if draws["sampled_family_indices"].shape != (count, len(families)):
            raise ValueError("family-draw denominator differs")
    return calibration, result, manifest


def calibration_draw(point, draws, index, label):
    """Derived parameter setting, explicitly NOT another observed calibration."""
    if not 0 <= index < len(draws["forward"]):
        raise ValueError("uncertainty draw index outside planned evidence")
    value = copy.deepcopy(point)
    indices = [point["models"].index(model) for model in point["receiver_type_model_assignment"]]
    forward, injection = draws["forward"][index], draws["injection"][index]
    value.update(status="derived_uncertainty_setting_not_new_observed_calibration", uncertainty_label=label, uncertainty_draw_index=index,
                 forward_fate_probabilities=forward.tolist(), injection_fate_probabilities=injection.tolist(),
                 receiver_type_role_matrices=forward[indices, ..., :5].tolist(),
                 receiver_type_role_injection_matrices=injection[indices, ..., :5].tolist(),
                 receiver_type_matrices=forward[indices, ..., :5].mean(1).tolist(),
                 receiver_type_role_benign_success=draws["benign_success"][index, indices].tolist())
    return value


def prediction_draws(forward, injection, model, role, source, condition, depth):
    """Same prespecified predictors as point analysis, vectorized over draws."""
    if condition not in ("attack", "seeded_fact") or depth < 1:
        raise ValueError("declared attack/seeded-fact horizon required")
    count = len(forward)
    first_all = injection[:, model] if condition == "attack" else forward[:, model]
    first, following = first_all[:, role], forward[:, model, role]
    stationary = first_all.mean(1)
    frequency = first_all.mean(axis=(1, 2))
    initial = np.zeros((count, 7))
    initial[:, source] = 1.
    state, baseline, role_only = initial.copy(), initial.copy(), initial.copy()
    output = {name: [] for name in PREDICTORS}

    def advance(current, kernel):
        next_state = np.einsum("dc,dce->de", current[:, :5], kernel)
        next_state[:, 5:] += current[:, 5:]
        return next_state

    for step in range(depth):
        kernel = first if step == 0 else following
        state = advance(state, kernel)
        baseline = advance(baseline, stationary)
        rates = kernel.mean(1)
        role_kernel = np.zeros_like(kernel)
        role_kernel[:, :, role] = rates[:, :5].sum(-1, keepdims=True)
        role_kernel[:, :, 5:] = rates[:, None, 5:]
        role_only = advance(role_only, role_kernel)
        for name, value in (("contextual", state), ("stationary", baseline), ("identity", initial),
                            ("frequency", frequency), ("role_only", role_only)):
            output[name].append(value.copy())
    return {name: np.stack(values, axis=1) for name, values in output.items()}
