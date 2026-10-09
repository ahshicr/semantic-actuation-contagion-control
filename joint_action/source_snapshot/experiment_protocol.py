"""Prospective, content-addressed protocols for non-generative experiments.

This is an internal freeze, not external preregistration. The formal runner
must validate the complete configuration, calibration, runtime, and sources.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evidence_io import digest, utc_now, write_json

HERE = Path(__file__).resolve().parent
SOURCE_FILES = {
    "certificates": ("run_certificate_study.py", "run_simulation.py", "sim_core.py",
                     "certificates.py", "optimization.py", "experiment_protocol.py", "simulation_evidence.py", "evidence_io.py"),
    "simulation": ("run_simulation.py", "sim_core.py", "certificates.py",
                   "optimization.py", "experiment_protocol.py", "simulation_evidence.py", "evidence_io.py"),
    "exact_reference": ("run_exact_reference_study.py", "exact_reference.py", "run_simulation.py", "sim_core.py",
                        "certificates.py", "optimization.py", "experiment_protocol.py", "simulation_evidence.py", "evidence_io.py"),
    "control_uncertainty": ("run_control_uncertainty.py", "semantic_uncertainty.py", "run_simulation.py", "sim_core.py",
                            "certificates.py", "optimization.py", "experiment_protocol.py", "simulation_evidence.py", "evidence_io.py"),
}


def source_hashes(kind):
    return {name: digest(HERE / name) for name in SOURCE_FILES[kind]}


def calibration_document(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("status") != "calibrated_from_observed_api_state_changes":
        raise ValueError("formal experiments require observed operational calibration")
    if "receiver_type_role_matrices" not in document:
        raise ValueError("receiving-role calibration is required")
    models = document["models"]
    if len(models) != len(set(models)) or not document["content_families"]:
        raise ValueError("unique model identities and calibration families required")
    indices = [models.index(name) for name in document["receiver_type_model_assignment"]]
    if len(indices) != 4:
        raise ValueError("all four receiving types need a measured model assignment")
    for prefix, destination in (("forward", "receiver_type_role_matrices"), ("injection", "receiver_type_role_injection_matrices")):
        counts = np.asarray(document[prefix + "_counts"], dtype=float)
        denominator = np.asarray(document[prefix + "_denominators"], dtype=float)
        probabilities = np.asarray(document[prefix + "_fate_probabilities"], dtype=float)
        if counts.shape != (len(models), 5, 5, 7) or denominator.shape != counts.shape[:-1]:
            raise ValueError("incorrect calibration count dimensions")
        if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
            raise ValueError("calibration requires nonnegative integer observed counts")
        if not np.equal(counts.sum(-1), denominator).all() or (denominator <= 0).any():
            raise ValueError("all calibration attempts must be counted in a positive denominator")
        expected = counts / denominator[..., None]
        if probabilities.shape != expected.shape or not np.allclose(probabilities, expected, atol=1e-12, rtol=0):
            raise ValueError("calibration probabilities do not equal unsmoothed observed frequencies")
        receiver = np.asarray(document[destination], dtype=float)
        if receiver.shape != (4, 5, 5, 5) or not np.allclose(receiver, expected[indices, ..., :5], atol=1e-12, rtol=0):
            raise ValueError("receiver-specific matrix is not the measured model/role matrix")
    pooled = np.asarray(document["receiver_type_matrices"], dtype=float)
    if pooled.shape != (4, 5, 5) or not np.allclose(pooled, np.asarray(document["receiver_type_role_matrices"]).mean(1), atol=1e-12, rtol=0):
        raise ValueError("pooled receiver matrices do not match measured role averages")
    quality = np.asarray(document["receiver_type_role_benign_success"], dtype=float)
    if quality.shape != (4, 5) or not np.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any():
        raise ValueError("invalid measured benign task-success probabilities")
    benign_counts = np.asarray(document["benign_success_counts"], dtype=float)
    benign_attempts = np.asarray(document["benign_attempt_counts"], dtype=float)
    if benign_counts.shape != (len(models), 5) or benign_attempts.shape != benign_counts.shape:
        raise ValueError("invalid benign task count dimensions")
    if not np.isfinite(benign_counts).all() or not np.isfinite(benign_attempts).all() or (benign_counts < 0).any() or (benign_attempts <= 0).any():
        raise ValueError("positive observed benign attempt denominators required")
    if not np.equal(benign_counts, np.floor(benign_counts)).all() or not np.equal(benign_attempts, np.floor(benign_attempts)).all() or (benign_counts > benign_attempts).any():
        raise ValueError("benign successes must be integer observed counts within the denominator")
    if not np.allclose(quality, (benign_counts / benign_attempts)[indices], atol=1e-12, rtol=0):
        raise ValueError("benign task-success table differs from observed counts")
    return document


def calibration_analysis_path(path):
    parent = Path(path).parent
    candidates = [parent / "analysis_manifest.json", parent / "calibration_analysis_manifest.json"]
    found = [item for item in candidates if item.exists()]
    if len(found) != 1:
        raise ValueError("one unambiguous completed calibration analysis manifest is required")
    analysis = json.loads(found[0].read_text(encoding="utf-8"))
    if analysis.get("status") != "completed" or analysis.get("mode") != "calibrate" or not analysis.get("provenance"):
        raise ValueError("calibration analysis is incomplete or missing raw-run provenance")
    if analysis.get("calibration_sha256") != digest(Path(path)):
        raise ValueError("calibration differs from completed analysis")
    return found[0]


def create_protocol(kind, configs, calibration, runtime):
    calibration_document(calibration)
    cases = []
    for path in configs:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if config.get("phase") != "formal":
            raise ValueError("a formal protocol cannot whitelist development configurations")
        cases.append({"setting_id": config["setting_id"], "config": config,
                      "config_sha256": digest(config), "file_sha256": digest(Path(path))})
    if len({case["setting_id"] for case in cases}) != len(cases):
        raise ValueError("setting IDs must be unique")
    return {"status": "frozen_after_development", "schema": 1, "kind": kind,
            "created_utc": utc_now(), "calibration_sha256": digest(Path(calibration)),
            "calibration_analysis_sha256": digest(calibration_analysis_path(calibration)),
            "sources": source_hashes(kind), "runtime": runtime, "configurations": cases}


def validate_protocol(protocol, kind, config, calibration, runtime):
    if protocol.get("status") != "frozen_after_development" or protocol.get("kind") != kind:
        raise ValueError("wrong or unfrozen experiment protocol")
    calibration_document(calibration)
    if protocol["calibration_sha256"] != digest(Path(calibration)):
        raise ValueError("calibration differs from frozen protocol")
    if protocol["calibration_analysis_sha256"] != digest(calibration_analysis_path(calibration)):
        raise ValueError("calibration analysis provenance differs from frozen protocol")
    if protocol["sources"] != source_hashes(kind):
        raise ValueError("experiment source differs from frozen protocol")
    if protocol["runtime"] != runtime:
        raise ValueError("runtime differs from frozen protocol")
    matches = [case for case in protocol["configurations"] if case["setting_id"] == config.get("setting_id")]
    if len(matches) != 1 or digest(config) != matches[0]["config_sha256"] or config != matches[0]["config"]:
        raise ValueError("entire configuration must match a frozen whitelisted setting")
    if config.get("phase") != "formal":
        raise ValueError("formal phase required")


def archive_inputs(out, kind, config, protocol=None, calibration=None):
    out = Path(out)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCE_FILES[kind]:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    write_json(out / "configuration.json", config)
    if protocol is not None:
        # Preserve exact original bytes, not a differently serialized document.
        (out / "frozen_protocol.json").write_bytes(Path(protocol).read_bytes())
    if calibration is not None:
        (out / "semantic_calibration.json").write_bytes(Path(calibration).read_bytes())
        analysis = calibration_analysis_path(calibration)
        (out / "calibration_analysis_manifest.json").write_bytes(analysis.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(SOURCE_FILES), required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite a frozen protocol")
    document = create_protocol(args.kind, args.configs, args.calibration,
                               {"device": args.device, "batch_size": args.batch_size})
    write_json(out, document)
    print(json.dumps({"protocol": str(out), "sha256": digest(out)}))


if __name__ == "__main__":
    main()
