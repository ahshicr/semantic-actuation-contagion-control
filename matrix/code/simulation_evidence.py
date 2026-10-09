"""Immutable simulation attempts, resumable units, and derived aggregate views.

Only fully verified successful units are skipped on resume. Failed/partial
attempts remain in their original directories and in the complete ledger.
The top-level JSONL files are derived views, never the only raw evidence.
"""
import contextlib
import json
import os
from pathlib import Path
import time
import uuid

from evidence_io import digest, utc_now, write_json


@contextlib.contextmanager
def run_lease(out):
    """Kernel-owned nonblocking lock; a leftover file is not a live process."""
    path = Path(out) / ".execution.lock"
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def planned_units(config, batch_size):
    if batch_size < 1 or config["trials"] < 1:
        raise ValueError("positive trial count and batch size required")
    if len(config["policies"]) != len(set(config["policies"])):
        raise ValueError("duplicate policies would repeat the experimental unit")
    return [{"unit_id": f"batch{offset:06d}_{policy}", "policy": policy,
             "trial_ids": list(range(offset, min(offset + batch_size, config["trials"])))}
            for offset in range(0, config["trials"], batch_size) for policy in config["policies"]]


def validate_resume(out, manifest, identity):
    if manifest.get("evidence_schema") != 2:
        raise ValueError("legacy runs cannot be silently converted into resumable evidence")
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise ValueError("resume identity mismatch: " + key)
    for name, sha in identity["code_hashes"].items():
        if digest(Path(out) / "source_snapshot" / name) != sha:
            raise ValueError("archived source changed: " + name)
    for name, field in (("frozen_protocol.json", "protocol_sha256"),
                        ("semantic_calibration.json", "calibration_sha256")):
        if identity[field] is not None and digest(Path(out) / name) != identity[field]:
            raise ValueError("archived input changed: " + name)
    configuration = json.loads((Path(out) / "configuration.json").read_text(encoding="utf-8"))
    if configuration != identity["config"]:
        raise ValueError("archived complete configuration changed")


def attempts_for(out, unit, config):
    folder = Path(out) / "units" / unit["unit_id"]
    history = []
    if not folder.exists():
        return history
    for attempt in sorted(folder.glob("attempt_*")):
        if not attempt.is_dir():
            raise ValueError("attempt path is not a directory")
        manifest_path = attempt / "manifest.json"
        if not manifest_path.exists():
            record = {**unit, "attempt_id": attempt.name, "status": "interrupted_without_manifest"}
        else:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(record.get(key) != unit[key] for key in unit) or record.get("attempt_id") != attempt.name:
                raise ValueError("attempt identity differs from planned unit")
            if record.get("status") in ("completed", "failed"):
                for name, key in (("trials.jsonl", "trials_sha256"), ("controller_calls.jsonl", "calls_sha256")):
                    if digest(attempt / name) != record[key]:
                        raise ValueError("closed attempt evidence changed: " + str(attempt / name))
            if record.get("status") == "completed":
                rows = [json.loads(line) for line in (attempt / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
                if len(rows) != len(unit["trial_ids"]) or [row["trial_id"] for row in rows] != unit["trial_ids"]:
                    raise ValueError("completed trial denominator differs from the plan")
                if any(row["policy"] != unit["policy"] or row["setting_id"] != config["setting_id"] or row["status"] != "completed" for row in rows):
                    raise ValueError("incorrect identity/status in completed trial rows")
                calls = [json.loads(line) for line in (attempt / "controller_calls.jsonl").read_text(encoding="utf-8").splitlines()]
                interval = config.get("params", {}).get("control_interval", 4)
                if [row["step"] for row in calls] != list(range(0, config["steps"], interval)):
                    raise ValueError("completed controller-call denominator differs from the plan")
                if any(row["policy"] != unit["policy"] or row["trial_ids"] != unit["trial_ids"] for row in calls):
                    raise ValueError("incorrect batch/policy in completed controller calls")
            elif record.get("status") not in ("running", "failed"):
                raise ValueError("unknown attempt status")
            record["attempt_manifest_sha256"] = digest(manifest_path)
        record = dict(record, relative_directory=str(attempt.relative_to(out)).replace("\\", "/"))
        history.append(record)
    if sum(item["status"] == "completed" for item in history) > 1:
        raise ValueError("multiple successful attempts would allow outcome-dependent selection")
    return history


def completed_attempt(history):
    return next((row for row in history if row["status"] == "completed"), None)


def start_attempt(out, unit, history):
    folder = Path(out) / "units" / unit["unit_id"]
    folder.mkdir(parents=True, exist_ok=True)
    numbers = [int(row["attempt_id"].split("_")[-1]) for row in history]
    attempt_id = f"attempt_{max(numbers, default=0) + 1:04d}"
    attempt = folder / attempt_id
    attempt.mkdir()
    manifest = {**unit, "attempt_id": attempt_id, "status": "running", "started_utc": utc_now(),
                "previous_attempts": [{"attempt_id": row["attempt_id"], "status": row["status"]} for row in history],
                "retry_rule": "retry_entire_unfinished_unit_with_identical_inputs_never_select_by_outcome"}
    write_json(attempt / "manifest.json", manifest)
    return attempt, manifest


def _replace(source, destination):
    for attempt in range(24):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 23:
                raise
            time.sleep(min(.03 * (attempt + 1), .25))


def assemble_views(out, units, config):
    """Regenerate derived top-level views; preserve every raw attempt untouched."""
    out = Path(out)
    history, selected = [], []
    completed_rows = failed_rows = interrupted_units = 0
    for unit in units:
        attempts = attempts_for(out, unit, config)
        history.extend(attempts)
        success = completed_attempt(attempts)
        if success is not None:
            selected.append(success)
            completed_rows += len(unit["trial_ids"])
        elif attempts:
            selected.append(attempts[-1])
        failed_rows += sum(len(unit["trial_ids"]) for row in attempts if row["status"] == "failed")
        interrupted_units += sum(row["status"] in ("running", "interrupted_without_manifest") for row in attempts)
    for filename in ("trials.jsonl", "controller_calls.jsonl"):
        temporary = out / (filename + "." + uuid.uuid4().hex + ".pending")
        with temporary.open("w", encoding="utf-8") as target:
            for attempt in selected:
                source = out / attempt["relative_directory"] / filename
                if not source.exists():
                    continue
                with source.open(encoding="utf-8") as stream:
                    for line in stream:
                        # A killed process may end on an incomplete last line.
                        # Never promote any partial line to a valid observation.
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            if attempt["status"] == "completed":
                                raise
                            continue
                        record.update(unit_id=attempt["unit_id"], attempt_id=attempt["attempt_id"])
                        target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        _replace(temporary, out / filename)
    write_json(out / "attempt_history.json", {
        "attempts": history, "selected_view_attempts": [{"unit_id": row["unit_id"], "attempt_id": row["attempt_id"]} for row in selected],
        "completed_trial_rows": completed_rows, "failed_trial_attempts": failed_rows,
        "interrupted_unit_attempts": interrupted_units,
        "selection_rule": "unique_verified_completed_attempt_else_latest_incomplete_attempt",
        "raw_attempts_preserved": True})
    return {"completed_trial_rows": completed_rows, "failed_trial_attempts": failed_rows,
            "interrupted_unit_attempts": interrupted_units,
            "trials_sha256": digest(out / "trials.jsonl"), "calls_sha256": digest(out / "controller_calls.jsonl"),
            "attempt_history_sha256": digest(out / "attempt_history.json")}
