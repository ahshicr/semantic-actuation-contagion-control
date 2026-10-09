"""Portable evidence writing, canonical hashes, and exclusive local GPU use."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(value):
    if isinstance(value, Path):
        hasher = hashlib.sha256()
        with value.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex[:8]}.pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # SMB/Windows readers and virus scanners occasionally open a destination
    # without delete-sharing. Preserve the complete temporary file and retry
    # the atomic commit, never truncate the existing authoritative manifest.
    for attempt in range(24):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 23:
                raise
            time.sleep(min(0.03 * (attempt + 1), 0.25))


def append_jsonl(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


@contextlib.contextmanager
def exclusive_gpu(device="cuda:0"):
    if not str(device).startswith("cuda"):
        yield
        return
    # Byte-range OS lock; a stale metadata file alone is never interpreted as a
    # running process. The kernel releases the lock when its process exits.
    lock = Path(tempfile.gettempdir()) / "tnse_cagmc_v3_gpu0.lock"
    with lock.open("a+b") as stream:
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
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
