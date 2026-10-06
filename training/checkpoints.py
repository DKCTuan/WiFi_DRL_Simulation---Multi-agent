"""Atomic, resumable episode-boundary checkpoints for the QMIX training loop."""

import hashlib
import json
import os
import random
from pathlib import Path
import tempfile

import numpy as np
import torch

import config


CHECKPOINT_VERSION = 1


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint config value: {type(value).__name__}")


def _weights_only_tree(value):
    """Remove NumPy scalar globals that PyTorch's restricted loader rejects."""
    if isinstance(value, dict):
        return {key: _weights_only_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_weights_only_tree(item) for item in value)
    if isinstance(value, list):
        return [_weights_only_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    return value


def training_signature(*, seed, use_water_filling, action_size, num_stas,
                       use_double_dqn, use_qmix, ib_beta, cooperation_enabled,
                       eval_mode):
    settings = {
        key: _jsonable(value)
        for key, value in vars(config).items()
        if key.isupper() and key != "TRAIN_EPISODES"
    }
    settings.update({
        "seed": int(seed),
        "use_water_filling": bool(use_water_filling),
        "action_size": int(action_size),
        "num_stas": None if num_stas is None else int(num_stas),
        "use_double_dqn": bool(use_double_dqn),
        "use_qmix": bool(use_qmix),
        "ib_beta": float(ib_beta),
        "cooperation_enabled": bool(cooperation_enabled),
        "eval_mode": str(eval_mode),
        "source_sha256": project_source_sha256(),
    })
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"settings": settings, "sha256": hashlib.sha256(encoded).hexdigest()}


def project_source_sha256():
    root = Path(__file__).resolve().parents[1]
    ignored = {".git", ".venv", "venv", "results", "results_data", "__pycache__"}
    digest = hashlib.sha256()
    sources = []
    for current, directories, files in os.walk(root):
        directories[:] = [
            name for name in directories
            if name not in ignored and not name.startswith(".audit_artifacts")
        ]
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix not in {".py", ".ipynb"} or name == "create_zip.py":
                continue
            sources.append(path)
    # Stable across Windows/Linux and archive extraction directory order.
    for path in sorted(sources, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].astype(np.uint32).tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(tuple(state["python"]))
    np_state = state["numpy"]
    np.random.set_state((
        np_state[0], np.asarray(np_state[1], dtype=np.uint32),
        int(np_state[2]), int(np_state[3]), float(np_state[4]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_checkpoint(path, payload):
    """Write in the destination directory then atomically replace the old file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(handle)
    try:
        torch.save(_weights_only_tree(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path, data):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, expected_signature):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("Unsupported training checkpoint version")
    actual = payload.get("signature", {})
    if actual.get("sha256") != expected_signature.get("sha256"):
        old = actual.get("settings", {})
        new = expected_signature.get("settings", {})
        changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        raise ValueError(
            "Checkpoint configuration does not match this run. "
            f"Changed settings: {', '.join(changed[:12]) or 'unknown'}"
        )
    return payload
