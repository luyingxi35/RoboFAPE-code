"""Bounded, local policy RPC independent of training frameworks."""
from __future__ import annotations
import base64
import importlib
import json
from pathlib import Path
import socket
import struct
import uuid
import numpy as np

MAX_MESSAGE = 32 * 1024 * 1024


def encode_array(value):
    value = np.ascontiguousarray(value)
    return {"dtype": value.dtype.str, "shape": list(value.shape), "data": base64.b64encode(value.tobytes()).decode()}


def decode_array(value):
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if dtype.kind not in "buif" or not isinstance(shape, list) or not 1 <= len(shape) <= 4:
        raise ValueError("unsupported array dtype or rank")
    if any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("invalid shape")
    expected = int(np.prod(shape, dtype=object)) * dtype.itemsize
    if expected > MAX_MESSAGE:
        raise ValueError("array too large")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != expected:
        raise ValueError("array byte count differs from shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def recv_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("policy connection closed during message")
        data.extend(chunk)
    return bytes(data)


def receive(connection):
    size = struct.unpack("!I", recv_exact(connection, 4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("invalid message size")
    return json.loads(recv_exact(connection, size))


def send(connection, value):
    raw = json.dumps(value, allow_nan=False).encode()
    if len(raw) > MAX_MESSAGE:
        raise ValueError("message too large")
    connection.sendall(struct.pack("!I", len(raw)) + raw)


def valid_actions(actions, dimension):
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or not 1 <= len(a) <= 1024 or a.shape[1] != dimension or not np.isfinite(a).all():
        raise ValueError(f"policy must return finite [horizon,{dimension}] environment actions")
    return a


class PolicyClient:
    def __init__(self, path, timeout=120):
        self.path, self.timeout = str(path), timeout

    def request(self, message):
        message = {"protocol": 1, "request_id": uuid.uuid4().hex, **message}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.path)
            send(connection, message)
            response = receive(connection)
        if response.get("request_id") != message["request_id"]:
            raise ValueError("policy response request_id mismatch")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "policy request failed"))
        return response

    def health(self):
        return self.request({"command": "health"})

    def infer(self, main_image, wrist_image, state, episode_id, step_index, task_description):
        response = self.request({"command": "infer", "main_image": encode_array(main_image),
                                 "wrist_image": encode_array(wrist_image), "state": encode_array(state),
                                 "episode_id": episode_id, "step_index": step_index,
                                 "task_description": task_description})
        return decode_array(response["actions"])


def load_factory(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def serve(path, checkpoint, config, factory):
    from .artifacts import digest, resource
    checkpoint, path = Path(checkpoint).resolve(), Path(path)
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise FileNotFoundError("provide a populated compatible checkpoint directory")
    if path.exists():
        raise FileExistsError("socket already exists; use another path")
    if len(str(path).encode()) > 100:
        raise ValueError("use a short Unix socket path")
    required = ("control_mode", "action_dim", "state_dim", "task_ids", "checkpoint_files", "execute_chunk")
    if any(k not in config for k in required):
        raise ValueError(f"policy config requires {required}")
    if not config["checkpoint_files"]:
        raise ValueError("checkpoint_files must list weights and normalization files")
    for field in ("action_dim", "state_dim", "execute_chunk"):
        if type(config[field]) is not int or not 1 <= config[field] <= 1024:
            raise ValueError(f"{field} must be a positive bounded integer")
    if not isinstance(config["task_ids"],list) or not config["task_ids"]:
        raise ValueError("task_ids must list supported task identifiers")
    hashes = {}
    for relative in config["checkpoint_files"]:
        target = resource(checkpoint, relative)
        hashes[relative] = digest(target)
    model = load_factory(factory)(checkpoint, config)
    metadata = {k: config[k] for k in required if k != "checkpoint_files"}
    metadata["checkpoint_sha256"] = hashes
    metadata["factory"] = factory
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        path.chmod(0o600)
        server.listen(4)
        try:
            while True:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(120)
                    request = {}
                    try:
                        request = receive(connection)
                        if request.get("protocol") != 1:
                            raise ValueError("unsupported protocol")
                        if request["command"] == "health":
                            result = metadata
                        elif request["command"] == "infer":
                            obs = {k: decode_array(request[k]) for k in ("main_image", "wrist_image", "state")}
                            for name in ("main_image", "wrist_image"):
                                if obs[name].dtype != np.uint8 or obs[name].ndim != 3 or obs[name].shape[-1] != 3:
                                    raise ValueError("policy images must be HWC RGB uint8")
                            if obs["state"].shape != (config["state_dim"],) or not np.isfinite(obs["state"]).all():
                                raise ValueError("policy state does not match checkpoint contract")
                            obs.update({k: request[k] for k in ("episode_id", "step_index", "task_description")})
                            actions = valid_actions(model.infer(obs), config["action_dim"])
                            result = {"actions": encode_array(actions)}
                        else:
                            raise ValueError("unknown command")
                        send(connection, {"ok": True, "request_id": request.get("request_id"), **result})
                    except (ValueError, KeyError, RuntimeError, ConnectionError, OSError) as error:
                        try:
                            send(connection, {"ok": False, "request_id": request.get("request_id"), "error": str(error)})
                        except OSError:
                            pass
        finally:
            path.unlink(missing_ok=True)


def openpi_factory(checkpoint, config):
    """Use an OpenPI-exported checkpoint with its registered data transforms.

    The selected OpenPI configuration must output environment-space actions;
    this adapter never guesses normalization or controller conversions.
    """
    from .openpi_adapter import create_policy
    trained = create_policy(checkpoint, config)
    class OpenPIAdapter:
        def infer(self, obs):
            data = {"observation/image": obs["main_image"], "observation/wrist_image": obs["wrist_image"],
                    "observation/state": obs["state"], "prompt": obs["task_description"]}
            return trained.infer(data)["actions"]
    return OpenPIAdapter()
