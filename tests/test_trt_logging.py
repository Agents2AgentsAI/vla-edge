"""Offline tests for TensorRT logger lifetime and narrowly scoped notices."""
import hashlib
import logging
from types import SimpleNamespace

import pytest

from vla_edge.backends.abcvla.trt_runtime import (
    TensorRTRuntimeError,
    _deserialize_sha_bound_plan,
)
from vla_edge.backends.tensorrt.logger import (
    _DEVICE_NOTICE,
    deserialization_scope,
    get_logger,
)


def fake_trt():
    messages = []
    runtime_loggers = []
    class Logger:
        WARNING = 2
        ERROR = 1
        def __init__(self, _level):
            pass
        def log(self, severity, message):
            messages.append((severity, message))
    class Runtime:
        def __init__(self, logger):
            runtime_loggers.append(logger)
        def deserialize_cuda_engine(self, payload):
            runtime_loggers[-1].log(Logger.WARNING, _DEVICE_NOTICE)
            return bytes(payload)
    return SimpleNamespace(ILogger=object, Logger=Logger, Runtime=Runtime, messages=messages, runtime_loggers=runtime_loggers)


def test_all_stage_loads_reuse_the_same_live_logger(tmp_path, caplog):
    trt = fake_trt()
    p = tmp_path / "engine.plan"
    p.write_bytes(b"test engine")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            _deserialize_sha_bound_plan(p, expected_sha256=digest, trt=trt, verified_device=True)
    assert len(trt.runtime_loggers) == 5
    assert all(item is get_logger(trt) for item in trt.runtime_loggers)
    assert not trt.messages
    assert "plan and runtime device match" in caplog.text


def test_only_exact_verified_warning_is_downgraded_and_scope_does_not_leak():
    trt = fake_trt()
    logger = get_logger(trt)
    logger.log(2, _DEVICE_NOTICE)
    with deserialization_scope(trt, verified_device=True):
        logger.log(2, _DEVICE_NOTICE)
        logger.log(1, _DEVICE_NOTICE)  # An error must never be downgraded.
        logger.log(2, "unrelated warning")
        logger.log(2, _DEVICE_NOTICE + " Additional mismatch detail")
        with deserialization_scope(trt):
            logger.log(2, _DEVICE_NOTICE)  # Unknown nested plan.
    logger.log(2, _DEVICE_NOTICE)
    assert trt.messages == [
        (2, _DEVICE_NOTICE), (1, _DEVICE_NOTICE), (2, "unrelated warning"),
        (2, _DEVICE_NOTICE + " Additional mismatch detail"),
        (2, _DEVICE_NOTICE), (2, _DEVICE_NOTICE),
    ]


def test_scope_is_reset_after_failed_deserialization():
    trt = fake_trt()
    with pytest.raises(RuntimeError), deserialization_scope(trt, verified_device=True):
        raise RuntimeError("failed load")
    get_logger(trt).log(2, _DEVICE_NOTICE)
    assert trt.messages == [(2, _DEVICE_NOTICE)]


def test_engine_hash_still_checked_before_any_runtime_is_created(tmp_path):
    trt = fake_trt()
    p = tmp_path / "engine.plan"
    p.write_bytes(b"changed")
    with pytest.raises(TensorRTRuntimeError, match="SHA-256 mismatch"):
        _deserialize_sha_bound_plan(p, expected_sha256="0" * 64, trt=trt, verified_device=True)
    assert not trt.runtime_loggers
