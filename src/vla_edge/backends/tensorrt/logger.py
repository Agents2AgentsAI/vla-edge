"""One process-lifetime TensorRT logger, shared by every backend.

The device notice is downgraded only inside a verified deserialization scope.
Unknown devices and every other TensorRT warning/error keep their severity.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

_DEVICE_NOTICE = (
    "Using an engine plan file across different models of devices is not supported "
    "and is likely to affect performance or even cause errors or deadlock."
)
_LOCK = threading.RLock()
_STATE = threading.local()
_LOGGERS: dict[int, tuple[Any, Any]] = {}


def get_logger(trt: Any) -> Any:
    # Keep the module and logger alive until process exit, including all engines
    # constructed from a runtime. Tests may supply a separate mock TRT module.
    with _LOCK:
        key = id(trt)
        if key not in _LOGGERS:
            class SharedLogger(trt.ILogger):
                def __init__(self):
                    super().__init__()
                    self.delegate = trt.Logger(trt.Logger.WARNING)

                def log(self, severity, message):
                    if (
                        severity == trt.Logger.WARNING
                        and message.strip() == _DEVICE_NOTICE
                        and getattr(_STATE, "verified_device", False)
                    ):
                        logging.getLogger(__name__).info(
                            "TensorRT device notice verified against build provenance: "
                            "plan and runtime device match"
                        )
                        return
                    self.delegate.log(severity, message)

            _LOGGERS[key] = (trt, SharedLogger())
        return _LOGGERS[key][1]


@contextmanager
def deserialization_scope(trt: Any, *, verified_device: bool = False):
    # Serialize scoped loads. A callback on another thread keeps the warning;
    # it cannot inherit verification from an unrelated engine load.
    with _LOCK:
        previous = getattr(_STATE, "verified_device", False)
        _STATE.verified_device = verified_device
        try:
            yield get_logger(trt)
        finally:
            _STATE.verified_device = previous
