"""ABC/OpenPI-compatible NumPy msgpack codec; numeric arrays only.

Wire format adapted from Physical Intelligence openpi (Apache-2.0) and
amazon-far/abc (Apache-2.0). No pickle fallback is supported.
"""

import msgpack
import numpy as np


def _pack(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in "VOc":
            raise ValueError(f"unsupported dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"unsupported msgpack value {type(value)}")


def _unpack(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise ValueError(f"unsupported dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def packb(value):
    return msgpack.packb(value, default=_pack)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=_unpack)
