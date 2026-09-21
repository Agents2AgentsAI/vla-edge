"""Check native ABC RTC settings against a running vla-edge server; no motion."""
from __future__ import annotations

import argparse
import json

from vla_edge.protocol.client import ActClient


def validate_abc_rtc(metadata: dict, prefix: int, chunk: int) -> None:
    expected = {
        "policy": "abcvla-bimanual-yam", "model_family": "abcvla",
        "action_horizon": 30, "action_dim": 14, "state_dim": 14,
        "rtc_mode": "hard-prefix",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"ABC server {key}: expected {value!r}, got {metadata.get(key)!r}")
    if not 1 <= prefix <= min(7, int(metadata.get("max_prefix_length", 0))):
        raise ValueError("ABC_RTC_PREFIX_LENGTH must be a supported value between 1 and 7")
    if not prefix < chunk <= 30 - prefix:
        raise ValueError("ABC_EXECUTE_CHUNK_DIM must exceed the prefix and fit the remaining 30-row output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    args = parser.parse_args()
    client = ActClient(f"127.0.0.1:{args.port}", timeout_s=5)
    try:
        metadata = client.health()
        validate_abc_rtc(metadata, args.prefix, args.chunk)
    finally:
        client.close()
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
