"""Run a task using the policy advertised by the configured inference server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vla_edge.protocol.client import ActClient

HERE = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", nargs="?")
    parser.add_argument("--prompt", help="alias for the positional instruction")
    parser.add_argument("--server", default=os.getenv("YAM_SERVER", "127.0.0.1:8202"))
    parser.add_argument(
        "--policy",
        choices=("abcvla-bimanual-yam", "molmoact2-bimanual-yam", "pi05-bimanual-yam"),
        help="require this policy; otherwise use the server's identity",
    )
    parser.add_argument("--rtc-prefix-length", type=int, default=None)
    parser.add_argument("--execute-chunk-dim", type=int, default=None)
    parser.add_argument(
        "--check", action="store_true", help="preflight only; never enable motors"
    )
    parser.add_argument(
        "--execute", action="store_true", help="explicit execution (also the default)"
    )
    args, extra = parser.parse_known_args(argv)
    if args.instruction and args.prompt:
        parser.error("use either a positional instruction or --prompt")
    args.instruction = args.instruction or args.prompt
    if not args.instruction:
        parser.error("an instruction is required")
    if args.check and args.execute:
        parser.error("choose --check or --execute")
    return parser, args, extra


def main(argv=None):
    parser, args, extra = arguments(argv)
    os.chdir(HERE)
    client = ActClient(args.server, timeout_s=5)
    try:
        contract = client.bind(
            policy=args.policy,
            embodiment="bimanual-yam",
            cameras=("top_cam", "left_cam", "right_cam"),
            state_dim=14,
            action_dim=14,
        )
        metadata = client.health()
    finally:
        client.close()
    if contract.model_family == "abcvla":
        if extra:
            parser.error("unsupported ABC options: " + " ".join(extra))
        from abc_rollout import preflight, run

        args.rtc_prefix_length = (
            args.rtc_prefix_length
            if args.rtc_prefix_length is not None
            else int(os.getenv("ABC_RTC_PREFIX_LENGTH", "5"))
        )
        args.execute_chunk_dim = (
            args.execute_chunk_dim
            if args.execute_chunk_dim is not None
            else int(os.getenv("ABC_EXECUTE_CHUNK_DIM", "6"))
        )
        rig = preflight(HERE, metadata, args.rtc_prefix_length, args.execute_chunk_dim)
        print(
            f"ABC-VLA: prefix={args.rtc_prefix_length}, execute={args.execute_chunk_dim}, server={args.server}"
        )
        if args.check:
            print(
                "Server, rig configuration, camera feed and CAN links verified; no motors enabled."
            )
            return 0
        return run(args, rig, HERE)
    if contract.model_family == "pi05" and contract.policy == "pi05-bimanual-yam":
        if args.rtc_prefix_length is not None or args.execute_chunk_dim is not None:
            parser.error("Pi0.5 uses YAM_ACTION_HORIZON, not ABC hard-prefix options")
        if os.getenv("YAM_RTC", "0") != "0":
            parser.error("this Pi0.5 release does not support RTC; use YAM_RTC=0")
        horizon = int(os.getenv("YAM_ACTION_HORIZON", "16"))
        if contract.action_horizon != 16 or not 1 <= horizon <= 16 or contract.gripper_state != "commanded":
            parser.error("Pi0.5 requires horizon 16, executed horizon 1..16 and commanded gripper state")
        if extra:
            parser.error("unsupported Pi0.5 options: " + " ".join(extra))
        from pi05_rollout import control_settings, preflight, run

        control_settings()
        configs = preflight(HERE)
        if args.check:
            print("Pi0.5 server, rig, cameras and bounded-motion settings verified; no motors enabled.")
            return 0
        return run(args, configs, HERE)
    if (
        contract.model_family != "molmoact2"
        or contract.policy != "molmoact2-bimanual-yam"
    ):
        parser.error(f"unsupported robot policy: {contract.policy}")
    if args.rtc_prefix_length is not None or args.execute_chunk_dim is not None:
        parser.error(
            "hard-prefix options apply to ABC-VLA; MolmoAct uses YAM_RTC settings"
        )
    if args.check:
        print("MolmoAct2 server contract verified; no robot process started.")
        return 0
    env = dict(os.environ, YAM_SERVER=args.server, YAM_POLICY=contract.policy)
    os.execvpe(
        "bash",
        ["bash", str(HERE / "run_molmoact_task.sh"), args.instruction, *extra],
        env,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"[run_task] {exc}") from None
