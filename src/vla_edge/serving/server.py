"""Inference server.

One endpoint, ``/act``, json_numpy encoded, batch 1. Deliberately boring: the
interesting engineering is behind the backend seam, and a serving layer that
changes shape per embodiment is a serving layer that has to be re-verified per
embodiment.
"""

import argparse
import errno
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import json_numpy
import numpy as np

from ..config import EMBODIMENTS, get_embodiment
from ..pipeline import Pipeline

# Patches the stdlib json module so ndarrays round-trip. Must happen before
# anything serializes.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("vla_edge.server")


def _reserve_listener(host: str, port: int, backlog: int = 2048) -> socket.socket:
    """Claim the serving address before loading a checkpoint.

    Keeping this socket open through model loading also prevents two concurrent
    launches from both paying the startup cost before one loses the port race.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    listener = socket.socket(family=family)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(backlog)
    except BaseException:
        listener.close()
        raise
    listener.set_inheritable(True)
    return listener


def _probe_host(host: str) -> str:
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host in {"::", "[::]"}:
        return "::1"
    return host


def _probe_existing_server(
    host: str,
    port: int,
    timeout_sec: float = 0.75,
) -> dict[str, Any] | None:
    """Read metadata from an HTTP inference server already using the port."""
    probe_host = _probe_host(host)
    url_host = f"[{probe_host}]" if ":" in probe_host else probe_host
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
    try:
        with opener.open(
            f"http://{url_host}:{port}/act",
            timeout=timeout_sec,
        ) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError, urlerror.URLError):
        return None
    if isinstance(payload, dict) and payload.get("status") == "ok":
        return payload
    return None


def _server_summary(metadata: dict[str, Any]) -> str:
    keys = (
        "embodiment",
        "repo_id",
        "norm_tag",
        "backend",
        "device",
        "dtype",
        "rtc",
        "rtc_available",
    )
    return ", ".join(
        f"{key}={metadata[key]!r}" for key in keys if key in metadata
    )


def _report_bind_error(host: str, port: int, exc: OSError) -> None:
    log.error("cannot listen on %s:%d: %s; no model was loaded", host, port, exc)
    if exc.errno == errno.EADDRINUSE:
        metadata = _probe_existing_server(host, port)
        if metadata is not None:
            log.error(
                "the existing inference service reports: %s",
                _server_summary(metadata),
            )
        next_port = port + 1 if port < 65535 else port - 1
        log.error(
            "use the existing service, stop its owner, or choose another port "
            "with `--port %d`; find the owner with "
            "`ss -ltnp 'sport = :%d'`",
            next_port,
            port,
        )


_REQUIRED_ENGINE_FILES = (
    "llm_prefill.plan",
    "vision.plan",
    "action_flow.plan",
)


def _declared_engine_repo(engine_dir: Path) -> str | None:
    """Return the checkpoint declared by an engine set's compact host."""
    try:
        serving = json.loads((engine_dir / "serving.json").read_text())
        host_dir = serving.get("host_dir")
        if host_dir is None:
            return None
        host_path = Path(str(host_dir))
        if not host_path.is_absolute():
            host_path = engine_dir / host_path
        host_manifest = json.loads((host_path.resolve() / "host.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    repo_id = host_manifest.get("repo_id")
    return str(repo_id) if repo_id else None


def _resolve_engine_dir(
    engine_dir: str | Path,
    *,
    repo_id: str,
    fast_vision: bool,
) -> Path:
    """Accept either an engine-set directory or a released bundle root."""
    root = Path(engine_dir).expanduser().resolve()
    if (root / "serving.json").is_file() or not (root / "MANIFEST.json").is_file():
        return root

    discovered = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and (child / "serving.json").is_file()
        and all((child / name).is_file() for name in _REQUIRED_ENGINE_FILES)
    ]
    candidates = [
        child
        for child in discovered
        if _declared_engine_repo(child) in (None, repo_id)
    ]
    if fast_vision:
        accelerated = [
            child for child in candidates if (child / "vision_fp8.plan").is_file()
        ]
        if len(accelerated) == 1:
            candidates = accelerated

    if len(candidates) == 1:
        selected = candidates[0]
        log.info("bundle root %s: selected engine set %s", root, selected.name)
        return selected

    if candidates:
        choices = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"{root} contains multiple engine sets for {repo_id}. "
            f"Pass one explicitly with --engine-dir: {choices}"
        )

    found = ", ".join(
        f"{path.name} ({_declared_engine_repo(path) or 'checkpoint unknown'})"
        for path in discovered
    ) or "none"
    raise ValueError(
        f"{root} is a bundle root but contains no engine set for {repo_id}. "
        f"Found: {found}"
    )


def build_app(pipeline: Pipeline, backend_name: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    emb = pipeline.embodiment
    app = FastAPI(title=f"vla-edge · {emb.name}", version="0.1.0")

    def _error(status: int, message: str):
        return Response(
            content=json_numpy.dumps({"error": message}),
            status_code=status,
            media_type="application/json",
        )

    @app.get("/act")
    async def health():
        return JSONResponse(
            {
                "status": "ok",
                "embodiment": emb.name,
                "repo_id": emb.repo_id,
                "norm_tag": emb.norm_tag,
                "backend": backend_name,
                "cameras": list(emb.camera_names),
                "state_dim": emb.state_dim,
                "default_num_steps": emb.default_num_steps,
                "rtc": bool(getattr(pipeline.backend, "rtc_available", False)),
            }
        )

    @app.get("/healthz")
    async def healthz():
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request):
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _error(400, f"failed to decode json_numpy body: {exc}")

        try:
            cameras = {name: payload[name] for name in emb.camera_names}
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as exc:
            return _error(
                400,
                f"missing required field: {exc}. This embodiment expects "
                f"cameras {list(emb.camera_names)}, an 'instruction' string, "
                f"and a ({emb.state_dim},) 'state' array.",
            )

        # Optional Real-Time Chunking guidance: a client mid-episode may send
        # the not-yet-executed rows of the previous chunk so the next chunk
        # stays consistent with what the robot is already committed to. The
        # fields are honored or refused loudly, never silently ignored.
        rtc_telemetry = None
        if payload.get("prefix_actions") is not None:
            arm = getattr(pipeline.backend, "arm_rtc", None)
            if arm is None:
                return _error(
                    400,
                    "this server's backend does not implement RTC guidance; "
                    "drop the prefix_actions field or serve a tensorrt "
                    "engine set whose bundle ships a flow package",
                )
            try:
                rtc_telemetry = arm(
                    payload["prefix_actions"],
                    inference_delay=int(payload.get("inference_delay", 0)),
                    execution_horizon=int(payload.get("execution_horizon", 10)),
                    rtc_schedule=payload.get("rtc_schedule"),
                    rtc_max_guidance=payload.get("rtc_max_guidance"),
                )
            except ValueError as exc:
                return _error(400, str(exc))

        num_steps = payload.get("num_steps")
        t0 = time.perf_counter()
        try:
            actions = pipeline.predict(
                cameras=cameras,
                instruction=instruction,
                state=state,
                num_steps=int(num_steps) if num_steps is not None else None,
                enable_cuda_graph=bool(payload.get("enable_cuda_graph", False)),
            )
        except ValueError as exc:
            return _error(400, str(exc))
        except Exception as exc:
            log.exception("inference failed")
            return _error(500, f"inference failed: {exc}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body: dict = {
            "actions": np.asarray(actions, dtype=np.float32),
            "dt_ms": dt_ms,
        }
        if rtc_telemetry is not None:
            body["rtc"] = rtc_telemetry
        return Response(
            content=json_numpy.dumps(body),
            media_type="application/json",
        )

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embodiment", default="bimanual-yam",
                    choices=sorted(EMBODIMENTS))
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8202)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--engine-dir", default=None,
                    help="tensorrt backend: an engine-set directory or a "
                         "released bundle root containing one matching set")
    ap.add_argument("--pad-multiple", type=int, default=None,
                    help="tensorrt backend: pad prompts to a multiple of "
                         "this length. Default: the engine set's own "
                         "serving.json (the bundle's fixed-bracket sets "
                         "declare 704 there), else 1 for dynamic-shape "
                         "plans. Set explicitly only to override the set's "
                         "declared value")
    ap.add_argument("--fast-vision", action="store_true",
                    help="tensorrt backend: serve the engine set's "
                         "vision_fp8.plan instead of vision.plan when the "
                         "bundle ships one. Faster vision stage; validated "
                         "on the same 400-episode evaluation as the default "
                         "set (see the bundle README for its numbers)")
    args = ap.parse_args(argv)
    emb = get_embodiment(args.embodiment)

    backend_kwargs: dict = {}
    if args.backend == "tensorrt":
        if not args.engine_dir:
            ap.error("--backend tensorrt requires --engine-dir")
        try:
            engine_dir = _resolve_engine_dir(
                args.engine_dir,
                repo_id=emb.repo_id,
                fast_vision=args.fast_vision,
            )
        except ValueError as exc:
            ap.error(str(exc))
        backend_kwargs = {
            "engine_dir": engine_dir,
            "pad_multiple": args.pad_multiple,
            "fast_vision": args.fast_vision,
        }
    elif args.fast_vision:
        ap.error("--fast-vision applies only to --backend tensorrt")

    try:
        listener = _reserve_listener(args.host, args.port)
    except OSError as exc:
        _report_bind_error(args.host, args.port, exc)
        return 1

    pipeline = None
    try:
        bound_port = int(listener.getsockname()[1])
        log.info("reserved inference port %s:%d", args.host, bound_port)

        log.info("loading %s on %s (%s), backend=%s",
                 emb.repo_id, args.device, args.dtype, args.backend)
        pipeline = Pipeline.load(
            emb, backend=args.backend, device=args.device, dtype=args.dtype,
            **backend_kwargs,
        )

        if not args.no_warmup:
            t0 = time.perf_counter()
            pipeline.warmup()
            log.info(
                "warmup complete (%.1f ms)",
                (time.perf_counter() - t0) * 1e3,
            )

        import uvicorn

        app = build_app(pipeline, args.backend)
        config = uvicorn.Config(
            app,
            host=args.host,
            port=bound_port,
            log_level="info",
        )
        log.info("listening on %s:%d", args.host, bound_port)
        uvicorn.Server(config).run(sockets=[listener])
    except KeyboardInterrupt:
        # Python 3.12's asyncio runner can re-raise Ctrl-C after Uvicorn has
        # already completed its graceful application shutdown. Treat that
        # user-requested stop as a successful server exit, while allowing all
        # actual startup and runtime errors to keep their tracebacks.
        log.info("server stopped by user")
    finally:
        try:
            if pipeline is not None:
                close = getattr(pipeline, "close", None)
                if callable(close):
                    close()
        finally:
            listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
