"""Experiment-only Inductor config dump for R3.0 Step2."""

from __future__ import annotations

import json
import os
import pathlib
import sys


def _log(message: str) -> None:
    print(f"[xdb-inductor] {message}", file=sys.stderr, flush=True)


if os.environ.get("XDB_INDUCTOR_DUMP_CONFIG") == "1":
    try:
        import torch._inductor.config as cfg

        payload = {
            "pid": os.getpid(),
            "TORCHINDUCTOR_MAX_AUTOTUNE": os.environ.get(
                "TORCHINDUCTOR_MAX_AUTOTUNE"
            ),
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": os.environ.get(
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"
            ),
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": os.environ.get(
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"
            ),
            "max_autotune": cfg.max_autotune,
            "max_autotune_gemm": cfg.max_autotune_gemm,
            "max_autotune_gemm_backends": cfg.max_autotune_gemm_backends,
            "search_autotune_cache": cfg.search_autotune_cache,
        }
        path_text = os.environ.get("XDB_INDUCTOR_CONFIG_JSON")
        if path_text:
            base = pathlib.Path(path_text)
            base.parent.mkdir(parents=True, exist_ok=True)
            path = base.with_name(
                f"{base.stem}.pid{os.getpid()}{base.suffix or '.json'}"
            )
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        _log(json.dumps(payload, sort_keys=True))
    except Exception as exc:  # pragma: no cover - diagnostic hook
        _log(f"dump_failed {type(exc).__name__}: {exc}")
