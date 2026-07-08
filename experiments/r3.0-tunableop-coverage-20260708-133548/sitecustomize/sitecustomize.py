"""Experiment-only TunableOp gate for R3.0 coverage probing.

Loaded via PYTHONPATH by guard_bench. It must be invisible when
XDB_TUNABLE_ENABLE is unset or set to anything other than "1".
"""

from __future__ import annotations

import os
import sys
import atexit
import json
import pathlib
import threading


def _log(message: str) -> None:
    print(f"[xdb-tunable] {message}", file=sys.stderr, flush=True)


if os.environ.get("XDB_TUNABLE_ENABLE") == "1":
    try:
        import torch

        tunable = torch.cuda.tunable
        dump_base = os.environ.get("XDB_TUNABLE_RESULTS_JSON")

        def _dump_results() -> None:
            if not dump_base:
                return
            try:
                base = pathlib.Path(dump_base)
                base.parent.mkdir(parents=True, exist_ok=True)
                path = base.with_name(
                    f"{base.stem}.pid{os.getpid()}{base.suffix or '.json'}"
                )
                results = tunable.get_results()
                payload = {
                    "pid": os.getpid(),
                    "results_len": len(results) if hasattr(results, "__len__") else None,
                    "results": [list(row) for row in results],
                }
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                _log(f"dump_failed {type(exc).__name__}: {exc}")

        def _start_periodic_dump() -> None:
            interval = float(os.environ.get("XDB_TUNABLE_DUMP_INTERVAL", "0") or "0")
            if interval <= 0 or not dump_base:
                return

            def loop() -> None:
                _dump_results()
                timer = threading.Timer(interval, loop)
                timer.daemon = True
                timer.start()

            timer = threading.Timer(interval, loop)
            timer.daemon = True
            timer.start()

        result_file = os.environ.get("XDB_TUNABLE_FILE")
        if result_file:
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            tunable.set_filename(result_file)

        max_iterations = os.environ.get("XDB_TUNABLE_MAX_ITERATIONS")
        if max_iterations:
            tunable.set_max_tuning_iterations(int(max_iterations))

        max_duration_ms = os.environ.get("XDB_TUNABLE_MAX_DURATION_MS")
        if max_duration_ms:
            tunable.set_max_tuning_duration(int(max_duration_ms))

        if os.environ.get("XDB_TUNABLE_RECORD_UNTUNED", "1") == "1":
            tunable.record_untuned_enable(True)

        if os.environ.get("XDB_TUNABLE_TUNE", "1") == "1":
            tunable.tuning_enable(True)
        else:
            tunable.tuning_enable(False)

        tunable.enable(True)
        atexit.register(_dump_results)
        _start_periodic_dump()
        _log(
            "enabled "
            f"file={tunable.get_filename()} "
            f"enabled={tunable.is_enabled()} "
            f"tuning={tunable.tuning_is_enabled()} "
            f"record_untuned={tunable.record_untuned_is_enabled()} "
            f"max_iter={tunable.get_max_tuning_iterations()} "
            f"max_ms={tunable.get_max_tuning_duration()}"
        )
    except Exception as exc:  # pragma: no cover - diagnostic hook
        _log(f"enable_failed {type(exc).__name__}: {exc}")
