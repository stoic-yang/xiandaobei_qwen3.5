#!/usr/bin/env python3
"""Profile a steady decode window with hipprof session control.

The script sends one streaming OpenAI-compatible chat request, waits until the
first non-empty content chunk, starts a hipprof session, records a fixed number
of subsequent stream chunks, then stops and flushes the session. It is intended
for R2.0 decode-only profiling after the vLLM server has already been launched
under:

    hipprof --trace-off --session <name> ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def load_prompt(path: Path, row: int) -> str:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == row:
                data = json.loads(line)
                prompt = data.get("prompt")
                if not isinstance(prompt, str):
                    raise SystemExit(f"{path}:{row + 1} does not contain a string prompt")
                return prompt
    raise SystemExit(f"{path} has no row {row}")


def ctrl(args: argparse.Namespace, action: str) -> dict[str, Any]:
    cmd = [args.hipprof, "--session-client", args.session, action]
    started = time.perf_counter_ns()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=os.environ.copy(),
    )
    ended = time.perf_counter_ns()
    return {
        "action": action,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "elapsed_ms": (ended - started) / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--prompt-row", type=int, default=0)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--trace-chunks", type=int, default=64)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hipprof", default="hipprof")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file, args.prompt_row)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    events: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    first_content_ns: int | None = None
    trace_start_ns: int | None = None
    trace_stop_ns: int | None = None
    content_chunks = 0
    traced_chunks = 0
    request_start_ns = time.perf_counter_ns()

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            now_ns = time.perf_counter_ns()
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                events.append({"kind": "done", "t_ns": now_ns})
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                events.append({"kind": "bad_json", "t_ns": now_ns, "line": data[:500]})
                continue

            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            if not content:
                continue

            content_chunks += 1
            if first_content_ns is None:
                first_content_ns = now_ns
                control = ctrl(args, "--start")
                controls.append(control)
                trace_start_ns = time.perf_counter_ns()
                events.append({"kind": "trace_start", "t_ns": trace_start_ns})
                if control["returncode"] != 0:
                    raise SystemExit(f"hipprof start failed: {control}")
                continue

            traced_chunks += 1
            events.append(
                {
                    "kind": "content",
                    "t_ns": now_ns,
                    "chunk_index": content_chunks,
                    "traced_index": traced_chunks,
                    "chars": len(content),
                }
            )
            if traced_chunks >= args.trace_chunks:
                control = ctrl(args, "--stop")
                controls.append(control)
                trace_stop_ns = time.perf_counter_ns()
                events.append({"kind": "trace_stop", "t_ns": trace_stop_ns})
                controls.append(ctrl(args, "--flush"))
                if control["returncode"] != 0:
                    raise SystemExit(f"hipprof stop failed: {control}")
                break

    request_end_ns = time.perf_counter_ns()
    content_times = [e["t_ns"] for e in events if e.get("kind") == "content"]
    intervals_ms = [
        (b - a) / 1e6 for a, b in zip(content_times, content_times[1:])
    ]
    result = {
        "prompt_file": str(args.prompt_file),
        "prompt_row": args.prompt_row,
        "prompt_chars": len(prompt),
        "max_tokens": args.max_tokens,
        "trace_chunks_requested": args.trace_chunks,
        "content_chunks_seen": content_chunks,
        "traced_chunks": traced_chunks,
        "request_wall_ms": (request_end_ns - request_start_ns) / 1e6,
        "ttft_ms": None if first_content_ns is None else (first_content_ns - request_start_ns) / 1e6,
        "trace_wall_ms": None
        if trace_start_ns is None or trace_stop_ns is None
        else (trace_stop_ns - trace_start_ns) / 1e6,
        "chunk_intervals_ms": intervals_ms,
        "controls": controls,
        "events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "events"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
