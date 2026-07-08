#!/usr/bin/env python3
"""Summarize R3.0 TunableOp result dumps."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TARGETS = {
    "mlp.gate_up_proj": {"n": 34816, "k": 5120},
    "mlp.down_proj": {"n": 5120, "k": 17408},
    "linear_attn.in_proj_qkvz": {"n": 16384, "k": 5120},
    "out_proj": {"n": 5120, "k": 6144},
    "self_attn.qkv_proj": {"n": 14336, "k": 5120},
}

SHAPE_RE = re.compile(r"tn_(?P<n>\d+)_(?P<m>\d+)_(?P<k>\d+)_ld_")


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("results.pid*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("results", []):
            if len(row) < 4:
                continue
            shape = str(row[1])
            match = SHAPE_RE.search(shape)
            parsed = match.groupdict() if match else {}
            rows.append(
                {
                    "file": str(path),
                    "pid": data.get("pid"),
                    "op": row[0],
                    "shape": shape,
                    "algo": row[2],
                    "time_ms": row[3],
                    "n": int(parsed["n"]) if parsed else None,
                    "m": int(parsed["m"]) if parsed else None,
                    "k": int(parsed["k"]) if parsed else None,
                }
            )
    return rows


def target_hits(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for name, target in TARGETS.items():
        hits = [
            row
            for row in rows
            if row.get("n") == target["n"] and row.get("k") == target["k"]
        ]
        out[name] = hits
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.root)
    hits = target_hits(rows)
    shape_counts = Counter((row.get("n"), row.get("m"), row.get("k")) for row in rows)
    payload = {
        "root": str(args.root),
        "rows": len(rows),
        "unique_shapes": len(shape_counts),
        "target_hits": {
            key: {"count": len(value), "examples": value[:5]} for key, value in hits.items()
        },
        "top_shapes": [
            {"n": n, "m": m, "k": k, "count": count}
            for (n, m, k), count in shape_counts.most_common(20)
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
