#!/usr/bin/env python3
"""Build expression-specific RMOT GT after inference assets are audited.

This command is intentionally separate from model inference.  It consumes the
audited global-ID manifest and ``dataset/labels_with_ids`` only when scoring;
the inference command does not invoke it and never sees these target IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "expression"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_video(root: Path, video: str) -> Path:
    for split in ("train", "val", "test"):
        candidate = root / "dataset" / split / video
        if (candidate / "img1").is_dir():
            return candidate
    raise FileNotFoundError(f"video not found: {video}")


def active_ids(record, frame):
    active = set()
    for interval in record.get("temporal_intervals", []):
        track_id = int(interval["track_id"])
        if any(int(start) <= frame <= int(end) for start, end in interval.get("intervals", [])):
            active.add(track_id)
    return active


def read_label_file(path: Path, width: int, height: int):
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.replace(",", " ").split()
        if len(fields) < 6:
            continue
        track_id = int(float(fields[1]))
        cx, cy, box_w, box_h = (float(fields[i]) for i in range(2, 6))
        rows.append((track_id, cx * width - box_w * width / 2.0,
                     cy * height - box_h * height / 2.0,
                     box_w * width, box_h * height))
    return rows


def sequence_size(sequence: Path):
    values = {}
    for raw in (sequence / "seqinfo.ini").read_text().splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            if key.strip() in {"imWidth", "imHeight"}:
                values[key.strip()] = int(value.strip())
    return values["imWidth"], values["imHeight"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-141", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row.get("original_split") == "test"]
    if args.historical_141:
        records = [row for row in records if row.get("expression_id") != "v_4LXTUim5anY_c010:players"]
    records.sort(key=lambda row: row["expression_id"])

    labels_root = args.sportsmot_root / "dataset" / "labels_with_ids"
    missing = []
    for record in records:
        label_dir = labels_root / record["original_sportsmot_id"]
        if not label_dir.is_dir():
            missing.append(str(label_dir))
    if missing:
        raise SystemExit(
            "RMOT Test GT is not locally available for {} expression video(s); "
            "refusing to synthesize metrics. First missing path: {}".format(
                len(missing), missing[0]
            )
        )

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.output}")
    gt_root = args.output / "gt"
    gt_root.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records:
        video = record["original_sportsmot_id"]
        sequence = find_video(args.sportsmot_root, video)
        width, height = sequence_size(sequence)
        output_dir = gt_root / safe_name(record["expression_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "gt.txt"
        label_dir = labels_root / video
        lines = []
        for path in sorted(label_dir.glob("*.txt"), key=lambda item: int(item.stem)):
            frame = int(path.stem)
            active = active_ids(record, frame)
            for track_id, x, y, box_w, box_h in read_label_file(path, width, height):
                if track_id in active:
                    lines.append(
                        f"{frame},{track_id},{x:.6f},{y:.6f},{box_w:.6f},{box_h:.6f},1,1,1"
                    )
        output_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written.append({"expression_id": record["expression_id"], "video": video, "gt_file": str(output_file), "rows": len(lines)})

    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "historical_141": bool(args.historical_141),
        "num_expressions": len(records),
        "num_videos": len({row["original_sportsmot_id"] for row in records}),
        "inference_policy": "GT is scoring-only and is never passed to model inference",
        "records": written,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("num_expressions", "num_videos", "historical_141")}, indent=2))


if __name__ == "__main__":
    main()
