"""Merge multiple company-format tactile datasets into one.

Wraps lerobot.datasets.aggregate.aggregate_datasets (which merges data/,
videos/, meta/) and additionally merges the raw-CSV sidecars under
sensors/<name>/episode_XXXXXX/, renumbering episode indices by the
cumulative episode offset of each source dataset.

Usage:
    python merge_tactile_datasets.py \
        --sources so101_paxini_test_20260818_152509 so101_paxini_test_20260818_154004 ... \
        --output so101_ball_pick_place_tactile \
        [--user Jingyi-Z]

Sources are folder names under ~/.cache/huggingface/lerobot/<user>/.
The output dataset is created next to them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", required=True,
                   help="Source dataset folder names (order = episode order).")
    p.add_argument("--output", required=True, help="Output dataset folder name.")
    p.add_argument("--user", default="Jingyi-Z", help="HF user / cache subfolder.")
    args = p.parse_args()

    base = Path.home() / ".cache" / "huggingface" / "lerobot" / args.user
    src_roots = [base / s for s in args.sources]
    dst_root = base / args.output

    for r in src_roots:
        if not (r / "meta" / "info.json").exists():
            sys.exit(f"missing dataset: {r}")
    if dst_root.exists():
        sys.exit(f"output already exists: {dst_root} — delete it first.")

    # 1. main tables + videos + metadata
    aggregate_datasets(
        repo_ids=[f"{args.user}/{s}" for s in args.sources],
        aggr_repo_id=f"{args.user}/{args.output}",
        roots=src_roots,
        aggr_root=dst_root,
    )

    # 2. raw-CSV sidecars, renumbered
    offset = 0
    copied = 0
    for r in src_roots:
        info = json.loads((r / "meta" / "info.json").read_text())
        n_eps = info["total_episodes"]
        sensors_dir = r / "sensors"
        if sensors_dir.exists():
            for sensor_name_dir in sorted(sensors_dir.iterdir()):
                if not sensor_name_dir.is_dir():
                    continue
                for ep_dir in sorted(sensor_name_dir.glob("episode_*")):
                    idx = int(ep_dir.name.split("_")[1])
                    if idx >= n_eps:
                        print(f"skipping orphan sidecar {ep_dir}")
                        continue
                    dst = (dst_root / "sensors" / sensor_name_dir.name
                           / f"episode_{offset + idx:06d}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(ep_dir, dst)
                    copied += 1
        offset += n_eps

    # 3. verify
    info = json.loads((dst_root / "meta" / "info.json").read_text())
    total_eps = info["total_episodes"]
    sidecars = sorted((dst_root / "sensors").rglob("episode_*"))
    print(f"\nmerged: {total_eps} episodes, {info['total_frames']} frames, "
          f"{copied} sidecar folders copied")
    ok = True
    if len(sidecars) != total_eps:
        print(f"WARNING: {len(sidecars)} sidecar folders != {total_eps} episodes")
        ok = False
    expected = {f"episode_{i:06d}" for i in range(total_eps)}
    actual = {d.name for d in sidecars}
    if expected != actual:
        print(f"WARNING: sidecar index gaps: missing={sorted(expected - actual)[:5]} "
              f"extra={sorted(actual - expected)[:5]}")
        ok = False
    for d in sidecars:
        files = {f.name for f in d.iterdir()}
        if not {"sensor_1.csv", "sensor_2.csv", "alignment.json"} <= files:
            print(f"WARNING: incomplete sidecar {d}: {files}")
            ok = False
    print("all sidecar checks passed" if ok else "CHECKS FAILED — inspect before upload")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
