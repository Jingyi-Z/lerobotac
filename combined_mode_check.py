"""Pre-flight for company-format recording: combined (2,52,3) + raw CSV.

Exercises the exact lerobot code path (factory -> PaxiniSensor combined mode
-> shared board -> raw writer) without arms/cameras/lerobot-record.

Usage:  python combined_mode_check.py --port COM10 --duration 20
Press each fingertip during the run; watch its finger's values move.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, tempfile, time, sys
import numpy as np

from lerobot.sensors import PaxiniSensorConfig
from lerobot.sensors.utils import make_sensors_from_configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--modules", type=int, nargs=2, default=[10, 18],
                    help="finger0 finger1 module indices")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    cfgs = {"paxini_fingertip": PaxiniSensorConfig(
        board_type="high_speed", port=args.port,
        output_format="combined", module_indices=list(args.modules),
        record_raw_csv=True, auto_calibrate=True,
    )}
    sensors = make_sensors_from_configs(cfgs)
    s = sensors["paxini_fingertip"]
    print("connecting (leave fingertips untouched for calibration)...")
    s.connect()
    print(f"  shape: {s.shape}  (expect (2, {s.metadata()['n_taxels']}, 3))")
    assert s.shape[0] == 2 and s.shape[2] == 3

    root = tempfile.mkdtemp(prefix="combined_check_")
    s.start_continuous_read()
    s.start_raw_episode(root, 0)
    s.wait_for_calibration(timeout_s=10.0)

    print(f"streaming {args.duration:.0f}s — press each fingertip in turn")
    print(f"{'t':>5}  {'f0 peak|F|':>10}  {'f1 peak|F|':>10}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.duration:
        obs = s.get_latest_data()
        if obs is not None:
            m0 = float(np.linalg.norm(obs[0], axis=1).max())
            m1 = float(np.linalg.norm(obs[1], axis=1).max())
            print(f"{time.monotonic()-t0:5.1f}  {m0:10.2f}  {m1:10.2f}")
        time.sleep(0.5)

    s.finish_raw_episode()
    s.disconnect()

    # verify the sidecar
    ep = os.path.join(root, "sensors", "paxini_fingertip", "episode_000000")
    csvs = sorted(glob.glob(os.path.join(ep, "sensor_*.csv")))
    print(f"\nsidecar dir: {ep}")
    for c in csvs:
        with open(c, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        dur = (int(rows[-1]["calibrated_timestamp_ns"])
               - int(rows[0]["calibrated_timestamp_ns"])) / 1e9
        hz = (len(rows) - 1) / dur if dur > 0 else 0
        cols = len(rows[0])
        peak = max(float(r["fz"]) for r in rows)
        print(f"  {os.path.basename(c)}: {len(rows)} rows, {hz:.1f} Hz, "
              f"{cols} cols, peak resultant Fz {peak:.1f} N")
    a = json.load(open(os.path.join(ep, "alignment.json")))
    print(f"  alignment anchor: {a['episode_start_timestamp_ns']}")
    print("\nPRE-FLIGHT PASS" if csvs else "\nNO CSVs — FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
