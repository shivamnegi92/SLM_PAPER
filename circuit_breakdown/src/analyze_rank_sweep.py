"""Rank sweep analysis: does the tracking-vs-complement gap track DIRECTION
or merely DIMENSIONALITY?

The reviewer objection to the dissociation result is:

    "Your tracking subspace moves the decision variable more than the random
     complement simply because it is a *better-conditioned* r-dimensional
     subspace, or because r itself is doing the work. Show the gap is about
     WHICH directions, not HOW MANY."

Discriminating prediction:
  - If DIRECTION drives it, the tracking/complement ratio on the continuous
    metric stays large across ranks, and complement stays ~flat near zero.
  - If DIMENSIONALITY drives it, the two converge as r grows (both subspaces
    eventually span enough of the space to matter equally).

Usage:
    python src/analyze_rank_sweep.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def boot_ci(vals, iters=10000, seed=0):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def diff_ci(a, b, iters=10000, seed=7):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0, (0.0, 0.0), 1.0
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(iters, b.size))].mean(axis=1)
    d = da - db
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(a.mean() - b.mean()), (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
    ), float(min(1.0, p))


def load(cond, rank, outdir, seeds):
    """Pool per-sample records for one (condition, rank) across seeds."""
    acc = {"dp2": [], "steer": [], "break": [], "dlogit": [], "edit": [],
           "ranks": [], "n_runs": 0}
    for s in seeds:
        for f in sorted(glob.glob(f"{outdir}/diss_{cond}{rank}_s{s}.json")):
            d = json.load(open(f))
            t = d["test"]
            acc["dp2"] += t.get("delta_p2way_all", [])
            acc["steer"] += t.get("steer_flags", [])
            acc["break"] += t.get("break_flags", [])
            acc["dlogit"].append(t.get("delta_logit_clean", 0.0))
            acc["edit"].append(t.get("edit_norm_mean", 0.0))
            acc["ranks"].append(d.get("subspace_rank_actual", rank))
            acc["n_runs"] += 1
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 26])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    print("\n" + "=" * 116)
    print("RANK SWEEP — is the dissociation about DIRECTION or DIMENSIONALITY?")
    print("=" * 116)
    print(f"{'rank':>5} {'act':>4} {'runs':>5} {'n':>4} "
          f"{'TRACK d p2way':>24} {'COMP d p2way':>24} {'ratio':>8} {'p':>8}")
    print("-" * 116)

    rows = []
    for r in args.ranks:
        tr = load("track", r, args.outdir, args.seeds)
        co = load("comp", r, args.outdir, args.seeds)
        if tr["n_runs"] == 0 or co["n_runs"] == 0:
            continue

        tm, tci = boot_ci(tr["dp2"], seed=1)
        cm, cci = boot_ci(co["dp2"], seed=2)
        _, _, p = diff_ci(tr["dp2"], co["dp2"], seed=3)
        ratio = (tm / cm) if abs(cm) > 1e-9 else float("inf")
        act = int(np.mean(tr["ranks"])) if tr["ranks"] else r

        print(f"{r:5d} {act:4d} {tr['n_runs']:2d}/{co['n_runs']:<2d} {len(tr['dp2']):4d} "
              f"{tm:+8.5f} [{tci[0]:+.5f},{tci[1]:+.5f}] "
              f"{cm:+8.5f} [{cci[0]:+.5f},{cci[1]:+.5f}] "
              f"{ratio:8.1f} {p:8.4f}")

        rows.append({
            "rank": r, "rank_actual": act,
            "n_test": len(tr["dp2"]),
            "track_dp2": tm, "track_dp2_ci": tci,
            "comp_dp2": cm, "comp_dp2_ci": cci,
            "ratio": ratio, "p": p,
            "track_steer": float(np.mean(tr["steer"])) if tr["steer"] else 0.0,
            "comp_steer": float(np.mean(co["steer"])) if co["steer"] else 0.0,
            "track_break": float(np.mean(tr["break"])) if tr["break"] else 0.0,
            "comp_break": float(np.mean(co["break"])) if co["break"] else 0.0,
            "track_dlogit": float(np.mean(tr["dlogit"])) if tr["dlogit"] else 0.0,
            "comp_dlogit": float(np.mean(co["dlogit"])) if co["dlogit"] else 0.0,
            "track_edit": float(np.mean(tr["edit"])) if tr["edit"] else 0.0,
            "comp_edit": float(np.mean(co["edit"])) if co["edit"] else 0.0,
        })

    if not rows:
        print("  (no completed rank pairs yet)")
        return

    print("\n" + "-" * 116)
    print(f"{'rank':>5} {'TRACK steer':>12} {'COMP steer':>11} "
          f"{'TRACK brk':>10} {'COMP brk':>9} {'TRACK dlogit':>13} {'COMP dlogit':>12} "
          f"{'T|edit|':>8} {'C|edit|':>8}")
    print("-" * 116)
    for r in rows:
        print(f"{r['rank']:5d} {r['track_steer']:12.1%} {r['comp_steer']:11.1%} "
              f"{r['track_break']:10.1%} {r['comp_break']:9.1%} "
              f"{r['track_dlogit']:+13.3f} {r['comp_dlogit']:+12.3f} "
              f"{r['track_edit']:8.2f} {r['comp_edit']:8.2f}")

    # ---- verdict ------------------------------------------------------
    print("\n" + "=" * 116)
    print("VERDICT")
    print("=" * 116)

    steers = [r["track_steer"] for r in rows] + [r["comp_steer"] for r in rows]
    if max(steers) == 0.0:
        print(f"  * top-1 STEER is 0.0% for BOTH conditions at EVERY rank "
              f"({args.ranks}). The discontinuous metric is blind here.")

    ratios = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
    sig = [r for r in rows if r["p"] < 0.05]
    if ratios:
        print(f"  * tracking/complement ratio on d p2way ranges "
              f"{min(ratios):.1f}x - {max(ratios):.1f}x")
    print(f"  * {len(sig)}/{len(rows)} ranks show a significant gap (p<0.05)")

    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        comp_growth = abs(hi["comp_dp2"]) - abs(abs(lo["comp_dp2"]))
        print(f"  * complement d p2way from r={lo['rank']} to r={hi['rank']}: "
              f"{lo['comp_dp2']:+.5f} -> {hi['comp_dp2']:+.5f} (change {comp_growth:+.5f})")
        print(f"  * tracking   d p2way from r={lo['rank']} to r={hi['rank']}: "
              f"{lo['track_dp2']:+.5f} -> {hi['track_dp2']:+.5f}")
        if len(sig) == len(rows) and abs(comp_growth) < abs(hi["track_dp2"]) * 0.5:
            print("\n  => DIRECTION, not dimensionality. The gap persists across ranks and")
            print("     the complement stays near-inert as r grows.")
        else:
            print("\n  => Mixed/converging: inspect whether the gap narrows with r.")

    out = Path(args.outdir) / "rank_sweep_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
