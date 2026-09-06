"""Aggregate the dissociation experiment: pool seeds, bootstrap CIs, run tests.

Reports the full metric ladder (most -> least discontinuous) so that effects
invisible to top-1 accuracy are still surfaced, per Schaeffer et al. (2304.15004).

Usage:
    python src/analyze_dissociation.py [--prefix diss] [--outdir results]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def boot_ci(vals, iters=10000, seed=0, stat=np.mean):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(iters, a.size))
    d = stat(a[idx], axis=1)
    return float(stat(a)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def load_group(prefix, cond, seeds, outdir):
    """Pool per-sample records across seeds for one condition."""
    g = {"steer": [], "steer2": [], "break": [], "dp2": [], "files": [],
         "dlogit": [], "ctrl": [], "edit": [], "pred_corrupt": []}
    for s in seeds:
        for f in sorted(glob.glob(f"{outdir}/{prefix}_{cond}_s{s}.json")):
            t = json.load(open(f))["test"]
            g["steer"] += t.get("steer_flags", [])
            g["steer2"] += t.get("steer2_flags", [])
            g["break"] += t.get("break_flags", [])
            g["dp2"] += t.get("delta_p2way_all", [])
            g["dlogit"].append(t.get("delta_logit_clean", 0.0))
            g["ctrl"].append(t.get("control_drop", 0.0))
            g["edit"].append(t.get("edit_norm_mean", 0.0))
            g["pred_corrupt"].append(t.get("pred_is_corrupt", 0.0))
            g["files"].append(f)
    return g


def summarize(g, label):
    if not g["files"]:
        return None
    steer, steer_ci = boot_ci(g["steer"], seed=1)
    s2, s2_ci = boot_ci(g["steer2"], seed=2)
    brk, brk_ci = boot_ci(g["break"], seed=3)
    dp2, dp2_ci = boot_ci(g["dp2"], seed=4)
    return {
        "label": label,
        "n_runs": len(g["files"]),
        "n_test": len(g["steer"]),
        "steer": steer, "steer_ci": steer_ci,
        "steer_2way": s2, "steer_2way_ci": s2_ci,
        "break": brk, "break_ci": brk_ci,
        "delta_p2way": dp2, "delta_p2way_ci": dp2_ci,
        "delta_logit": float(np.mean(g["dlogit"])),
        "delta_logit_sd": float(np.std(g["dlogit"])),
        "control_drop": float(np.mean(g["ctrl"])),
        "edit_norm": float(np.mean(g["edit"])),
        "pred_is_corrupt": float(np.mean(g["pred_corrupt"])),
    }


def diff_ci(a, b, iters=10000, seed=7):
    """Bootstrap CI for mean(a) - mean(b) on unpaired per-sample data."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0, (0.0, 0.0), 1.0
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(iters, b.size))].mean(axis=1)
    d = da - db
    obs = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    # two-sided bootstrap p: fraction of resamples crossing zero
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, (lo, hi), float(min(1.0, p))


def mcnemar(a, b):
    """Exact-ish McNemar on paired binary outcomes (same samples, 2 methods)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    tot = b01 + b10
    if tot == 0:
        return b01, b10, 1.0
    # two-sided exact binomial under p=0.5
    from math import comb
    k = min(b01, b10)
    p = sum(comb(tot, i) for i in range(0, k + 1)) / (2 ** tot) * 2
    return b01, b10, float(min(1.0, p))


def pct(x):
    return f"{x:6.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="diss")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    conds = [
        ("full", "unconstrained margin (full space)"),
        ("kl10", "+ KL trust region (lam=10)"),
        ("track8", "confined to TRACKING subspace r8"),
        ("comp8", "confined to ORTHOGONAL COMPLEMENT r8"),
    ]

    groups, rows = {}, []
    for key, label in conds:
        g = load_group(args.prefix, key, args.seeds, args.outdir)
        groups[key] = g
        s = summarize(g, label)
        if s:
            rows.append(s)

    print("\n" + "=" * 122)
    print("DISSOCIATION — metric ladder, most -> least discontinuous")
    print("=" * 122)
    hdr = (f"{'condition':38} {'run':>3} {'n':>4} {'STEER top1':>20} "
           f"{'STEER 2way':>20} {'BREAK':>18} {'d p2way':>16}")
    print(hdr)
    print("-" * 122)
    for r in rows:
        print(f"{r['label']:38} {r['n_runs']:3d} {r['n_test']:4d} "
              f"{pct(r['steer'])} [{r['steer_ci'][0]:4.0%},{r['steer_ci'][1]:4.0%}] "
              f"{pct(r['steer_2way'])} [{r['steer_2way_ci'][0]:4.0%},{r['steer_2way_ci'][1]:4.0%}] "
              f"{pct(r['break'])} [{r['break_ci'][0]:3.0%},{r['break_ci'][1]:3.0%}] "
              f"{r['delta_p2way']:+7.4f}")

    print("\n" + "-" * 122)
    print(f"{'condition':38} {'d logit':>18} {'ctrl drop':>11} {'|edit|':>9} {'pred=corrupt':>14}")
    print("-" * 122)
    for r in rows:
        print(f"{r['label']:38} {r['delta_logit']:+9.3f} +/-{r['delta_logit_sd']:5.3f} "
              f"{r['control_drop']:11.1%} {r['edit_norm']:9.2f} {r['pred_is_corrupt']:14.1%}")

    # ---- key statistical comparisons -----------------------------------
    print("\n" + "=" * 122)
    print("STATISTICAL TESTS")
    print("=" * 122)

    if groups["track8"]["files"] and groups["comp8"]["files"]:
        obs, ci, p = diff_ci(groups["track8"]["dp2"], groups["comp8"]["dp2"])
        print(f"\n[1] DISSOCIATION: tracking vs complement, continuous metric (delta p2way)")
        print(f"    diff = {obs:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  bootstrap p = {p:.4f}")
        st, _ = boot_ci(groups["track8"]["steer"], seed=1)
        sc, _ = boot_ci(groups["comp8"]["steer"], seed=1)
        print(f"    ...while top-1 STEER is {st:.1%} vs {sc:.1%} (both floored)")
        if p < 0.05 and abs(st - sc) < 0.05:
            print("    => CONFIRMED: continuous metric separates what top-1 cannot.")

    if groups["full"]["files"] and groups["track8"]["files"]:
        obs, ci, p = diff_ci(groups["full"]["steer"], groups["track8"]["steer"])
        print(f"\n[2] CONTROL: full-space vs tracking-subspace, top-1 STEER")
        print(f"    diff = {obs:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]  bootstrap p = {p:.4f}")
        if p < 0.05:
            print("    => Same layers, same budget: unconstrained steers, subspace-confined does not.")

    if groups["full"]["files"] and groups["kl10"]["files"]:
        nf, nk = len(groups["full"]["break"]), len(groups["kl10"]["break"])
        if nf == nk and nf > 0:
            b01, b10, p = mcnemar(groups["full"]["break"], groups["kl10"]["break"])
            print(f"\n[3] KL TRUST REGION: paired BREAK (McNemar, n={nf})")
            print(f"    full-broke/KL-ok = {b01}   full-ok/KL-broke = {b10}   exact p = {p:.4f}")
        obs, ci, p = diff_ci(groups["full"]["break"], groups["kl10"]["break"], seed=11)
        print(f"    unpaired BREAK diff = {obs:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]  p = {p:.4f}")
        so, sci, sp = diff_ci(groups["full"]["steer"], groups["kl10"]["steer"], seed=12)
        print(f"    STEER cost          = {so:+.1%}  95% CI [{sci[0]:+.1%}, {sci[1]:+.1%}]  p = {sp:.4f}")

    out = Path(args.outdir) / f"{args.prefix}_pooled_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
