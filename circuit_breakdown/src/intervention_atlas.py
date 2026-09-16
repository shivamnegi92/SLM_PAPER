"""Block A: Intervention Failure Atlas

Compares 3 intervention families under one protocol:
1) single-layer additive
2) multi-layer additive
3) DAS-style low-rank map

Expanded sweep focuses on decision-level failure points:
- alphas: 2, 4, 8, 16
- layer windows include downstream shift [18,20,22,24]

Outputs:
- per-method per-alpha mapping: alpha -> (logit lift, top-1 steer, break)
- semantic collapse flags on control prompts at high alpha

No weight updates. Pure inference-time hooks.
"""
from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device, build_fewshot


CONTROL_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "A short bedtime story begins",
]


@dataclass
class PairRec:
    cid: int
    kid: int
    clean_ids: torch.Tensor
    corrupt_ids: torch.Tensor
    dpos: int


def prepare_pairs(h: Harness, task: str, n: int, fewshot: int, seed: int) -> list[PairRec]:
    if task == "intermediate":
        min_len, max_len = 3, 4
    else:
        min_len, max_len = 1, 2
    pairs = ds.generate(task, n + fewshot, seed=seed, min_len=min_len, max_len=max_len)
    ds.self_check(pairs)
    prefix = build_fewshot(pairs, fewshot, task)

    out: list[PairRec] = []
    for p in pairs[:n]:
        cid, kid = h.first_id(p.clean_target), h.first_id(p.corrupt_target)
        if cid == kid:
            continue
        clean_ids = h.encode(prefix + p.clean_prompt)
        corrupt_ids = h.encode(prefix + p.corrupt_prompt)
        if clean_ids.shape[1] != corrupt_ids.shape[1]:
            continue
        diffs = [i for i, (a, b) in enumerate(zip(clean_ids[0].tolist(), corrupt_ids[0].tolist())) if a != b]
        if len(diffs) != 1:
            continue
        out.append(PairRec(cid, kid, clean_ids, corrupt_ids, diffs[0]))
    return out


def split_records(records: list[PairRec], seed: int, train_frac=0.6, dev_frac=0.2):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    ntr = int(n * train_frac)
    ndv = int(n * dev_frac)
    tr = [records[i] for i in idx[:ntr]]
    dv = [records[i] for i in idx[ntr:ntr + ndv]]
    te = [records[i] for i in idx[ntr + ndv:]]
    return tr, dv, te


def cache_pos(h: Harness, ids: torch.Tensor, dpos: int):
    outs = h.cache_layer_outputs(ids)
    ent = [o[0, dpos].detach().clone() for o in outs]
    fin = [o[0, -1].detach().clone() for o in outs]
    return ent, fin


def fit_additive(h: Harness, train: list[PairRec]):
    n_layers = len(h.layers)
    s_ent = [None] * n_layers
    s_fin = [None] * n_layers
    for r in train:
        ce, cf = cache_pos(h, r.clean_ids, r.dpos)
        ke, kf = cache_pos(h, r.corrupt_ids, r.dpos)
        for L in range(n_layers):
            de = ce[L] - ke[L]
            df = cf[L] - kf[L]
            s_ent[L] = de if s_ent[L] is None else (s_ent[L] + de)
            s_fin[L] = df if s_fin[L] is None else (s_fin[L] + df)

    def normed(arr):
        out = []
        for d in arr:
            d = d / max(1, len(train))
            out.append(d / (d.norm().item() + 1e-9))
        return out

    return normed(s_ent), normed(s_fin)


def fit_das(h: Harness, train: list[PairRec], rank: int = 8, ridge: float = 1e-3):
    n_layers = len(h.layers)
    per_clean = [[] for _ in range(n_layers)]
    per_corr = [[] for _ in range(n_layers)]

    for r in train:
        ce, _ = cache_pos(h, r.clean_ids, r.dpos)
        ke, _ = cache_pos(h, r.corrupt_ids, r.dpos)
        for L in range(n_layers):
            per_clean[L].append(ce[L].float().cpu())
            per_corr[L].append(ke[L].float().cpu())

    params = []
    for L in range(n_layers):
        Xc = torch.stack(per_corr[L], 0)
        Xk = torch.stack(per_clean[L], 0)
        D = Xk - Xc
        mu = Xc.mean(dim=0)
        X0 = Xc - mu
        _, _, Vh = torch.linalg.svd(X0, full_matrices=False)
        rk = int(min(rank, Vh.shape[0]))
        B = Vh[:rk]
        Z = X0 @ B.T
        A = Z.T @ Z + ridge * torch.eye(rk)
        M = torch.linalg.solve(A, Z.T @ D)
        clip = D.norm(dim=1).mean().item() * 1.5 + 1e-6
        params.append({"mu": mu, "Bt": B.T.contiguous(), "M": M.contiguous(), "clip": clip})
    return params


def add_hooks_additive(h: Harness, layers, dpos, ve, vf, a_ent, a_fin):
    hs = []

    def mk(v1, v2):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            p = dpos if dpos >= 0 else t.shape[1] + dpos
            if a_ent != 0:
                t[:, p, :] = t[:, p, :] + a_ent * v1.to(t.dtype)
            if a_fin != 0:
                t[:, -1, :] = t[:, -1, :] + a_fin * v2.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return hook

    for L in layers:
        hs.append(h.layers[L].register_forward_hook(mk(ve[L], vf[L])))
    return hs


def add_hooks_das(h: Harness, layers, dpos, params, alpha):
    hs = []

    def mk(p):
        mu = p["mu"].to(h.device)
        Bt = p["Bt"].to(h.device)
        M = p["M"].to(h.device)
        clip = p["clip"]

        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            pos = dpos if dpos >= 0 else t.shape[1] + dpos
            x = t[:, pos, :].float()
            z = (x - mu) @ Bt
            d = z @ M
            n = torch.norm(d, dim=-1, keepdim=True) + 1e-9
            fac = torch.clamp(torch.tensor(clip, device=t.device) / n, max=1.0)
            d = d * fac
            t[:, pos, :] = t[:, pos, :] + alpha * d.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t

        return hook

    for L in layers:
        hs.append(h.layers[L].register_forward_hook(mk(params[L])))
    return hs


def eval_records(h: Harness, recs: list[PairRec], apply_fn):
    steer_flags, break_flags, dlog = [], [], []
    for r in recs:
        with torch.no_grad():
            base = h.model(r.corrupt_ids).logits[0, -1]
        base_clean = base[r.cid].item()

        hs = apply_fn(r, mode="steer")
        with torch.no_grad():
            logits = h.model(r.corrupt_ids).logits[0, -1]
            pred = logits.argmax().item()
        for hh in hs:
            hh.remove()

        steer_flags.append(int(pred == r.cid))
        dlog.append(float(logits[r.cid].item() - base_clean))

        hb = apply_fn(r, mode="break")
        with torch.no_grad():
            p2 = h.model(r.clean_ids).logits[0, -1].argmax().item()
        for hh in hb:
            hh.remove()
        break_flags.append(int(p2 != r.cid))

    return {
        "steer": float(np.mean(steer_flags)) if steer_flags else 0.0,
        "break": float(np.mean(break_flags)) if break_flags else 0.0,
        "delta_logit_clean": float(np.mean(dlog)) if dlog else 0.0,
        "steer_flags": steer_flags,
        "break_flags": break_flags,
    }


def is_semantic_collapse(txt: str) -> bool:
    if not txt:
        return True
    if "�" in txt:
        return True
    printable = sum(ch.isprintable() for ch in txt) / len(txt)
    alnum = sum(ch.isalnum() for ch in txt) / len(txt)
    punct = sum(ch in string.punctuation for ch in txt) / len(txt)

    # long repeated char run
    mx = 1
    cur = 1
    for i in range(1, len(txt)):
        if txt[i] == txt[i - 1]:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 1

    return printable < 0.9 or (alnum < 0.2 and punct > 0.45) or mx >= 6


def generate_with_hooks(h: Harness, prompt: str, add_hooks_fn, max_new_tokens: int = 5) -> str:
    ids = h.encode(prompt)
    for _ in range(max_new_tokens):
        hs = add_hooks_fn()
        with torch.no_grad():
            logits = h.model(ids).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        for hh in hs:
            hh.remove()
        ids = torch.cat([ids, nxt], dim=1)
    txt = h.tok.decode(ids[0], skip_special_tokens=True)
    return txt[len(prompt):]


def semantic_collapse_rate(h: Harness, method: str, cfg: dict, assets: dict):
    # control generation with intervention applied at final token position
    collapse = 0
    samples = []

    for p in CONTROL_PROMPTS:
        if method in {"single_layer_additive", "multi_layer_additive"}:
            ve, vf = assets["additive"]
            layers = cfg["layers"]
            ae = cfg.get("alpha_ent", cfg.get("alpha", 0.0))
            af = cfg.get("alpha_fin", 0.0)

            def add():
                return add_hooks_additive(h, layers, -1, ve, vf, ae, af)

        elif method == "das_multilayer":
            params = assets["das"]
            layers = cfg["layers"]
            a = cfg["alpha"]

            def add():
                return add_hooks_das(h, layers, -1, params, a)

        else:
            raise ValueError(method)

        cont = generate_with_hooks(h, p, add, max_new_tokens=5)
        bad = is_semantic_collapse(cont)
        collapse += int(bad)
        samples.append({"prompt": p, "continuation": cont, "collapse": bad})

    return collapse / len(CONTROL_PROMPTS), samples


def choose_best_on_dev(rows):
    # prioritize steer, then less break, then higher logit lift
    rows = sorted(rows, key=lambda r: (r["metrics"]["steer"], -r["metrics"]["break"], r["metrics"]["delta_logit_clean"]), reverse=True)
    return rows[0]


def run(args):
    device = pick_device(args.device)
    h = Harness(args.model, device)
    ds.restrict_to_single_token(h.tok)

    recs = prepare_pairs(h, args.task, args.n, args.fewshot, args.seed)
    tr, dv, te = split_records(recs, args.seed)

    print(f"device={device} model={args.model} task={args.task}")
    print(f"usable={len(recs)} split train/dev/test={len(tr)}/{len(dv)}/{len(te)}")

    layer_windows = [
        [14, 16, 18, 20],
        [18, 20, 22, 24],  # downstream shift window (requested)
    ]
    if args.layers:
        layer_windows = [sorted(set(args.layers))]

    alphas = args.alphas
    ve, vf = fit_additive(h, tr)
    das_params = fit_das(h, tr, rank=args.das_rank, ridge=args.das_ridge)

    assets = {"additive": (ve, vf), "das": das_params}

    all_results = []

    # --- method 1: single-layer additive ---
    single_layers = sorted(set([14, 16, 18, 20, 22, 24]))
    for a in alphas:
        dev_rows = []
        for L in single_layers:
                def ap(r, mode="steer"):
                    aa = a if mode == "steer" else -a
                    return add_hooks_additive(h, [L], r.dpos, ve, vf, aa, 0.0)

                m = eval_records(h, dv, ap)
                dev_rows.append({"cfg": {"layers": [L], "alpha": a, "alpha_ent": a, "alpha_fin": 0.0}, "metrics": m})

        best = choose_best_on_dev(dev_rows)
        cfg = best["cfg"]

        def ap_te(r, mode="steer"):
            aa = cfg["alpha"] if mode == "steer" else -cfg["alpha"]
            return add_hooks_additive(h, cfg["layers"], r.dpos, ve, vf, aa, 0.0)

        mt = eval_records(h, te, ap_te)
        collapse_rate, samples = semantic_collapse_rate(h, "single_layer_additive", cfg, assets) if a >= 8 else (0.0, [])
        all_results.append({
            "method": "single_layer_additive",
            "alpha": a,
            "best_cfg": cfg,
            "test": mt,
            "semantic_collapse_rate": collapse_rate,
            "semantic_samples": samples,
        })

    # --- method 2: multi-layer additive ---
    for a in alphas:
        dev_rows = []
        for w in layer_windows:
            for af in [0.0, a]:
                def ap(r, mode="steer"):
                    se = a if mode == "steer" else -a
                    sf = af if mode == "steer" else -af
                    return add_hooks_additive(h, w, r.dpos, ve, vf, se, sf)

                m = eval_records(h, dv, ap)
                dev_rows.append({"cfg": {"layers": w, "alpha_ent": a, "alpha_fin": af}, "metrics": m})

        best = choose_best_on_dev(dev_rows)
        cfg = best["cfg"]

        def ap_te(r, mode="steer"):
            se = cfg["alpha_ent"] if mode == "steer" else -cfg["alpha_ent"]
            sf = cfg["alpha_fin"] if mode == "steer" else -cfg["alpha_fin"]
            return add_hooks_additive(h, cfg["layers"], r.dpos, ve, vf, se, sf)

        mt = eval_records(h, te, ap_te)
        collapse_rate, samples = semantic_collapse_rate(h, "multi_layer_additive", cfg, assets) if a >= 8 else (0.0, [])
        all_results.append({
            "method": "multi_layer_additive",
            "alpha": a,
            "best_cfg": cfg,
            "test": mt,
            "semantic_collapse_rate": collapse_rate,
            "semantic_samples": samples,
        })

    # --- method 3: DAS-style ---
    for a in alphas:
        dev_rows = []
        for w in layer_windows:
            def ap(r, mode="steer"):
                aa = a if mode == "steer" else -a
                return add_hooks_das(h, w, r.dpos, das_params, aa)

            m = eval_records(h, dv, ap)
            dev_rows.append({"cfg": {"layers": w, "alpha": a, "rank": args.das_rank}, "metrics": m})

        best = choose_best_on_dev(dev_rows)
        cfg = best["cfg"]

        def ap_te(r, mode="steer"):
            aa = cfg["alpha"] if mode == "steer" else -cfg["alpha"]
            return add_hooks_das(h, cfg["layers"], r.dpos, das_params, aa)

        mt = eval_records(h, te, ap_te)
        collapse_rate, samples = semantic_collapse_rate(h, "das_multilayer", cfg, assets) if a >= 8 else (0.0, [])
        all_results.append({
            "method": "das_multilayer",
            "alpha": a,
            "best_cfg": cfg,
            "test": mt,
            "semantic_collapse_rate": collapse_rate,
            "semantic_samples": samples,
        })

    # clean array requested: alpha vs logit-lift vs behavior flip
    array_map = []
    for r in all_results:
        array_map.append({
            "method": r["method"],
            "alpha": r["alpha"],
            "behavior_flip_top1": r["test"]["steer"],
            "logit_lift": r["test"]["delta_logit_clean"],
            "break_side_effect": r["test"]["break"],
            "semantic_collapse_rate": r["semantic_collapse_rate"],
        })

    # print concise scoreboard
    print("\n=== BLOCK A SCOREBOARD (TEST) ===")
    print(f"{'method':24} {'alpha':>6} {'flip':>8} {'break':>8} {'dlogit':>9} {'collapse':>10}")
    for r in array_map:
        print(f"{r['method'][:24]:24} {r['alpha']:6.1f} {r['behavior_flip_top1']:7.1%} {r['break_side_effect']:7.1%} {r['logit_lift']:9.3f} {r['semantic_collapse_rate']:9.1%}")

    out = {
        "model": args.model,
        "task": args.task,
        "n": len(recs),
        "split": {"train": len(tr), "dev": len(dv), "test": len(te)},
        "alphas": alphas,
        "layer_windows": layer_windows,
        "results": all_results,
        "array_map": array_map,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    p = outdir / f"intervention_atlas_{tag}_{args.task}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--alphas", type=float, nargs="*", default=[2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--das-rank", type=int, default=8)
    ap.add_argument("--das-ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
