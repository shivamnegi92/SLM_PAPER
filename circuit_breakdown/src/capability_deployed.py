"""Capability exposure at scored prefix positions, using optimized tracking edits.

HellaSwag edits the last context token before scoring each ending. Perplexity
edits the last token of a fixed prefix and scores only its continuation. This
is active exposure on unrelated inputs, not an entity-trigger deployment test.
The legacy final-sequence-token protocol did not affect any scored prediction.

Results are versioned separately from the legacy artifacts. Confidence bounds
are reported per edit; repeated evaluations of the same items are not pooled
as independent observations. Global exposure is a stress test, not a bound.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from localize import Harness, pick_device
from capability import continuation_nll, hellaswag_correct, load_hellaswag, load_text
from intervene_margin import (
    get_control_baseline,
    make_layers,
    prepare_pairs,
    split_records,
)
from intervene_pareto import forward_logits, optimize_sample
from experiment_metrics import binary_flags, rate_ci, paired_delta_ci


PROTOCOL_VERSION = "active_prefix_v1"


def hooks_at(h, layers, vecs, mode, position=None):
    """Use a scored prefix position; mode='last' without position is legacy-only."""
    if mode not in {"prefix", "last", "global"}:
        raise ValueError(f"Unknown intervention mode: {mode}")
    if mode == "prefix" and position is None:
        raise ValueError("Prefix intervention requires an explicit position")
    if not layers or len(layers) != len(vecs) or len(set(layers)) != len(layers):
        raise ValueError("Provide one vector per distinct intervention layer")
    for layer_index, vector in zip(layers, vecs):
        if not 0 <= layer_index < len(h.layers):
            raise ValueError(f"Invalid intervention layer: {layer_index}")
        if vector.ndim not in (1, 2) or (vector.ndim == 2 and vector.shape[0] != 1):
            raise ValueError("Edit vectors must have shape [hidden] or [1, hidden]")
        if not torch.isfinite(vector).all().item():
            raise ValueError("Edit vectors must be finite")

    def make_hook(vector):
        def hook(module, inputs, output):
            tensor = (output[0] if isinstance(output, tuple) else output).clone()
            if vector.shape[-1] != tensor.shape[-1]:
                raise ValueError("Edit width must match the residual width")
            edit = vector.to(device=tensor.device, dtype=tensor.dtype)
            if mode == "global":
                tensor = tensor + edit
            else:
                edit_position = tensor.shape[1] - 1 if position is None else position
                if position is not None and not 0 <= edit_position < tensor.shape[1] - 1:
                    raise ValueError("The edit must precede at least one scored token")
                tensor[:, edit_position, :] = tensor[:, edit_position, :] + edit
            return (tensor,) + output[1:] if isinstance(output, tuple) else tensor
        return hook

    handles = []
    try:
        for layer_index, vector in zip(layers, vecs):
            handles.append(h.layers[layer_index].register_forward_hook(make_hook(vector)))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def measure(h, layers, vecs, mode, hs_items, text, tag, *, baseline=None,
            ppl_prefix_tokens=1, ppl_max_tokens=768):
    positions = []

    @contextmanager
    def intervention(position):
        handles = hooks_at(h, layers, vecs, mode, position=position)
        try:
            positions.append(position)
            yield
        finally:
            for handle in handles:
                handle.remove()

    flags = hellaswag_correct(h, hs_items, intervention=intervention)
    context_positions = list(positions)
    nll = continuation_nll(h, text, max_tokens=ppl_max_tokens,
                           prefix_tokens=ppl_prefix_tokens, intervention=intervention)
    acc, ci = rate_ci(flags)
    ppl = float(np.exp(nll.mean()))
    if not np.isfinite(nll).all() or not np.isfinite(ppl):
        raise ValueError("Capability scoring produced non-finite losses or perplexity")
    row = {
        "tag": tag, "hellaswag": acc, "hellaswag_ci": list(ci),
        "hellaswag_flags": flags.astype(int).tolist(), "n_hs": len(flags),
        "ppl": ppl, "mean_nll": float(nll.mean()), "token_nll": nll.tolist(),
        "n_ppl_tokens": len(nll), "ppl_prefix_tokens": ppl_prefix_tokens,
        "exposure": {
            "mode": "prefix" if mode == "last" else mode,
            "hellaswag_context_positions": context_positions,
            "ppl_position": ppl_prefix_tokens - 1,
            "edit_norms": [float(vector.norm().item()) for vector in vecs],
        },
    }
    if baseline is not None:
        delta, delta_ci = paired_delta_ci(baseline["hellaswag_flags"], flags)
        base_nll = np.asarray(baseline["token_nll"], dtype=np.float64)
        if base_nll.shape != nll.shape or baseline["ppl_prefix_tokens"] != ppl_prefix_tokens:
            raise ValueError("Baseline and intervention must score the same text continuation")
        row.update({
            "delta_hs": delta, "delta_hs_ci": list(delta_ci),
            "delta_hs_ci_method": "paired_gain_loss_wilson_bonferroni_approx95",
            "delta_mean_nll": float((nll - base_nll).mean()),
            "ppl_ratio": float(np.exp((nll - base_nll).mean())),
        })
    print(f"  {tag:30} HellaSwag {acc:6.1%} [{ci[0]:5.1%},{ci[1]:5.1%}]   ppl {ppl:8.2f}")
    if baseline is not None:
        print(f"    paired accuracy change {delta:+.1%} [{delta_ci[0]:+.1%},{delta_ci[1]:+.1%}]")
    return row


def summarize_edits(rows, baseline):
    """Descriptive edit averages; uncertainty stays with each paired evaluation."""
    if not rows:
        raise ValueError("At least one evaluated edit is required")
    mean_nll = float(np.mean([row["mean_nll"] for row in rows]))
    return {
        "hellaswag": float(np.mean([row["hellaswag"] for row in rows])),
        "delta_hs": float(np.mean([row["delta_hs"] for row in rows])),
        "ppl": float(np.exp(mean_nll)),
        "ppl_ratio": float(np.exp(mean_nll - baseline["mean_nll"])),
        "n_edits": len(rows), "n_hs_items": len(baseline["hellaswag_flags"]),
        "aggregation": "descriptive_mean_over_edits; paired_intervals_per_edit_only",
        "per_edit": rows,
    }


def run(args):
    if min(args.n_hs, args.n_edits, args.n_eval, args.n, args.stage_a_steps, args.ppl_chars) < 1:
        raise ValueError("Sample, edit and optimizer-step counts must be positive")
    if args.n_eval > args.n_edits:
        raise ValueError("n_eval cannot exceed n_edits")
    if not 1 <= args.ppl_prefix_tokens < args.ppl_max_tokens:
        raise ValueError("Choose a perplexity prefix shorter than the token limit")
    outdir = Path(args.outdir)
    output_path = outdir / f"capability_active_prefix_{Path(args.model).name}_s{args.seed}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new --outdir")

    hs_items = load_hellaswag(args.hellaswag, args.n_hs)
    text = load_text(args.text, args.ppl_chars)
    if len(hs_items) != args.n_hs:
        raise ValueError(f"Requested {args.n_hs} labeled HellaSwag items, found {len(hs_items)}")
    item_ids = [hashlib.sha256(json.dumps(item, ensure_ascii=True).encode()).hexdigest()
                for item in hs_items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Duplicate HellaSwag items would inflate the sample count")
    device = pick_device(args.device)
    harness = Harness(args.model, device)
    harness.model.requires_grad_(False)
    ds.restrict_to_single_token(harness.tok)
    layers = make_layers(len(harness.layers)) if not args.layers else sorted(set(args.layers))

    print(f"protocol={PROTOCOL_VERSION} device={device} model={args.model} layers={layers}")
    flags = hellaswag_correct(harness, hs_items)
    base_acc, base_ci = rate_ci(flags)
    base_nll = continuation_nll(harness, text, max_tokens=args.ppl_max_tokens,
                                prefix_tokens=args.ppl_prefix_tokens)
    baseline = {
        "hellaswag": base_acc, "hellaswag_ci": list(base_ci),
        "hellaswag_flags": flags.astype(int).tolist(),
        "ppl": float(np.exp(base_nll.mean())), "mean_nll": float(base_nll.mean()),
        "token_nll": base_nll.tolist(), "ppl_prefix_tokens": args.ppl_prefix_tokens,
    }
    print(f"  BASELINE HellaSwag {base_acc:.1%} [{base_ci[0]:.1%},{base_ci[1]:.1%}] "
          f"ppl {baseline['ppl']:.2f}")

    recs = prepare_pairs(harness, args.task, args.n, args.fewshot, args.seed)
    _, _, test_records = split_records(recs, args.seed)
    if len(test_records) < args.n_edits:
        raise ValueError(f"Requested {args.n_edits} edits, only {len(test_records)} test pairs available")
    ctrl_ids, ctrl_base = get_control_baseline(harness)
    cfg = {
        "lr": args.lr, "lr_b": args.lr, "stage_a_steps": args.stage_a_steps,
        "stage_b_steps": 0, "two_stage": False, "l2": 1e-3, "lam_kl": 0.0,
        "lam_kl_b": 0.0, "norm_budget": args.norm_budget,
        "max_control_drop": args.max_control_drop, "guard_every": 4,
        "margin_floor": 0.5, "keep_frac": 0.3, "w_norm": 1e-3, "w_hinge": 10.0,
    }
    edits, steer_flags = [], []
    for record in test_records[:args.n_edits]:
        edit_params = optimize_sample(harness, record, layers, ctrl_ids, ctrl_base, None, cfg)
        vectors = edit_params.detached()
        with torch.no_grad():
            pred = int(forward_logits(harness, record.corrupt_ids, layers, vectors, record.dpos).argmax().item())
        steer_flags.append(int(pred == record.cid))
        edits.append(vectors)

    rows = {key: [] for key in ("active_prefix", "global", "random_prefix", "zero_prefix")}
    for edit_index, vectors in enumerate(edits[:args.n_eval]):
        generator = torch.Generator(device="cpu").manual_seed(900 + args.seed + edit_index)
        random_vectors = [torch.randn(vector.shape, generator=generator).to(vector.device)
                          for vector in vectors]
        random_vectors = [random_vector / (random_vector.norm() + 1e-9) * vector.norm()
                          for random_vector, vector in zip(random_vectors, vectors)]
        conditions = [
            ("active_prefix", vectors, "prefix"),
            ("global", vectors, "global"),
            ("random_prefix", random_vectors, "prefix"),
            ("zero_prefix", [torch.zeros_like(vector) for vector in vectors], "prefix"),
        ]
        for key, condition_vectors, mode in conditions:
            row = measure(harness, layers, condition_vectors, mode, hs_items, text,
                          f"{key} edit {edit_index + 1}", baseline=baseline,
                          ppl_prefix_tokens=args.ppl_prefix_tokens,
                          ppl_max_tokens=args.ppl_max_tokens)
            if key == "zero_prefix":
                if row["hellaswag_flags"] != baseline["hellaswag_flags"] or not np.allclose(
                    row["token_nll"], baseline["token_nll"], rtol=1e-6, atol=1e-6
                ):
                    raise RuntimeError("Zero-edit control changed the baseline scores")
            row["edit_index"] = edit_index
            rows[key].append(row)

    steer, steer_ci = rate_ci(steer_flags)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "exposure_description": "active_prefix_on_unrelated_inputs; not an entity-trigger deployment",
        "model": args.model, "device": device, "dtype": "float32", "seed": args.seed,
        "layers": layers, "optimizer": cfg, "run_args": vars(args),
        "baseline": baseline, "steer": steer, "steer_ci": list(steer_ci),
        "steer_flags": steer_flags, "n_edits_optimized": len(edits),
        "n_edits_evaluated": args.n_eval, "hellaswag_item_ids": item_ids,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "uncertainty": "Wilson accuracy; approximate paired gain/loss bounds per edit; "
                       "no pooled edit CI or token-independence perplexity CI",
        "conditions": {key: summarize_edits(condition_rows, baseline)
                       for key, condition_rows in rows.items()},
    }
    print("\nMean outcomes over evaluated edits (paired intervals are in per_edit):")
    for key, condition in summary["conditions"].items():
        print(f"  {key:16} HellaSwag {condition['hellaswag']:.1%} "
              f"change {condition['delta_hs']:+.1%}  ppl ratio {condition['ppl_ratio']:.2f}")

    payload = json.dumps(summary, indent=2, allow_nan=False)
    outdir.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as output_file:
        output_file.write(payload)
    print(f"saved -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="../llama-3.2-3b")
    ap.add_argument("--task", default="intermediate", choices=["intermediate", "transfer"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=42)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--n-edits", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=3)
    ap.add_argument("--stage-a-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--norm-budget", type=float, default=4.0)
    ap.add_argument("--max-control-drop", type=float, default=0.20)
    ap.add_argument("--n-hs", type=int, default=240)
    ap.add_argument("--ppl-chars", type=int, default=3000)
    ap.add_argument("--ppl-prefix-tokens", type=int, default=32,
                    help="Prefix length including BOS; only the continuation is scored")
    ap.add_argument("--ppl-max-tokens", type=int, default=768)
    ap.add_argument("--hellaswag", default="data_bench/hellaswag_val.jsonl")
    ap.add_argument("--text", default="data_bench/tinyshakespeare.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--outdir", default=f"results/capability_{PROTOCOL_VERSION}")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
