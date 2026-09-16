"""Sequential, resumable local evaluation of a frozen tracking-control study."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from analyze_dissociation import analyze
from experiment_metrics import rate_ci
from intervene_margin import CONTROL_PROMPTS, control_degradation
from intervene_pareto import eval_set, fit_subspace_bases, optimize_sample
from localize import Harness, pick_device
from study_data import fingerprint, load_manifest, manifest_records, write_new


VERSION = "tracking_control_v1"
HELDOUT_CONTROLS = [
    "Water freezes at", "The opposite of hot is", "A triangle has",
    "The days after Monday are", "Plants need sunlight and", "The largest ocean is",
    "An hour contains sixty", "A dictionary lists", "The color of fresh grass is",
    "When ice melts it becomes", "A week contains seven", "Three times three equals",
]


def model_fingerprint(model):
    root = Path(model)
    files = {name: (root / name).read_text() for name in
             ("config.json", "tokenizer_config.json", "special_tokens_map.json") if (root / name).exists()}
    return fingerprint({"name": root.name, "configuration": files,
                        "weights": [(path.name, path.stat().st_size)
                                    for path in sorted(root.glob("*.safetensors"))]})


def matched_layers(count):
    return [round((layer + 1) / 28 * count) - 1 for layer in (18, 20, 22, 24)]


def source_fingerprint():
    root = Path(__file__).parent
    return fingerprint({name: (root / name).read_text() for name in
                        ("dataset.py", "study_data.py", "experiment_metrics.py",
                         "intervene_margin.py", "intervene_pareto.py", "run_validated_study.py")})


def control_set(harness, prompts):
    identifiers = [harness.encode(prompt) for prompt in prompts]
    with torch.no_grad():
        predictions = [int(harness.model(ids).logits[0, -1].argmax()) for ids in identifiers]
    return identifiers, predictions


def baseline_metrics(harness, records):
    clean, corrupt = [], []
    with torch.no_grad():
        for index, record in enumerate(records):
            clean_logits = harness.model(record.clean_ids).logits[0, -1]
            corrupt_logits = harness.model(record.corrupt_ids).logits[0, -1]
            if not torch.isfinite(clean_logits).all() or not torch.isfinite(corrupt_logits).all():
                raise ValueError("Non-finite baseline logits")
            clean.append(int(clean_logits.argmax() == record.cid))
            corrupt.append(int(corrupt_logits.argmax() == record.kid))
            if (index + 1) % 8 == 0:
                print(f"baseline {index + 1}/{len(records)}", flush=True)
    clean_mean, clean_ci = rate_ci(clean)
    corrupt_mean, corrupt_ci = rate_ci(corrupt)
    return {"clean_accuracy": clean_mean, "clean_ci": clean_ci,
            "counterfactual_accuracy": corrupt_mean, "counterfactual_ci": corrupt_ci,
            "clean_flags": clean, "counterfactual_flags": corrupt, "n": len(clean)}


def layer_budgets(harness, records, layers, relative):
    norms = []
    for record in records:
        cached = harness.cache_layer_outputs(record.corrupt_ids)
        norms.append([cached[layer][0, record.dpos].norm().item() for layer in layers])
    mean = np.asarray(norms).mean(axis=0)
    return (mean * relative).tolist(), mean.tolist()


def configuration(budgets, steps, learning_rate, objective="cross_entropy"):
    return {"lr": learning_rate, "lr_b": learning_rate, "stage_a_steps": steps,
            "stage_b_steps": 0, "two_stage": False, "l2": 1e-3, "lam_kl": 0.0,
            "lam_kl_b": 0.0, "norm_budget": budgets, "max_control_drop": 0.2,
            "guard_every": 4, "margin_floor": 0.5, "keep_frac": 0.3,
            "w_norm": 1e-3, "w_hinge": 10.0, "objective": objective}


def basis_sets(harness, train, layers, seed):
    tracking, info = fit_subspace_bases(harness, train, layers, 8, return_info=True)
    controls, control_info = [], []
    for layer, basis in zip(layers, tracking):
        generator = torch.Generator().manual_seed(12000 + seed * 100 + layer)
        random = torch.randn(basis.shape, generator=generator, dtype=torch.float64)
        cpu_basis = basis.detach().cpu().double()
        random -= (random @ cpu_basis.T) @ cpu_basis
        orthonormal, _ = torch.linalg.qr(random.T)
        control = orthonormal.T[:len(basis)].float().to(harness.device)
        controls.append(control)
        control_info.append({"layer": layer, "rank": len(control),
                             "orthogonal_error": float((control @ basis.T).abs().max()),
                             "seed": 12000 + seed * 100 + layer})
    if any(item["tracking_rank"] != 8 for item in info):
        raise ValueError("Training set does not support the preregistered rank8")
    return {"full": None, "track8": tracking, "comp8": controls}, {"tracking": info, "control": control_info}


def effective_rate(harness, basis, learning_rate):
    return learning_rate * (harness.model.config.hidden_size / len(basis[0])) ** 0.5 if basis else learning_rate


def read_completed(path, comparison):
    if not path.exists():
        return None
    with path.open() as stream:
        saved = json.load(stream)
    if not saved.get("completed") or saved.get("comparison") != comparison:
        raise ValueError(f"Existing result has an incompatible protocol: {path}")
    return saved


def calibrate(harness, splits, layers, budgets, bases, controls, steps, checkpoint_dir=None,
              comparison=None, max_new_cases=None):
    cases = splits["dev"][:4]
    results, selected = {}, {}
    new_cases = 0
    for condition in ("full", "track8", "comp8"):
        choices = []
        for rate in (0.025, 0.05, 0.1):
            print(f"calibration {condition} lr={rate}: {len(cases)} development cases", flush=True)
            cfg = configuration(budgets, steps, effective_rate(harness, bases[condition], rate))
            traces = []
            for index, record in enumerate(cases):
                path = (Path(checkpoint_dir) / f"{condition}_lr{rate}_case{index}.json"
                        if checkpoint_dir is not None else None)
                saved = read_completed(path, comparison) if path is not None else None
                if saved is not None:
                    if saved["sample_id"] != record.sample_id or saved["input_hash"] != record.input_hash:
                        raise ValueError("Calibration checkpoint input differs")
                    traces.append(saved["trace"])
                    continue
                if max_new_cases is not None and new_cases >= max_new_cases:
                    print("Calibration checkpoint batch complete; resume with the same command", flush=True)
                    return None
                params = optimize_sample(harness, record, layers, *controls, bases[condition], cfg)
                traces.append(params.trace)
                if path is not None:
                    write_new(path, {"completed": True, "comparison": comparison,
                                     "sample_id": record.sample_id, "input_hash": record.input_hash,
                                     "trace": params.trace})
                new_cases += 1
                print(f"  saved case {index + 1}/{len(cases)}", flush=True)
                del params
                gc.collect()
                if harness.device == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
            margin = float(np.mean([trace[-1]["target_margin"] for trace in traces]))
            choices.append({"base_lr": rate, "mean_target_margin": margin,
                            "success": float(np.mean([trace[-1]["target_success"] for trace in traces])),
                            "traces": traces})
        best = max(choices, key=lambda choice: choice["mean_target_margin"])
        selected[condition] = best["base_lr"]
        results[condition] = choices
        print(f"calibration {condition}: selected lr={best['base_lr']}, dev margin={best['mean_target_margin']:.3f}", flush=True)
    return {"selected": selected, "candidates": results, "n_dev": len(cases),
            "criterion": "max mean full-vocabulary target margin on development only"}


def run(args):
    manifest = load_manifest(args.manifest)
    model_name = Path(args.model).name
    root = Path(args.outdir) / model_name
    comparison_base = {
        "version": VERSION, "manifest_sha256": manifest["sha256"],
        "model": model_name, "model_fingerprint": model_fingerprint(args.model),
        "source_sha256": source_fingerprint(), "relative_budget": args.relative_budget,
        "steps": args.steps, "objective": "cross_entropy", "lr_grid": [0.025, 0.05, 0.1],
        "seed_policy": manifest["seeds"], "dtype": "float32",
        "torch_version": torch.__version__, "guard_prompts": CONTROL_PROMPTS,
    }
    device = pick_device(args.device)
    harness = Harness(args.model, device)
    harness.model.requires_grad_(False)
    layers = matched_layers(len(harness.layers))
    comparison_base["layers"] = layers
    controls = control_set(harness, CONTROL_PROMPTS)
    heldout = control_set(harness, HELDOUT_CONTROLS)
    for task in args.tasks:
        comparison = {**comparison_base, "task": task}
        output = root / task
        calibration_path = output / "calibration.json"
        calibration = read_completed(calibration_path, comparison)
        if calibration is None:
            splits = manifest_records(harness, manifest, task, 0)
            competence = baseline_metrics(harness, splits["dev"])
            eligible = min(competence["clean_accuracy"], competence["counterfactual_accuracy"]) >= 0.8
            calibration = {"comparison": comparison, "baseline_dev": competence,
                           "eligible_for_tracking_claim": eligible}
            if eligible:
                budgets, _ = layer_budgets(harness, splits["train"], layers, args.relative_budget)
                bases, _ = basis_sets(harness, splits["train"], layers, 0)
                calibrated = calibrate(harness, splits, layers, budgets, bases, controls, args.steps,
                                       output / "calibration_cases", comparison,
                                       getattr(args, "max_new_cases", None))
                del bases
                if calibrated is None:
                    return
                calibration.update(calibrated)
            calibration["completed"] = True
            write_new(calibration_path, calibration)
        if args.stage == "calibrate":
            continue
        if not calibration["eligible_for_tracking_claim"]:
            print(f"{model_name}/{task}: baseline competence gate not met; report task limitation, no causal-control claim", flush=True)
            if args.stage == "all":
                for seed in args.seeds:
                    path = output / f"baseline_s{seed}.json"
                    if read_completed(path, comparison) is None:
                        records = manifest_records(harness, manifest, task, seed)["test"]
                        write_new(path, {"completed": True, "comparison": comparison, "seed": seed,
                                         "baseline_test": baseline_metrics(harness, records)})
            continue
        for seed in args.seeds:
            splits = manifest_records(harness, manifest, task, seed)
            pending = [condition for condition in ("full", "track8", "comp8")
                       if read_completed(output / f"diss_{condition}_s{seed}.json", comparison) is None]
            if not pending:
                continue
            budgets, residual_norms = layer_budgets(harness, splits["train"], layers, args.relative_budget)
            bases, information = basis_sets(harness, splits["train"], layers, seed)
            for condition in pending:
                started = time.monotonic()
                cfg = configuration(budgets, args.steps, effective_rate(
                    harness, bases[condition], calibration["selected"][condition]))
                summaries = []
                for index, record in enumerate(splits["test"]):
                    result = eval_set(harness, [record], layers, bases[condition], cfg, *controls)
                    result["records"][0]["seed"] = seed
                    summaries.append(result)
                    if (index + 1) % 10 == 0:
                        print(f"{model_name}/{task}/{condition}/s{seed}: {index + 1}/{len(splits['test'])}", flush=True)
                records = [summary["records"][0] for summary in summaries]
                from experiment_metrics import summarize_predictions
                metrics = summarize_predictions(records)
                metrics.update({"records": records,
                                "delta_p2way": float(np.mean([record["delta_p2way"] for record in records])),
                                "control_drop": float(np.mean([summary["control_drop"] for summary in summaries]))})
                result = {"completed": True, "protocol_version": VERSION, "condition": condition,
                          "comparison": comparison, "seed": seed,
                          "split": {name: len(values) for name, values in splits.items()},
                          "test": metrics, "cfg": cfg, "residual_norms": residual_norms,
                          "basis_diagnostics": information, "elapsed_seconds": time.monotonic() - started,
                          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "calibration_sha256": fingerprint(calibration),
                          "guard_note": "optimizer-visible generic agreement; not held-out capability"}
                write_new(output / f"diss_{condition}_s{seed}.json", result)
                print(f"SAVED {task}/{condition}/s{seed}: steer={metrics['steer']:.1%}, same-sign damage={metrics['same_sign_damage']}", flush=True)
            del bases
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
        if args.stage == "all":
            analysis_path = output / "paired_summary.json"
            if not analysis_path.exists():
                write_new(analysis_path, analyze("diss", ["full", "track8", "comp8"], args.seeds, output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    parser.add_argument("--tasks", nargs="+", default=["intermediate", "transfer", "container_swap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--relative-budget", type=float, default=0.3)
    parser.add_argument("--stage", choices=["calibrate", "all"], default="all")
    parser.add_argument("--max-new-cases", type=int, default=None,
                        help="Checkpoint and return after this many new calibration cases")
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.relative_budget <= 1:
        parser.error("Positive steps and a relative budget in (0,1] are required")
    run(args)


if __name__ == "__main__":
    main()