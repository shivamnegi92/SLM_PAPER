"""Run predictive-vs-causal subspace diagnostics, v2.

Changes from the v1 Phi pilot, per reviewer follow-up on v1's results:

  - Model: llama-3.2-3b (v1 used phi-3.5-mini; Llama has behaved differently
    from Phi in prior dissociation experiments, so it is the priority
    replication target, not a third model).
  - Ranks: [8, 16, 32, 64] only. v1 also tested 1, 2, 4, but Phi showed the
    interesting transition occurs within [8, 32]; low ranks are dropped here
    to spend the compute budget where the signal is.
  - CONVERGENCE-BASED STOPPING instead of a fixed 32-step budget. v1's
    tracking-rank curve (2/8 -> 5/8 -> 6/8 -> 3/8 across k=8,16,32,64) is a
    known optimizer-confound warning sign: more capacity should not reduce
    success under a fair comparison. Every optimize call here runs until
    |delta loss| < eps for `convergence_patience` consecutive steps or a
    generous max_steps cap, so ranks are compared at equal EFFORT, not equal
    step count.
  - "causal" renamed to "gradient_coordinate" (raw-coordinate gradient
    sensitivity is a narrow, likely-too-weak comparator for causal control;
    see `predictive_vs_causal_v2.py` docstring).
  - ADDED "learned_causal": a DAS-lite basis, a shared per-layer orthonormal
    rotation fit directly against the behavioral objective (see
    `predictive_vs_causal_v2.learned_causal_basis`).
  - For EVERY basis, both experiments are retained:
      (a) projection of the independently-optimized full-space solution
      (b) an independently, freshly optimized solution within the basis
    These test different questions ("does the basis CONTAIN the known
    solution" vs "does the basis PERMIT a different solution to be learned")
    and must not be conflated.
  - For every example where full AND fresh-in-basis both succeed, degeneracy
    diagnostics are computed: cosine(v_full, v_basis), ||v_basis||/||v_full||,
    and KL(p_full || p_basis). A low cosine similarity with a low KL would be
    direct evidence of intervention degeneracy: multiple geometrically
    distinct edits producing functionally similar behavior.

Development-only diagnostic; not a confirmatory test. Same tier of rigor as
`run_predictive_vs_causal_subspace.py` (v1): no compute-lock/provenance-hash
ceremony from the frozen G1/G2 pipeline. Do not cite this as a declared,
confirmatory result without untouched-example follow-up.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from audit_study import assert_matches, read_json
from experiment_metrics import paired_delta_ci, rate_ci
from intervene_margin import CONTROL_PROMPTS
from intervene_pareto import fit_subspace_bases, forward_logits, optimize_sample_converged
from localize import Harness, pick_device
from predictive_vs_causal_subspace import (
    coordinate_basis_from_scores,
    energy_fraction,
    project_vectors,
)
from predictive_vs_causal_v2 import (
    degeneracy_diagnostics,
    gradient_coordinate_scores,
    learned_causal_basis,
)
from run_convergence_pilot import input_identity, pilot_records
from run_predictive_vs_causal_subspace import evaluate, pca_bases, random_bases, state_matrices
from run_validated_study import control_set, effective_rate, layer_budgets
from study_data import fingerprint, load_manifest, write_new

BASIS_NAMES = ["tracking", "pca", "random", "gradient_coordinate", "learned_causal"]


def gradient_coordinate_bases(scores, rank, device):
    bases, indices = [], []
    for score in scores:
        basis, selected = coordinate_basis_from_scores(score, rank)
        bases.append(basis.to(device))
        indices.append(selected)
    return bases, indices


def fit_all_bases(harness, train, layers, rank, clean_train, corrupt_train, gradient_scores,
                   learned_causal_cfg, device, seed):
    tracking = fit_subspace_bases(harness, train, layers, rank)
    pca, pca_info = pca_bases(clean_train, corrupt_train, rank, device)
    random = random_bases(harness.model.config.hidden_size, layers, rank, seed + rank, device)
    gradient_coordinate, gc_indices = gradient_coordinate_bases(gradient_scores, rank, device)
    learned_causal, lc_diagnostics = learned_causal_basis(
        harness, train[:learned_causal_cfg["n_examples"]], layers, rank, learned_causal_cfg)
    bases = {"tracking": tracking, "pca": pca, "random": random,
             "gradient_coordinate": gradient_coordinate, "learned_causal": learned_causal}
    metadata = {"pca": pca_info, "gradient_coordinate_indices": gc_indices,
                "learned_causal": lc_diagnostics}
    return bases, metadata


def run_example(harness, record, layers, controls, predictions, bases, base_cfg, effective_lr_fn):
    vectors_full, diag_full = optimize_sample_converged(
        harness, record, layers, controls, predictions, None, base_cfg)
    full_vectors = vectors_full.detached()
    full_eval = evaluate(harness, record, layers, full_vectors)
    baseline = evaluate(harness, record, layers, None)

    conditions = {"full": {**full_eval, "energy_fraction": 1.0, "optimizer": diag_full}}
    for name, basis in bases.items():
        projected = project_vectors(full_vectors, basis)
        projected_eval = evaluate(harness, record, layers, projected)
        conditions[f"{name}_projected"] = {
            **projected_eval, "energy_fraction": energy_fraction(projected, full_vectors)}

        rate = effective_lr_fn(basis)
        fresh_cfg = {**base_cfg, "lr": rate}
        edit_fresh, diag_fresh = optimize_sample_converged(
            harness, record, layers, controls, predictions, basis, fresh_cfg)
        fresh_vectors = edit_fresh.detached()
        fresh_eval = evaluate(harness, record, layers, fresh_vectors)
        entry = {**fresh_eval, "energy_fraction": energy_fraction(fresh_vectors, full_vectors),
                  "optimizer": diag_fresh}
        if full_eval["target_success"] and fresh_eval["target_success"]:
            entry["degeneracy"] = degeneracy_diagnostics(harness, layers, full_vectors, fresh_vectors, record)
        conditions[f"{name}_fresh"] = entry

    return {"baseline": baseline, "full": full_eval, "conditions": conditions}


def summarize_rank(rows, rank):
    baseline_probabilities = [row["baseline"]["target_probability"] for row in rows]
    full_probabilities = [row["full"]["target_probability"] for row in rows]
    full_delta = float(np.mean(full_probabilities) - np.mean(baseline_probabilities))
    summary = {}
    degeneracy_pool = {name: [] for name in BASIS_NAMES}
    for key in rows[0]["conditions"]:
        flags = [row["conditions"][key]["target_success"] for row in rows]
        probabilities = [row["conditions"][key]["target_probability"] for row in rows]
        rate, interval = rate_ci(flags)
        probability_delta = float(np.mean(probabilities) - np.mean(baseline_probabilities))
        recovery = probability_delta / full_delta if abs(full_delta) > 1e-9 else None
        energies = [row["conditions"][key].get("energy_fraction", np.nan) for row in rows]
        summary[key] = {
            "rank": rank, "n": len(rows), "successes": int(sum(flags)), "success_rate": rate,
            "success_ci": interval, "mean_target_probability": float(np.mean(probabilities)),
            "probability_recovery_fraction_vs_full": recovery,
            "mean_energy_fraction": float(np.mean(energies)),
        }
        for basis_name in BASIS_NAMES:
            if key == f"{basis_name}_fresh":
                degeneracies = [row["conditions"][key]["degeneracy"] for row in rows
                                if "degeneracy" in row["conditions"][key]]
                degeneracy_pool[basis_name] = degeneracies
    degeneracy_summary = {}
    for basis_name, degeneracies in degeneracy_pool.items():
        if not degeneracies:
            degeneracy_summary[basis_name] = {"n_double_success": 0}
            continue
        cosines = [d["cosine_full_vs_basis"] for d in degeneracies if d["cosine_full_vs_basis"] is not None]
        kls = [d["kl_full_intervention_vs_basis_intervention"] for d in degeneracies]
        norm_ratios = [d["basis_norm_over_full_norm"] for d in degeneracies if d["basis_norm_over_full_norm"] is not None]
        degeneracy_summary[basis_name] = {
            "n_double_success": len(degeneracies),
            "mean_cosine_full_vs_basis": float(np.mean(cosines)) if cosines else None,
            "mean_kl_full_vs_basis": float(np.mean(kls)) if kls else None,
            "mean_basis_norm_over_full_norm": float(np.mean(norm_ratios)) if norm_ratios else None,
        }
    return summary, degeneracy_summary


def markdown(result):
    lines = ["# Predictive vs Causal Subspace Pilot v2", "",
             "Development-only diagnostic; not a confirmatory test.", "",
             "## Projection vs fresh optimization, by basis and rank", "",
             "| Condition | Successes | Energy | Recovery vs full |",
             "|---|---:|---:|---:|"]
    for rank, rank_summary in result["summary_by_rank"].items():
        for condition, values in rank_summary["conditions"].items():
            recovery = values["probability_recovery_fraction_vs_full"]
            lines.append("| k={rank} {condition} | {successes}/{n} | {energy:.1f}% | {recovery} |".format(
                rank=rank, condition=condition, successes=values["successes"], n=values["n"],
                energy=100 * values["mean_energy_fraction"],
                recovery="n/a" if recovery is None else f"{100 * recovery:.1f}%"))
    lines.extend(["", "## Degeneracy diagnostics (double-success cases only)", "",
                  "| Rank | Basis | n double-success | mean cosine(full, basis) | mean KL(full \\|\\| basis) | mean \\|\\|basis\\|\\|/\\|\\|full\\|\\| |",
                  "|---|---|---:|---:|---:|---:|"])
    for rank, rank_summary in result["summary_by_rank"].items():
        for basis_name, values in rank_summary["degeneracy"].items():
            n = values["n_double_success"]
            if n == 0:
                lines.append(f"| {rank} | {basis_name} | 0 | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {rank} | {basis_name} | {n} | "
                    f"{values['mean_cosine_full_vs_basis']:.3f} | "
                    f"{values['mean_kl_full_vs_basis']:.4f} | "
                    f"{values['mean_basis_norm_over_full_norm']:.3f} |")
    lines.extend(["", "## Guardrails", "",
                  "- This is a development-only diagnostic until confirmed on untouched examples.",
                  "- Convergence-based stopping does not prove a converged global optimum was found.",
                  "- A low cosine similarity combined with a low KL divergence indicates intervention "
                  "degeneracy (multiple geometrically distinct edits with similar behavioral effect), "
                  "not that the subspace contains the original solution.",
                  "- gradient_coordinate is a narrow raw-coordinate-sensitivity baseline, not a general "
                  "causal-subspace baseline; learned_causal is the stronger comparator.", ""])
    return "\n".join(lines)


def run(project, protocol_path, device):
    started = time.monotonic()
    protocol = read_json(protocol_path)
    manifest = load_manifest(project / protocol["manifest"])
    assert_matches(manifest["sha256"], protocol["manifest_sha256"], "manifest")
    output = project / protocol["output"]
    output.mkdir(parents=True, exist_ok=True)

    harness = Harness(str(project.parent / protocol["model"]), pick_device(device))
    harness.model.requires_grad_(False)

    train, development = pilot_records(harness, manifest, protocol)
    layers = protocol["layers"]
    controls, predictions = control_set(harness, CONTROL_PROMPTS)
    budgets, residual_norms = layer_budgets(harness, train, layers, protocol["relative_budget"])

    base_cfg = {
        "lr": protocol["base_learning_rate"], "l2": 1e-3, "lam_kl": 0.0,
        "norm_budget": budgets, "max_control_drop": 0.2, "guard_every": 4,
        "objective": "cross_entropy",
        "convergence_eps": protocol["convergence_eps"],
        "convergence_patience": protocol["convergence_patience"],
        "max_steps": protocol["max_steps"],
    }
    learned_causal_cfg = {
        "lr": protocol["learned_causal_lr"], "seed": protocol["training_seed"],
        "n_examples": protocol["learned_causal_examples"],
        "convergence_eps": protocol["learned_causal_convergence_eps"],
        "convergence_patience": protocol["learned_causal_convergence_patience"],
        "max_steps": protocol["learned_causal_max_steps"],
    }

    clean_train = state_matrices(harness, train, layers, True)
    corrupt_train = state_matrices(harness, train, layers, False)
    gradient_scores = gradient_coordinate_scores(
        harness, train[:min(protocol["gradient_coordinate_examples"], len(train))], layers)

    def effective_lr_fn(basis):
        return effective_rate(harness, basis, protocol["base_learning_rate"])

    rows_by_rank = {}
    basis_metadata = {}
    for rank in protocol["ranks"]:
        print(f"=== rank {rank}: fitting bases ===", flush=True)
        bases, metadata = fit_all_bases(
            harness, train, layers, rank, clean_train, corrupt_train, gradient_scores,
            learned_causal_cfg, harness.device, protocol["training_seed"])
        basis_metadata[str(rank)] = metadata

        rank_rows = []
        for index, record in enumerate(development):
            row = run_example(harness, record, layers, controls, predictions, bases, base_cfg, effective_lr_fn)
            row.update({"rank": rank, "sample_id": record.sample_id, "input": input_identity(record)})
            rank_rows.append(row)
            successes = ", ".join(f"{name}={row['conditions'][f'{name}_fresh']['target_success']}"
                                   for name in BASIS_NAMES)
            print(f"rank {rank} example {index + 1}/{len(development)}: "
                  f"full={row['full']['target_success']} {successes}", flush=True)
        rows_by_rank[str(rank)] = rank_rows

    summary_by_rank = {}
    for rank_key, rows in rows_by_rank.items():
        conditions_summary, degeneracy_summary = summarize_rank(rows, int(rank_key))
        summary_by_rank[rank_key] = {"conditions": conditions_summary, "degeneracy": degeneracy_summary}

    n_rows = sum(len(rows) for rows in rows_by_rank.values())
    result = {"completed": True, "protocol": protocol, "protocol_sha256": fingerprint(protocol),
              "device": harness.device, "layers": layers, "residual_norms": residual_norms,
              "n_rows": n_rows, "basis_metadata": basis_metadata,
              "rows_by_rank": rows_by_rank, "summary_by_rank": summary_by_rank,
              "elapsed_seconds": time.monotonic() - started,
              "reviewer_gap_status": "G1 remains open until untouched confirmation; this pilot "
                                     "adds convergence-based stopping and a learned-causal comparator"}
    write_new(output / "summary.json", result)
    with (output / "RESULTS.md").open("x") as stream:
        stream.write(markdown(result))
    print(f"Saved predictive-vs-causal v2 pilot -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).resolve().parents[1] / "protocols/predictive_vs_causal_subspace_v2.json")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), args.device)


if __name__ == "__main__":
    main()
