"""Experiment 1: are successful activation edits UNIQUE?

Question: if two edits produce the same behavior, must they be internally
similar? Prior observation: cos(v_full, v_basis) ~ 0.08-0.15 between
independently found solutions, while KL between their output distributions
FALLS as rank grows -- geometric divergence with behavioral convergence.

Method: optimize the SAME example to the SAME target many times, varying only
the initialization, and measure pairwise geometry among the SUCCESSFUL edits.

THE CONTROL THAT DECIDES THIS. The optimizer here is otherwise fully
deterministic (zero init, Adam, eval mode), so variation must be introduced
via random init. But random high-dimensional vectors are nearly orthogonal by
construction, so low pairwise cosine among final vectors is NOT by itself
evidence of diverse solutions -- it may just mean the optimizer barely moved
from a scattered set of starting points. Two diagnostics separate these:

  travel_ratio  = ||v_final - v_init|| / ||v_init||
                  how far the optimizer actually moved. Near 0 => the result
                  is inherited from the init and proves nothing.
  cos_to_init   = cos(v_final, v_init)
                  near 1 => same story.

Only if edits travel substantially AND still end up mutually dissimilar does
non-uniqueness hold. A zero-init run is included as the reference solution
(the canonical v_full), and every random-init solution is compared to it.

Behavioral equivalence is verified, not assumed: success, final margin, and
pairwise symmetric KL between output distributions are reported, so
"same behavior, different vector" is a measured claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from localize import Harness, pick_device
import dataset
from intervene_margin import get_control_baseline
from intervene_pareto import forward_logits, optimize_sample_converged
from run_convergence_pilot import pilot_records
from study_data import load_manifest

LAYERS = [18, 20, 22, 24]


def flat(vectors):
    return torch.cat([v.reshape(-1).float() for v in vectors])


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def symmetric_kl(p_logits, q_logits):
    p = F.log_softmax(p_logits.float(), dim=0)
    q = F.log_softmax(q_logits.float(), dim=0)
    p_prob, q_prob = p.exp(), q.exp()
    return float(((p_prob * (p - q)).sum() + (q_prob * (q - p)).sum()).item() / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama-3.2-3b")
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    harness = Harness(str(project.parent / args.model), pick_device(args.device))
    harness.model.requires_grad_(False)
    dataset.restrict_to_single_token(harness.tok)

    manifest = load_manifest(project / "data/validated_manifest_v1.json")
    protocol = {"task": "transfer", "development_seed": 1,
                "development_indices": list(range(8))}
    train, _ = pilot_records(harness, manifest, protocol)
    record = train[args.example]

    # control_degradation expects a LIST of control prompts and their baseline
    # predictions; use the project's standard helper rather than improvising a
    # single tensor (doing so iterates over token rows and blows up in rotary).
    control_ids, control_base = get_control_baseline(harness)

    base_cfg = {"lr": 0.05, "l2": 1e-4, "lam_kl": 0.0, "guard_every": 4,
                "max_control_drop": 0.15, "norm_budget": 12.0,
                "convergence_eps": 1e-4, "convergence_patience": 5, "max_steps": 120,
                "objective": "cross_entropy"}

    print(f"model={args.model} layers={LAYERS} restarts={args.restarts} "
          f"init_scale={args.init_scale}")
    print(f"example={args.example} target={harness.tok.decode([record.cid])!r}\n")

    solutions = []
    for index in range(args.restarts + 1):
        cfg = dict(base_cfg)
        if index == 0:
            cfg["init_seed"], cfg["init_scale"] = None, 0.0  # canonical zero-init
            label = "zero-init (reference)"
        else:
            cfg["init_seed"], cfg["init_scale"] = 1000 + index, args.init_scale
            label = f"random-init seed {1000 + index}"

        edit, diagnostics = optimize_sample_converged(
            harness, record, LAYERS, control_ids, control_base, None, cfg)
        vectors = edit.detached()
        with torch.no_grad():
            logits = forward_logits(harness, record.corrupt_ids, LAYERS, vectors, record.dpos)
        success = int(logits.argmax().item() == record.cid)

        final = flat(vectors)
        start = flat(edit.init_vectors)
        travel = float((final - start).norm())
        travel_ratio = travel / (float(start.norm()) + 1e-12) if index > 0 else float("inf")

        solutions.append({
            "label": label, "success": success, "vector": final, "logits": logits.detach(),
            "norm": float(final.norm()), "travel": travel, "travel_ratio": travel_ratio,
            "cos_to_init": cosine(final, start) if index > 0 else 0.0,
            "steps": diagnostics.get("steps_taken"), "converged": diagnostics.get("converged"),
        })
        print(f"  {label:>26}: success={success} norm={float(final.norm()):7.3f} "
              f"travel={travel:7.3f} travel_ratio={travel_ratio:8.2f} "
              f"cos_to_init={solutions[-1]['cos_to_init']:+.3f} steps={diagnostics.get('steps_taken')}",
              flush=True)

    successful = [s for s in solutions if s["success"]]
    print(f"\n=== {len(successful)}/{len(solutions)} restarts succeeded ===")
    if len(successful) < 2:
        print("Too few successful solutions to compare; nothing to conclude.")
        return

    random_successful = [s for s in successful if s["label"].startswith("random")]
    if random_successful:
        ratios = [s["travel_ratio"] for s in random_successful]
        cos_inits = [s["cos_to_init"] for s in random_successful]
        print(f"\n--- CONTROL: did the optimizer actually move? ---")
        print(f"  travel_ratio ||v_final - v_init|| / ||v_init||: "
              f"median {np.median(ratios):.2f}, min {min(ratios):.2f}, max {max(ratios):.2f}")
        print(f"  cos(v_final, v_init): median {np.median(cos_inits):+.3f}, "
              f"max {max(cos_inits):+.3f}")
        if np.median(ratios) < 1.0 or np.median(cos_inits) > 0.5:
            print("  WARNING: solutions stayed near their initializations. Low pairwise\n"
                  "  cosine below would then reflect scattered STARTS, not distinct SOLUTIONS.")
        else:
            print("  Solutions moved far from their starting points, so pairwise geometry\n"
                  "  below reflects where the optimizer went, not where it began.")

    cosines, kls = [], []
    for i in range(len(successful)):
        for j in range(i + 1, len(successful)):
            cosines.append(cosine(successful[i]["vector"], successful[j]["vector"]))
            kls.append(symmetric_kl(successful[i]["logits"], successful[j]["logits"]))

    print(f"\n--- pairwise geometry among {len(successful)} SUCCESSFUL edits "
          f"({len(cosines)} pairs) ---")
    print(f"  cosine:        median {np.median(cosines):+.3f}  "
          f"min {min(cosines):+.3f}  max {max(cosines):+.3f}")
    print(f"  symmetric KL:  median {np.median(kls):.4f}  "
          f"min {min(kls):.4f}  max {max(kls):.4f}")

    reference = successful[0]
    if reference["label"].startswith("zero"):
        to_reference = [cosine(reference["vector"], s["vector"]) for s in successful[1:]]
        print(f"  cosine to the canonical zero-init solution: "
              f"median {np.median(to_reference):+.3f}, max {max(to_reference):+.3f}")

    print("\n--- verdict ---")
    median_cos = float(np.median(cosines))
    median_kl = float(np.median(kls))
    # Thresholds are stated explicitly and checked in order of what they rule
    # out. An earlier version tested >0.8 first and otherwise fell through to a
    # "dissimilar" branch, which printed DISSIMILAR for a median cosine of
    # 0.788 -- a conclusion contradicting its own number. Cosine similarity is
    # the primary axis; behavioral closeness only qualifies it.
    if median_cos >= 0.6:
        print(f"  SIMILAR (median cosine {median_cos:+.3f}). Independently initialized")
        print("  runs converge to substantially the SAME direction, so successful edits")
        print("  look close to unique on this example. This does NOT support the")
        print("  non-uniqueness story; do not run the interpolation test on this basis.")
    elif median_cos <= 0.3 and median_kl < 0.1:
        print(f"  DISSIMILAR yet behaviorally CLOSE (median cosine {median_cos:+.3f},")
        print(f"  median symmetric KL {median_kl:.4f}). Consistent with non-uniqueness;")
        print("  the interpolation test is now worth running.")
    elif median_cos <= 0.3:
        print(f"  Solutions differ geometrically (median cosine {median_cos:+.3f}) AND")
        print(f"  behaviorally (median symmetric KL {median_kl:.4f}). They are not")
        print("  equivalent solutions, so this does not demonstrate non-uniqueness.")
    else:
        print(f"  AMBIGUOUS: median cosine {median_cos:+.3f} sits between the similar")
        print(f"  (>=0.6) and dissimilar (<=0.3) thresholds; median KL {median_kl:.4f}.")
        print("  Report the distribution rather than a binary verdict, and note the")
        print(f"  spread: min {min(cosines):+.3f}, max {max(cosines):+.3f}.")

    if args.out:
        payload = {
            "model": args.model, "layers": LAYERS, "restarts": args.restarts,
            "init_scale": args.init_scale, "n_successful": len(successful),
            "pairwise_cosine": cosines, "pairwise_symmetric_kl": kls,
            "travel_ratio": [s["travel_ratio"] for s in random_successful],
            "cos_to_init": [s["cos_to_init"] for s in random_successful],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as stream:
            json.dump(payload, stream, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
