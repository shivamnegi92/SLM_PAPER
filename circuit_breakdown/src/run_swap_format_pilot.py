"""Diagnose response format versus swap reasoning on fresh development prompts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import time

import torch
from transformers import GenerationConfig

from audit_study import TokenOnlyHarness, assert_matches, file_sha256, read_json
from capture_reproducibility import verify_files
from localize import Harness
from run_convergence_pilot import compute_lock, write_tensor_case
from run_validated_study import source_fingerprint
from study_data import fingerprint, load_manifest, write_new
from swap_format_data import build_cases, parse_first_answer


def validate_protocol(protocol):
    required = {"version": "swap_format_pilot_v1", "scope": "fresh_development_format_diagnostic_only",
                "model": "phi-3.5-mini", "chain_lengths": [0, 1, 3, 4], "pairs_per_length": 8,
                "wordings": ["swap", "exchange"], "answer_cues": ["in_box", "is"],
                "sides": ["clean", "counterfactual"], "precision": "float32",
                "generation": {"do_sample": False, "max_new_tokens": 8, "use_cache": True},
                "output": "results/swap_format_pilot_v1"}
    for name, value in required.items():
        assert_matches(protocol[name], value, f"protocol.{name}")
    if protocol["generator_seed"] == protocol["demonstration_seed"]:
        raise ValueError("Demonstrations require a separate generator seed")


def prepare(project, protocol, device):
    manifest = load_manifest(project / protocol["original_manifest"])
    assert_matches(manifest["sha256"], protocol["original_manifest_sha256"], "original_manifest")
    excluded = {pair[name] for seeds in manifest["tasks"].values() for splits in seeds.values()
                for pairs in splits.values() for pair in pairs for name in ("clean_prompt", "corrupt_prompt")}
    design = build_cases(protocol, excluded)
    capture = read_json(project / protocol["provenance_reference"])
    assert_matches(capture["sha256"], fingerprint({key: value for key, value in capture.items() if key != "sha256"}),
                   "provenance_reference")
    assets = capture["models"][protocol["model"]]
    model = project.parent / protocol["model"]
    print("Verifying swap diagnostic model assets before execution", flush=True)
    verify_files(assets, model)
    tokenizer = TokenOnlyHarness(model)
    for case in design["cases"]:
        tokens = tokenizer.encode(case["prompt"])[0].tolist()
        target_id = tokenizer.first_id(case["target"])
        extended = tokenizer.encode(case["prompt"] + " " + case["target"])[0].tolist()
        if extended != tokens + [target_id]:
            raise ValueError("Canonical answer is not one aligned continuation token")
        case["input_ids"] = tokens
        case["input_hash"] = fingerprint(tokens)
        case["target_token_id"] = target_id
    source_names = ("run_swap_format_pilot.py", "swap_format_data.py", "run_convergence_pilot.py",
                    "convergence_pilot.py", "dataset.py", "study_data.py", "localize.py",
                    "intervene_margin.py", "run_validated_study.py", "audit_study.py",
                    "capture_reproducibility.py", "experiment_metrics.py", "intervene_pareto.py")
    comparison = {"protocol": protocol, "protocol_sha256": fingerprint(protocol),
                  "design_sha256": fingerprint(design), "model_assets": assets,
                  "source_sha256": {name: file_sha256(project / "src" / name) for name in source_names},
                  "frozen_core_sha256": source_fingerprint(),
                  "generation_special_ids": {"bos_token_id": tokenizer.tok.bos_token_id,
                                             "eos_token_id": tokenizer.tok.eos_token_id,
                                             "pad_token_id": tokenizer.tok.pad_token_id},
                  "environment": {"device": device, "python": platform.python_version(),
                                  "system": platform.system(), "machine": platform.machine(),
                                  "packages": {name: metadata.version(name) for name in
                                               ("torch", "numpy", "transformers", "tokenizers")}}}
    return comparison, design


def interpret(logits, response, target, target_id):
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("Invalid saved next-token scores")
    parsed = parse_first_answer(response)
    prediction = int(logits.argmax())
    return {"next_token_prediction_id": prediction, "strict_correct": int(prediction == target_id),
            "parsed_answer": parsed, "parsed_correct": int(parsed == target), "unparsed": int(parsed is None),
            "format_only_strict_miss": int(parsed == target and prediction != target_id),
            "target_token_rank": int((logits > logits[target_id]).sum()) + 1,
            "target_token_logit": float(logits[target_id])}


@torch.no_grad()
def measure(harness, case, generation):
    ids = torch.tensor([case["input_ids"]], dtype=torch.long, device=harness.device)
    output = harness.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    generation_config=generation, return_dict_in_generate=True, output_scores=True)
    new_tokens = output.sequences[0, ids.shape[1]:].detach().cpu()
    if not output.scores or len(new_tokens) > generation.max_new_tokens:
        raise ValueError("Generation did not respect the declared bounded response")
    logits = output.scores[0][0].detach().float().cpu()
    response = harness.tok.decode(new_tokens.tolist(), skip_special_tokens=True)
    top_values, top_ids = torch.topk(logits, min(10, len(logits)))
    return {"raw_response": response, "generated_token_ids": new_tokens.tolist(),
            "next_token_logits": logits,
            "top_tokens": [{"id": int(identifier), "text": harness.tok.decode([int(identifier)]), "logit": float(value)}
                           for identifier, value in zip(top_ids, top_values)],
            **interpret(logits, response, case["target"], case["target_token_id"])}


def validate_case(saved, comparison, expected):
    if saved.get("completed") is not True:
        raise ValueError("Incomplete swap-format response")
    assert_matches(saved["comparison_sha256"], fingerprint(comparison), "response.comparison")
    assert_matches(saved["case"], expected, "response.identity")
    response = saved["response"]
    rebuilt = interpret(response["next_token_logits"], response["raw_response"], expected["target"], expected["target_token_id"])
    for name, value in rebuilt.items():
        assert_matches(response[name], value, f"response.{name}")
    if not 1 <= len(response["generated_token_ids"]) <= 8:
        raise ValueError("Missing or excessive generated tokens")


def summarize(responses, comparison):
    if len(responses) != 256 or len({row["case"]["case_id"] for row in responses}) != 256:
        raise ValueError("Swap-format report requires 256 unique declared responses")
    cells = []
    protocol = comparison["protocol"]
    for depth in protocol["chain_lengths"]:
        for wording in protocol["wordings"]:
            for cue in protocol["answer_cues"]:
                rows = [row for row in responses if row["case"]["depth"] == depth
                        and row["case"]["wording"] == wording and row["case"]["answer_cue"] == cue]
                if len(rows) != 16 or len({row["case"]["sample_id"] for row in rows}) != 8:
                    raise ValueError("Incomplete crossed diagnostic cell")
                counts = {name: sum(row["response"][name] for row in rows)
                          for name in ("strict_correct", "parsed_correct", "unparsed", "format_only_strict_miss")}
                cells.append({"depth": depth, "wording": wording, "answer_cue": cue,
                              "n_semantic_pairs": 8, "n_responses": len(rows), **counts,
                              "by_side": {side: {name: sum(row["response"][name] for row in rows if row["case"]["side"] == side)
                                                   for name in counts} for side in protocol["sides"]}})
    return {"completed": True, "version": protocol["version"], "comparison_sha256": fingerprint(comparison),
            "scope": protocol["scope"], "n_semantic_pairs": 32, "n_responses": len(responses), "cells": cells,
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in responses), "limitations": protocol["limitations"],
            "reviewer_gap_status": "G2 remains open; this diagnoses format and depth, not held-out competence or causal control"}


def markdown(summary):
    lines = ["# Swap Format Pilot v1", "",
             "Completed fresh-development, baseline-only diagnosis for Phi. No original test results "
             "were replaced. Strict and parsed metrics answer different questions.", "",
             "| Swaps | Wording | Answer cue | Strict correct /16 | Parsed correct /16 | Format-only strict misses | Unparsed /16 |",
             "|---:|---|---|---:|---:|---:|---:|"]
    for cell in summary["cells"]:
        lines.append(f"| {cell['depth']} | {cell['wording']} | {cell['answer_cue']} | {cell['strict_correct']} | "
                     f"{cell['parsed_correct']} | {cell['format_only_strict_miss']} | {cell['unparsed']} |")
    lines.extend(["", "Each cell reuses eight semantic pairs with two counterfactual sides. "
                  "Rows across wordings and cues are paired, not independent experiments. "
                  "The unchanged demonstration prefix is swap/in_box in every condition. "
                  "No population confidence interval or competence certification is claimed.", "",
                  "The parser accepts a leading uppercase A-H, optionally preceded by `box`. "
                  "Unrecognized response forms remain unparsed, not manually rescued. Raw generated "
                  "text, token IDs, top tokens and full first-step logits are preserved per case.", "",
                  "## Remaining Gap", "", summary["reviewer_gap_status"], ""])
    lines.extend(f"- {value}" for value in summary["limitations"])
    return "\n".join(lines) + "\n"


def run(project, protocol_path, device):
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    convergence = read_json(project / "results/convergence_pilot_v1/summary.json")
    if convergence.get("completed") is not True:
        raise ValueError("Finish the first convergence diagnosis before this model pilot")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    output = project / protocol["output"]
    with compute_lock(project / "results/revision_compute"):
        comparison, design = prepare(project, protocol, device)
        run_path = output / "run.json"
        if run_path.exists():
            assert_matches(read_json(run_path)["comparison"], comparison, "resume.comparison")
            assert_matches(read_json(output / "design.json"), design, "resume.design")
        else:
            write_new(output / "design.json", design)
            write_new(run_path, {"comparison": comparison, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        responses, pending = [], []
        for case in design["cases"]:
            path = output / "cases" / f"{case['case_id']}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                validate_case(saved, comparison, case)
                responses.append(saved)
            else:
                pending.append(case)
        if pending:
            harness = Harness(str(project.parent / protocol["model"]), device)
            harness.model.requires_grad_(False)
            generation = GenerationConfig(**protocol["generation"], **comparison["generation_special_ids"], num_beams=1)
            for case in pending:
                started = time.monotonic()
                response = measure(harness, case, generation)
                saved = {"completed": True, "comparison_sha256": fingerprint(comparison),
                         "case": case, "response": response, "elapsed_seconds": time.monotonic() - started}
                validate_case(saved, comparison, case)
                write_tensor_case(output / "cases" / f"{case['case_id']}.pt", saved)
                responses.append(saved)
                if len(responses) % 16 == 0:
                    print(f"SAVED swap-format responses {len(responses)}/256", flush=True)
                    if device == "mps":
                        torch.mps.empty_cache()
            del harness
        for name, digest in comparison["source_sha256"].items():
            assert_matches(file_sha256(project / "src" / name), digest, "postrun.source")
        responses.sort(key=lambda row: row["case"]["case_id"])
        summary = summarize(responses, comparison)
        if (output / "summary.json").exists():
            assert_matches(read_json(output / "summary.json"), summary, "resume.summary")
        else:
            write_new(output / "summary.json", summary)
        report = markdown(summary)
        if (output / "RESULTS.md").exists():
            assert_matches((output / "RESULTS.md").read_text(), report, "resume.report")
        else:
            with (output / "RESULTS.md").open("x") as stream:
                stream.write(report)
        print("Swap-format pilot complete: 256 responses; G2 generalization claim remains open", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocols/swap_format_pilot_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), pick_device(args.device))


if __name__ == "__main__":
    main()