"""Run the fixed fresh-question context diagnostic, without activation edits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import time

import torch
from transformers import GenerationConfig

from audit_study import TokenOnlyHarness, assert_matches, file_sha256, read_json
from baseline_context_data import build_design
from capture_reproducibility import verify_files
from localize import Harness
from run_convergence_pilot import compute_lock, write_tensor_case
from run_swap_format_pilot import measure, validate_case
from run_validated_study import source_fingerprint
from study_data import fingerprint, load_manifest, write_new


def validate_protocol(protocol):
    required = {"version": "baseline_context_pilot_v1", "scope": "fresh_development_context_control_only",
                "model": "phi-3.5-mini", "generator_seed": 9072701,
                "demonstration_seeds": {"fresh_a": 9072702, "fresh_b": 9072703},
                "chain_lengths": [0, 1], "pairs_per_length": 8,
                "contexts": ["none", "prior", "fresh_a", "fresh_b"], "wording": "swap", "answer_cue": "in_box",
                "query_objects": ["key", "book", "ball", "coin", "ring", "map", "cup", "pen"],
                "sides": ["clean", "counterfactual"], "precision": "float32",
                "generation": {"do_sample": False, "max_new_tokens": 8, "use_cache": True},
                "stopping_rule": "complete_all_128_responses_no_data_dependent_extension",
                "output": "results/baseline_context_pilot_v1"}
    for name, value in required.items():
        assert_matches(protocol[name], value, f"protocol.{name}")


def prepare(project, protocol, device):
    manifest = load_manifest(project / protocol["original_manifest"])
    assert_matches(manifest["sha256"], protocol["original_manifest_sha256"], "manifest")
    previous = read_json(project / protocol["previous_design"])
    previous_run = read_json(project / protocol["previous_run"])
    assert_matches(fingerprint(previous), previous_run["comparison"]["design_sha256"], "previous_design")
    excluded = {pair[name] for seeds in manifest["tasks"].values() for splits in seeds.values()
                for pairs in splits.values() for pair in pairs for name in ("clean_prompt", "corrupt_prompt")}
    design = build_design(protocol, previous, excluded)
    capture = read_json(project / protocol["provenance_reference"])
    assert_matches(capture["sha256"], fingerprint({key: value for key, value in capture.items() if key != "sha256"}), "capture")
    assets = capture["models"][protocol["model"]]
    model = project.parent / protocol["model"]
    print("Verifying baseline-context model contents before execution", flush=True)
    verify_files(assets, model)
    tokenizer = TokenOnlyHarness(model)
    for case in design["cases"]:
        ids = tokenizer.encode(case["prompt"])[0].tolist()
        target_id = tokenizer.first_id(case["target"])
        if tokenizer.encode(case["prompt"] + " " + case["target"])[0].tolist() != ids + [target_id]:
            raise ValueError("Context changed the canonical answer token boundary")
        case.update(input_ids=ids, input_hash=fingerprint(ids), target_token_id=target_id)
    names = ("run_baseline_context_pilot.py", "baseline_context_data.py", "run_swap_format_pilot.py",
             "swap_format_data.py", "run_convergence_pilot.py", "convergence_pilot.py", "dataset.py",
             "study_data.py", "localize.py", "intervene_margin.py", "intervene_pareto.py",
             "run_validated_study.py", "experiment_metrics.py", "audit_study.py", "capture_reproducibility.py")
    comparison = {"protocol": protocol, "protocol_sha256": fingerprint(protocol), "design_sha256": fingerprint(design),
                  "previous_design_sha256": fingerprint(previous), "model_assets": assets,
                  "source_sha256": {name: file_sha256(project / "src" / name) for name in names},
                  "frozen_core_sha256": source_fingerprint(),
                  "special_ids": {"bos_token_id": tokenizer.tok.bos_token_id, "eos_token_id": tokenizer.tok.eos_token_id,
                                  "pad_token_id": tokenizer.tok.pad_token_id},
                  "environment": {"device": device, "python": platform.python_version(), "system": platform.system(),
                                  "machine": platform.machine(), "packages": {name: metadata.version(name) for name in
                                                                              ("torch", "numpy", "transformers", "tokenizers")}}}
    return comparison, design


def summarize(responses, comparison):
    if len(responses) != 128 or len({row["case"]["case_id"] for row in responses}) != 128:
        raise ValueError("The context diagnostic requires 128 distinct declared responses")
    protocol = comparison["protocol"]
    cells, contrasts = [], []
    metrics = ("strict_correct", "parsed_correct", "unparsed", "format_only_strict_miss")
    for depth in protocol["chain_lengths"]:
        grouped = {}
        for context in protocol["contexts"]:
            rows = [row for row in responses if row["case"]["depth"] == depth and row["case"]["context"] == context]
            ordered = sorted(rows, key=lambda row: (row["case"]["sample_id"], row["case"]["side"]))
            identities = [(row["case"]["sample_id"], row["case"]["side"], row["case"]["target"], row["case"]["query"]) for row in ordered]
            if len(ordered) != 16 or len(set(identities)) != 16 or len({item[0] for item in identities}) != 8:
                raise ValueError("Invalid context cell pairing")
            if grouped:
                reference = grouped["none"]
                expected = [(row["case"]["sample_id"], row["case"]["side"], row["case"]["target"], row["case"]["query"]) for row in reference]
                assert_matches(identities, expected, "context.pairing")
            grouped[context] = ordered
            counts = {name: sum(row["response"][name] for row in rows) for name in metrics}
            answers = Counter(row["response"]["parsed_answer"] or "unparsed" for row in rows)
            cells.append({"depth": depth, "context": context, "n_semantic_pairs": 8, "n_responses": 16, **counts,
                          "answer_counts": dict(sorted(answers.items())),
                          "by_side": {side: {name: sum(row["response"][name] for row in rows if row["case"]["side"] == side)
                                               for name in metrics} for side in protocol["sides"]}})
        for context in protocol["contexts"][1:]:
            for metric in ("strict_correct", "parsed_correct"):
                first, second = grouped[context], grouped["none"]
                gains = sum(left["response"][metric] > right["response"][metric] for left, right in zip(first, second))
                losses = sum(left["response"][metric] < right["response"][metric] for left, right in zip(first, second))
                contrasts.append({"depth": depth, "context": context, "reference": "none", "metric": metric,
                                  "gains": gains, "losses": losses, "rate_difference": (gains - losses) / 16})
    return {"completed": True, "protocol": protocol["version"], "scope": protocol["scope"],
            "comparison_sha256": fingerprint(comparison), "n_semantic_pairs": 16, "n_responses": 128,
            "cells": cells, "contrasts": contrasts, "elapsed_seconds": sum(row["elapsed_seconds"] for row in responses),
            "limitations": protocol["limitations"],
            "reviewer_gap_status": "G2 remains open: no multistep competence or held-out causal-control claim"}


def markdown(summary):
    lines = ["# Baseline Context Pilot v1", "",
             "Fresh development-only questions with fixed swap wording and in_box cue. "
             "Only the demonstration prefix changes within each paired question. No activation edits are used.", "",
             "| Swaps | Context | Strict /16 | Parsed /16 | Unparsed /16 | Parsed answer counts |",
             "|---:|---|---:|---:|---:|---|"]
    for cell in summary["cells"]:
        distribution = ", ".join(f"{name}:{count}" for name, count in cell["answer_counts"].items())
        lines.append(f"| {cell['depth']} | {cell['context']} | {cell['strict_correct']} | {cell['parsed_correct']} | "
                     f"{cell['unparsed']} | {distribution} |")
    lines.extend(["", "Each cell has eight semantic pairs and two sides; the four contexts reuse those "
                  "same questions. Counts and paired gains/losses are descriptive, not 128 independent "
                  "examples or a population competence bound.", "",
                  "| Swaps | Context minus none | Metric | Gained | Lost | Difference, pp |",
                  "|---:|---|---|---:|---:|---:|"])
    for contrast in summary["contrasts"]:
        lines.append(f"| {contrast['depth']} | {contrast['context']} | {contrast['metric']} | {contrast['gains']} | "
                     f"{contrast['losses']} | {contrast['rate_difference'] * 100:.1f} |")
    lines.extend(["", "The prior prefix is copied exactly from the first format diagnostic. Fresh prefixes "
                  "use independently generated examples with the same depths, target labels and ordering. "
                  "None removes all demonstrations, which changes length as well as content. All prompts, "
                  "raw responses, token IDs and next-token scores are saved. Parser rules are unchanged.", "",
                  "## Remaining Gap", "", summary["reviewer_gap_status"], ""])
    lines.extend(f"- {value}" for value in summary["limitations"])
    return "\n".join(lines) + "\n"


def load_responses(output, comparison, design, require_complete=False):
    responses = []
    for case in design["cases"]:
        path = output / "cases" / f"{case['case_id']}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            validate_case(saved, comparison, case)
            assert_matches(fingerprint(case["input_ids"]), case["input_hash"], "input_hash")
            assert_matches(saved["response"]["generated_token_ids"][0], saved["response"]["next_token_prediction_id"], "first_generated_token")
            responses.append(saved)
        elif require_complete:
            raise ValueError(f"Missing baseline-context response: {case['case_id']}")
    return responses


def verify_outputs(project, output):
    comparison = read_json(output / "run.json")["comparison"]
    design = read_json(output / "design.json")
    assert_matches(fingerprint(design), comparison["design_sha256"], "design")
    for name, digest in comparison["source_sha256"].items():
        assert_matches(file_sha256(project / "src" / name), digest, f"source.{name}")
    responses = load_responses(output, comparison, design, True)
    tokenizer = TokenOnlyHarness(project.parent / comparison["protocol"]["model"])
    for saved in responses:
        case, response = saved["case"], saved["response"]
        assert_matches(tokenizer.encode(case["prompt"])[0].tolist(), case["input_ids"], "tokenized_input")
        assert_matches(tokenizer.tok.decode(response["generated_token_ids"], skip_special_tokens=True), response["raw_response"], "decoded_answer")
    report = summarize(responses, comparison)
    assert_matches(read_json(output / "summary.json"), report, "summary")
    assert_matches((output / "RESULTS.md").read_text(), markdown(report), "report")
    print("Verified all 128 context responses, paired summaries, token identities and source hashes", flush=True)


def run(project, protocol_path, device, verify_only=False):
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    output = project / protocol["output"]
    if verify_only:
        assert_matches(read_json(output / "run.json")["comparison"]["protocol"], protocol, "saved_protocol")
        verify_outputs(project, output)
        return
    if not read_json(project / "results/guard_rate_pilot_v1/summary.json").get("completed"):
        raise ValueError("Complete the guard/rate diagnostic before this model run")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    with compute_lock(project / "results/revision_compute"):
        comparison, design = prepare(project, protocol, device)
        for name, value in (("design.json", design), ("run.json", {"comparison": comparison})):
            path = output / name
            if path.exists():
                actual = read_json(path)
                if name == "run.json":
                    assert_matches(actual["comparison"], comparison, "resume.comparison")
                else:
                    assert_matches(actual, value, "resume.design")
            else:
                if name == "run.json":
                    value = {**value, "created_at_utc": datetime.now(timezone.utc).isoformat()}
                write_new(path, value)
        responses = load_responses(output, comparison, design)
        completed = {row["case"]["case_id"] for row in responses}
        pending = [case for case in design["cases"] if case["case_id"] not in completed]
        if pending:
            harness = Harness(str(project.parent / protocol["model"]), device)
            harness.model.requires_grad_(False)
            generation = GenerationConfig(**protocol["generation"], **comparison["special_ids"], num_beams=1)
            for case in pending:
                started = time.monotonic()
                response = measure(harness, case, generation)
                saved = {"completed": True, "comparison_sha256": fingerprint(comparison), "case": case,
                         "response": response, "elapsed_seconds": time.monotonic() - started}
                validate_case(saved, comparison, case)
                write_tensor_case(output / "cases" / f"{case['case_id']}.pt", saved)
                responses.append(saved)
                if len(responses) % 16 == 0:
                    print(f"SAVED baseline-context responses {len(responses)}/128", flush=True)
                    if device == "mps":
                        torch.mps.empty_cache()
            del harness
        responses = load_responses(output, comparison, design, True)
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
        verify_outputs(project, output)
        print("Baseline-context diagnostic complete; original study and first pilots remain unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocols/baseline_context_pilot_v1.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run(args.project.resolve(), args.protocol.resolve(), pick_device(args.device), args.verify_only)


if __name__ == "__main__":
    main()