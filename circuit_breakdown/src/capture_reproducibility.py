"""Capture post-run file identities and installed versions without loading weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform

from audit_study import file_sha256, read_json
from collect_study import MODELS
from study_data import fingerprint, write_new


def file_record(path, root):
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during checksum capture: {path.name}")
    return {"path": path.relative_to(root).as_posix(), "bytes": after.st_size, "sha256": digest}


def model_assets(root):
    paths = sorted(path for path in root.iterdir() if path.is_file() and
                   (path.suffix in (".safetensors", ".bin", ".json", ".model", ".py", ".jinja", ".txt")
                    or path.name in ("LICENSE", "README.md")))
    if not any(path.suffix in (".safetensors", ".bin") for path in paths):
        raise ValueError(f"No active root checkpoint weights: {root.name}")
    return paths


def project_assets(project):
    paths = set()
    for pattern in ("src/*.py", "tests/test_*.py", "paper/*.md", "docs/*.md",
                    "results/validated_v1/**/*.json", "results/heads_validated_v1/*.json",
                    "results/capability_benchmarks_v1/**/*.json", "results/capability_benchmarks_v1/**/*.pt",
                    "results/final_validated_evidence_v1/*", "results/postrun_audit_v1/consistency.json"):
        paths.update(path for path in project.glob(pattern) if path.is_file())
    for name in ("data/validated_manifest_v1.json", "data_bench/hellaswag_val.jsonl",
                 "data_bench/arc_easy_test_200.json", "data_bench/tinyshakespeare.txt",
                 "requirements.txt", "run_validated_all.sh", "STUDY_PROTOCOL.md", "METRICS.md",
                 "PENDING_PLAN.md", "NEXT_STEPS.md", "README.md", "BUDGET_CONTROL_RESULTS.md",
                 "DISSOCIATION_RESULTS.md", "CAPABILITY_DEPLOYED.md"):
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    return sorted(paths)


def verify_files(records, root):
    for record in records:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Captured file differs: {record['path']}")


def verify_capture(saved, project):
    body = {name: value for name, value in saved.items() if name != "sha256"}
    if saved.get("sha256") != fingerprint(body):
        raise ValueError("Reproduction manifest hash mismatch")
    verify_files(saved["project_files"], project)
    for model, records in saved["models"].items():
        verify_files(records, project.parent / model)


def capture(project):
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in metadata.distributions() if distribution.metadata["Name"]}
    packages = dict(sorted(packages.items(), key=lambda item: item[0].lower()))
    manifest = {"capture_version": "postrun_reproduction_v1",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Content available at capture time, not proof of historical run-time bytes or remote model revisions.",
                "environment": {"python": platform.python_version(), "system": platform.system(),
                                "release": platform.release(), "machine": platform.machine(),
                                "packages": packages},
                "project_files": [], "models": {},
                "limitations": [
                    "Original weights were not content-hashed at run time; current hashes cannot authenticate historical execution.",
                    "The original size fingerprint enumerates only .safetensors, not Nemotron's .bin weights.",
                    "Model inventory covers active root loader assets, not redundant conversion archives or download caches.",
                    "Installed versions are captured now; only limited dependency versions were recorded in the original runs.",
                    "Version pins do not guarantee package availability or clean-room/platform-independent replay.",
                    "Original artifacts can contain identifying absolute paths; anonymize a separate release copy before submission."]}
    paths = project_assets(project)
    print(f"Hashing {len(paths)} project/code/data/result files", flush=True)
    manifest["project_files"] = [file_record(path, project) for path in paths]
    for model in MODELS:
        root = project.parent / model
        assets = model_assets(root)
        print(f"Hashing {model}: {len(assets)} active root assets (streamed, no model loading)", flush=True)
        manifest["models"][model] = [file_record(path, root) for path in assets]
    manifest["sha256"] = fingerprint(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    if args.verify:
        saved = read_json(args.verify)
        verify_capture(saved, project)
        print(f"Verified {len(saved['project_files'])} project files and {len(saved['models'])} model inventories")
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a fresh capture directory: {args.output_dir}")
    manifest = capture(project)
    write_new(args.output_dir / "manifest.json", manifest)
    pins = "".join(f"{name}=={version}\n" for name, version in manifest["environment"]["packages"].items())
    with (args.output_dir / "requirements.lock.txt").open("x") as stream:
        stream.write(pins)
    print(f"Saved post-run reproduction capture -> {args.output_dir}")


if __name__ == "__main__":
    main()