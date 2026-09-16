#!/bin/zsh
set -eu
set -o pipefail

project_dir=${0:A:h}
python="$project_dir/.venv/bin/python"
manifest="$project_dir/data/validated_manifest_v1.json"
study="$project_dir/results/validated_v1"
capability="$project_dir/results/capability_benchmarks_v1"
heads="$project_dir/results/heads_validated_v1"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for model in llama-3.2-3b phi-3.5-mini nemotron-mini-4b; do
    "$python" -u "$project_dir/src/run_validated_study.py" \
        --manifest "$manifest" --model "$project_dir/../$model" --outdir "$study" \
        --device mps --stage calibrate
    for task in intermediate transfer container_swap; do
        eligible=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["eligible_for_tracking_claim"]))' \
            "$study/$model/$task/calibration.json")
        if [[ "$eligible" == 1 ]]; then
            "$python" -u "$project_dir/src/run_study_cases.py" \
                --manifest "$manifest" --model "$project_dir/../$model" --study "$study" \
                --task "$task" --device mps
        else
            "$python" -u "$project_dir/src/run_validated_study.py" \
                --manifest "$manifest" --model "$project_dir/../$model" --outdir "$study" \
                --device mps --stage all --tasks "$task"
        fi
    done
    if [[ ! -f "$heads/heads_${model}_intermediate.json" ]]; then
        "$python" -u "$project_dir/src/validate_heads.py" \
            --manifest "$manifest" --model "$project_dir/../$model" --task intermediate \
            --device mps --output "$heads/heads_${model}_intermediate.json"
    fi
    "$python" -u "$project_dir/src/run_capability_study.py" \
        --manifest "$manifest" --model "$project_dir/../$model" --study "$study" \
        --device mps --outdir "$capability" --n-hs 240 \
        --hellaswag "$project_dir/data_bench/hellaswag_val.jsonl" \
        --arc "$project_dir/data_bench/arc_easy_test_200.json" \
        --text "$project_dir/data_bench/tinyshakespeare.txt"
done

"$python" "$project_dir/src/collect_study.py" \
    --study "$study" --capability "$capability" --heads "$heads" \
    --output-dir "$project_dir/results/final_validated_evidence_v1" --require-complete

"$python" "$project_dir/src/finalize_study.py" \
    --snapshot "$project_dir/results/final_validated_evidence_v1" \
    --paper "$project_dir/paper" --plan "$project_dir/PENDING_PLAN.md"

printf '\nValidated experiment queue completed\n'