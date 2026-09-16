# Modern Backbone Extension Status (US-trained, non-Qwen)

Date: 2026-08-13

## Scope requested
- Keep GPT-2 as anchor
- Add one modern backbone (user constrained to US-trained, non-Chinese)
- Re-run core claims only:
  - probe picks shallow depth
  - d3 vs d12 tradeoff
  - parse-failure elimination still holds
  - latency trend near-linear with depth

## Completed
1. **Code generalized from GPT2-only to architecture-agnostic decoder backbone handling**
   - Added `src/slmpaper/backbone.py`:
     - `get_num_layers(model)` supports GPT-style (`model.h`) and Llama-style (`model.model.layers`)
     - `truncate_backbone_layers(model, depth)` updates both module stacks and config layer counts
   - Updated `src/slmpaper/pruned_model.py` to use `AutoModel` + generic truncation helpers
   - Updated `scripts/probe_sweep.py` to use architecture-agnostic layer count
2. **TDD evidence**
   - Added RED tests: `tests/test_backbone_utils.py`
   - Verified fail-first: `ModuleNotFoundError: No module named 'slmpaper.backbone'`
   - Implemented helpers and re-ran targeted tests: **10 passed**

## Blocker
Could not run modern US-trained decoder experiment due model availability/network constraints in this environment:

- Hugging Face access fails (`nodename nor servname provided`)
- Local cached decoder checkpoints with weights:
  - `Qwen/Qwen2.5-0.5B-Instruct` (disallowed by user constraint)
- Local/available US-trained modern decoder candidates (Nemotron/Phi/Pythia/OPT) are not cached with usable weights and cannot be downloaded in current network state

## Exact next step once network/model path is available
Use one US-trained decoder (preferred Nemotron-small local path), then run:

```bash
cd experiments/code
source .venv/bin/activate

# Probe sweep (core claim 1)
python scripts/probe_sweep.py --dataset atis --model-path <US_MODEL_PATH_OR_ID> \
  --max-train 2000 --output results/atis_probe_sweep_usmodern.json

# d3 vs d12 + latency (core claims 2 and 4)
python scripts/train_pruned.py --dataset atis --depth 3 --model-path <US_MODEL_PATH_OR_ID> \
  --max-train 5000 --output results/atis_pruned_depth3_usmodern.json
python scripts/train_pruned.py --dataset atis --depth 12 --model-path <US_MODEL_PATH_OR_ID> \
  --max-train 5000 --output results/atis_pruned_depth12_usmodern.json

# Generative baseline parse-failure check (core claim 3)
python scripts/train_generative.py --dataset atis --model-path <US_MODEL_PATH_OR_ID> \
  --max-train 5000 --max-eval 300 --output results/atis_generative_usmodern.json
```

Then aggregate into final tables and paper text.
