"""RED: CRF-aware evaluation (Viterbi decode -> word-level spans)."""
import torch
from transformers import GPT2Config

from slmpaper.batch import Batch
from slmpaper.evaluation import evaluate_model_crf
from slmpaper.pruned_model import PrunedGenerativeClassifier


def test_evaluate_model_crf_runs_and_returns_metrics():
    cfg = GPT2Config(vocab_size=50, n_positions=32, n_embd=8, n_layer=3, n_head=2)
    model = PrunedGenerativeClassifier(cfg, depth=2, num_intents=2, num_tags=3, use_crf=True)
    id2intent = {0: "a", 1: "b"}
    id2tag = {0: "O", 1: "B-x", 2: "I-x"}

    batch = Batch(
        input_ids=torch.randint(0, 50, (2, 5)),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        intent_labels=torch.tensor([0, 1]),
        slot_labels=torch.tensor([[0, 1, -100, 2, -100], [0, -100, 0, -100, -100]]),
    )
    res = evaluate_model_crf(model, [batch], id2intent, id2tag)
    assert 0.0 <= res["intent_accuracy"] <= 1.0
    assert "f1" in res["slot_f1"]
