"""slmpaper: public-benchmark experiment core for the efficient SLM paper.

Pure-logic modules (bio, metrics, implicit, datasets) have zero heavy deps so the
core test suite runs anywhere. Model code (probe/train/latency) is opt-in.
"""
__all__ = ["bio", "metrics", "implicit", "datasets"]
