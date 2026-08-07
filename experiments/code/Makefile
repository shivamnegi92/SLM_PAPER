.PHONY: test smoke repro

test:
	python -m pytest -q

smoke:
	python scripts/smoke.py

# M0 gate: full unit suite + offline end-to-end smoke.
repro: test smoke
	@echo "M0 repro: GREEN"
