"""Guard: every number in the paper must match the frozen snapshot.

The paper quotes results transcribed by hand from results/frozen_e91985c/
(commit e91985c), because the panel runs predate the --out flag and cannot be
regenerated -- each seed resamples both pairs and the few-shot prefix. That
transcription is the weak link, so it is checked mechanically rather than by
eye. Also asserts the agreed wording constraints, which are as easy to violate
during editing as a digit is.

Run before any commit that touches paper/.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
FROZEN = PROJECT / "results/frozen_e91985c"


def main():
    paper = (PROJECT / "paper/PAPER_DRAFT.md").read_text()
    abstract = (PROJECT / "paper/ABSTRACT_DRAFT.md").read_text()
    data = json.loads((PROJECT / "paper/figure_data.json").read_text())
    phi = (FROZEN / "panel_phi_final.log").read_text()
    llama = (FROZEN / "panel_llama_final.log").read_text()
    uniqueness = (FROZEN / "uniqueness.log").read_text()

    checks = []

    def check(label, condition):
        checks.append((label, bool(condition)))

    # headline results, verbatim from the logs
    check("phi selectivity 0/240", "0/240 = 0.0% [0.0%, 1.6%]" in phi)
    check("llama selectivity 0/240", "0/240 = 0.0% [0.0%, 1.6%]" in llama)
    check("phi completeness 172/240", "172/240 = 71.7% [65.7%, 77.0%]" in phi)
    check("llama completeness 53/120", "53/120 = 44.2% [35.6%, 53.1%]" in llama)

    # competence, including the two exclusions the paper depends on
    check("phi current_holder 99.2%", "current_holder  119/120   99.2%" in phi)
    check("phi object_identity 100%", "object_identity  120/120  100.0%" in phi)
    check("llama current_holder excluded at 46.7%",
          "46.7% [38.0%, 55.6%]  <- EXCLUDED" in llama)
    check("llama object_identity 100%", "object_identity  120/120  100.0%" in llama)
    check("phi transfer_count excluded", "transfer_count   25/120   20.8%" in phi)
    check("llama transfer_count excluded", "transfer_count   22/120   18.3%" in llama)

    # per-probe transport rates quoted in the tables
    for needle, log, label in (
            ("74.2% [65.7%,81.2%]", phi, "phi current_holder -> donor"),
            ("71.7% [63.0%,79.0%]", phi, "phi object_identity -> donor"),
            ("60.0% [51.1%,68.3%]", phi, "phi original_holder -> donor"),
            ("69.2% [60.4%,76.7%]", phi, "phi last_recipient -> donor"),
            ("49.2% [40.4%,58.0%]", llama, "llama object_identity -> donor"),
            ("46.7% [38.0%,55.6%]", llama, "llama original_holder -> donor"),
            ("44.2% [35.6%,53.1%]", llama, "llama last_recipient -> donor")):
        check(label, needle in log)

    # sampling and stability
    check("phi per-seed 0/40 on both probes", phi.count("0/40, 0/40, 0/40") == 2)
    check("llama per-seed 0/40 on both probes", llama.count("0/40, 0/40, 0/40") == 2)
    check("phi N decomposition",
          "independent reasoning pairs = 120; graded probes = 4; probe-outcomes = 480" in phi)
    check("llama N decomposition",
          "independent reasoning pairs = 120; graded probes = 3; probe-outcomes = 360" in llama)

    # uniqueness falsification
    for value in ("+0.840", "+0.912", "+0.948"):
        check(f"uniqueness median cosine {value}",
              f"SIMILAR (median cosine {value})" in uniqueness)

    # figure data matches the logs
    phi_json = data["models"]["phi-3.5-mini"]
    llama_json = data["models"]["llama-3.2-3b"]
    check("json phi completeness", abs(phi_json["completeness"]["rate"] - 0.717) < 1e-9)
    check("json phi selectivity zero", phi_json["selectivity"]["rate"] == 0.0)
    check("json llama completeness", abs(llama_json["completeness"]["rate"] - 0.442) < 1e-9)
    check("json phi 4 graded probes", phi_json["graded_probes"] == 4)
    check("json llama 3 graded probes", llama_json["graded_probes"] == 3)
    check("json norm-matched flat zero",
          all(v == 0.0 for v in data["magnitude"]["p_donor_norm_matched"]))
    check("json dose-response endpoint",
          abs(data["magnitude"]["p_donor_real"][-1] - 0.718) < 1e-9)

    # prior work must be credited, in both paper and abstract
    for needle, label in (("2402.17700", "RAVEL arXiv id"),
                          ("2504.13151", "MIB arXiv id"),
                          ("2507.08802", "Sutter arXiv id")):
        check(f"paper cites {label}", needle in paper)
    check("abstract names RAVEL", "RAVEL" in abstract)
    check("abstract names MIB", "MIB" in abstract)

    # wording constraints agreed after the literature check
    check("no 'competence amplifies' claim", "amplifies" not in paper.lower())
    check("safer competence wording present",
          "cannot be explained by poor baseline task competence" in paper)
    check("non-comparability of completeness flagged", "not comparable" in paper.lower())
    check("position objection conceded", "we agree" in paper.lower())
    check("specificity measurement error recorded", "That inference is wrong" in paper)
    check("limitations section present", "## 8. Limitations" in paper)
    # A blunt substring search for "new diagnostic" fires on the sentence that
    # DISCLAIMS novelty ("not a new method or a new diagnostic"), so match the
    # assertive phrasings a claim would actually use.
    lowered = paper.lower()
    novelty_claims = ["we introduce a new diagnostic", "we propose a new diagnostic",
                      "our novel diagnostic", "a novel diagnostic",
                      "we introduce a novel", "first to show that intervention"]
    check("no novelty claim for the diagnostic",
          not any(phrase in lowered for phrase in novelty_claims))
    check("novelty explicitly disclaimed",
          "not a new method or a new diagnostic" in lowered)

    failures = [label for label, ok in checks if not ok]
    for label, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    if failures:
        print("PAPER DISAGREES WITH FROZEN SNAPSHOT:")
        for label in failures:
            print(f"  - {label}")
        sys.exit(1)
    print("Paper is consistent with results/frozen_e91985c/.")


if __name__ == "__main__":
    main()
