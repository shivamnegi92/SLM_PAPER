from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from audit_study import read_json
from run_convergence_pilot import compute_lock, markdown, summarize, validate_protocol, write_tensor_case


class ConvergenceRunnerTests(unittest.TestCase):
    def protocol(self):
        return read_json(Path(__file__).resolve().parents[1] / "protocols/convergence_pilot_v1.json")

    def test_declared_protocol_is_development_only(self):
        protocol = self.protocol()
        validate_protocol(protocol)
        protocol["scope"] = "test"
        with self.assertRaisesRegex(ValueError, "declared convergence pilot"):
            validate_protocol(protocol)

    def test_cannot_switch_cases_after_observing_results(self):
        protocol = self.protocol()
        protocol["development_indices"][0] = 8
        with self.assertRaisesRegex(ValueError, "first eight"):
            validate_protocol(protocol)

    def test_binary_case_write_is_exclusive_and_readable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.pt"
            write_tensor_case(path, {"completed": True, "logits": torch.tensor([1., 2.])})
            with self.assertRaises(FileExistsError):
                write_tensor_case(path, {"completed": False})
            self.assertTrue(torch.load(path, weights_only=True)["completed"])
            self.assertEqual(list(path.parent.glob("*.partial")), [])

    def test_two_pilots_cannot_own_the_compute_lock(self):
        with TemporaryDirectory() as directory:
            with compute_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with compute_lock(Path(directory)):
                        self.fail("Acquired an already-held compute lock")

    def test_summary_requires_complete_matrix(self):
        comparison = {"protocol": self.protocol(), "development_inputs": [{"sample_id": "first"}]}
        with self.assertRaisesRegex(ValueError, "all declared cases"):
            summarize({"full": [], "track8": []}, comparison)

    def test_report_keeps_all_method_rows_in_the_table(self):
        values = {"checkpoints": {"32": {"override_ci": [.2, 1.], "successes": 1, "n": 1,
                                         "override_rate": 1., "mean_target_margin": 1., "mean_target_cross_entropy": .3}},
                  "override_128_minus_32": 0., "paired_ci": [-1., 1.], "mean_margin_128_minus_32": 0.,
                  "ever_successes": 1, "ever_success_but_final_failed": 0, "guard_shrinks": 0}
        result = markdown({"conditions": {"full": values, "track8": values},
                           "reviewer_gap_status": "open", "limitations": []})
        self.assertLess(result.index("| track8 |"), result.index("**full:**"))


if __name__ == "__main__":
    unittest.main()