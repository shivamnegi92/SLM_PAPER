from tempfile import TemporaryDirectory
import unittest

from collect_study import collect, markdown


class EvidenceLedgerTests(unittest.TestCase):
    def test_empty_directory_is_incomplete_not_success(self):
        with TemporaryDirectory() as root:
            result = collect(root, root, root)
        self.assertFalse(result["matrix_complete"])
        self.assertEqual(len(result["missing"]), 21)
        self.assertIn("INCOMPLETE", markdown(result))

    def test_accuracy_delta_table_uses_percentage_points(self):
        ledger = {"matrix_complete": True, "missing": [], "study": [], "heads": [],
                  "capability": [{"comparison": {"model": "tiny"}, "seed": 0,
                                  "conditions": {"active_prefix": {"benchmarks": {
                                      "tiny": {"delta": .01, "delta_ci": [-.02, .04]}}}}}]}
        result = markdown(ledger)
        self.assertIn("accuracy change (pp)", result)
        self.assertIn("| 1.0 | [-2.0, 4.0] |", result)
        self.assertNotIn("1.0%", result)


if __name__ == "__main__":
    unittest.main()