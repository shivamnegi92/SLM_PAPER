from copy import deepcopy
from pathlib import Path
import unittest

import torch

from audit_study import read_json
from convergence_pilot import optimize_trajectory
from guard_rate_pilot import cell_config, cells, run_cell, validate_guard_records
import test_convergence_pilot


class GuardRatePilotTests(unittest.TestCase):
    def test_factorial_has_all_eight_unique_cells(self):
        protocol = read_json(Path(__file__).resolve().parents[1] / "protocols/guard_rate_pilot_v1.json")
        design = cells(protocol)
        self.assertEqual(len(design), 8)
        self.assertEqual(len({cell["key"] for cell in design}), 8)

    def test_only_guard_threshold_changes_between_matched_cells(self):
        common = {"method": "track8", "rate_factor": 1.}
        shrink = cell_config([1., 2.], .1, {**common, "guard": "shrink"})
        observe = cell_config([1., 2.], .1, {**common, "guard": "observe"})
        self.assertEqual(shrink["max_control_drop"], .2)
        self.assertEqual(observe["max_control_drop"], 1.)
        self.assertEqual({**shrink, "max_control_drop": 1.}, observe)

    def test_quarter_rate_does_not_change_budget_or_guard(self):
        first = cell_config([1.], .8, {"guard": "shrink", "rate_factor": 1.})
        lower = cell_config([1.], .8, {"guard": "shrink", "rate_factor": .25})
        self.assertEqual(lower, {**first, "lr": .2, "lr_b": .2})

    def test_observe_guard_records_breaches_without_shrinking(self):
        harness, record, _ = test_convergence_pilot.ConvergencePilotTests().fixture()
        cfg = cell_config([10.], .2, {"guard": "observe", "rate_factor": 1.})
        controls, baseline = [record.corrupt_ids], [2]
        result = run_cell(harness, record, [0], controls, baseline, None, cfg, [32, 64, 128])
        self.assertTrue(any(event["pre_shrink_control_drop"] > .2 for event in result["trajectory"]["guard_events"]))
        self.assertFalse(any(event["shrunk"] for event in result["trajectory"]["guard_events"]))
        validate_guard_records({**result, "cfg": cfg})
        unguarded = optimize_trajectory(harness, record, [0], [], [], None, cfg, [32, 64, 128])
        self.assertTrue(torch.equal(result["trajectory"]["snapshots"]["128"]["vectors"][0],
                                    unguarded["snapshots"]["128"]["vectors"][0]))
        self.assertEqual(len(harness.layers[0]._forward_hooks), 0)

    def test_rejects_falsified_shrink_flag(self):
        harness, record, _ = test_convergence_pilot.ConvergencePilotTests().fixture()
        cfg = cell_config([10.], .2, {"guard": "observe", "rate_factor": 1.})
        result = run_cell(harness, record, [0], [record.corrupt_ids], [2], None, cfg, [32, 64, 128])
        changed = deepcopy(result)
        changed["trajectory"]["guard_events"][0]["shrunk"] = True
        with self.assertRaisesRegex(ValueError, "guard.policy"):
            validate_guard_records({**changed, "cfg": cfg})


if __name__ == "__main__":
    unittest.main()