from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from capture_reproducibility import file_record, model_assets, verify_capture, verify_files


class ReproductionCaptureTests(unittest.TestCase):
    def test_captures_bytes_and_detects_same_size_content_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "weights.bin"
            path.write_bytes(b"first")
            record = file_record(path, root)
            self.assertEqual(record["path"], "weights.bin")
            verify_files([record], root)
            path.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "Captured file differs"):
                verify_files([record], root)

    def test_bin_weights_are_included_and_cache_is_excluded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytorch_model.bin").write_bytes(b"fixture")
            (root / "config.json").write_text("{}")
            (root / ".cache").mkdir()
            (root / ".cache" / "duplicate.bin").write_bytes(b"fixture")
            names = [path.name for path in model_assets(root)]
        self.assertEqual(names, ["config.json", "pytorch_model.bin"])

    def test_missing_weights_are_not_silently_accepted(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No active root checkpoint"):
                model_assets(Path(directory))

    def test_manifest_hash_is_checked_before_files(self):
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            verify_capture({"sha256": "wrong"}, Path("."))


if __name__ == "__main__":
    unittest.main()