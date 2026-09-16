import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from study_data import build_manifest, fingerprint, load_manifest, validate_manifest, write_new


class StudyDataTests(unittest.TestCase):
    def test_disjoint_manifest_and_held_out_template(self):
        manifest = build_manifest([], seeds=[0, 1], train=8, dev=4, test=6, fewshot=2)
        self.assertIs(validate_manifest(manifest), manifest)
        for splits in manifest["tasks"]["container_swap"].values():
            self.assertEqual({pair["metadata"]["template"] for pair in splits["test"]}, {1})
            self.assertEqual({pair["metadata"]["template"] for pair in splits["train"]}, {0})
        with TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            write_new(path, manifest)
            self.assertEqual(load_manifest(path), manifest)
            with self.assertRaises(FileExistsError):
                write_new(path, manifest)

    def test_manifest_tampering_is_rejected(self):
        manifest = build_manifest([], seeds=[0], train=2, dev=2, test=2, fewshot=1)
        manifest["counts"]["train"] = 4
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_manifest(manifest)

    def test_overlap_is_rejected_even_with_updated_hash(self):
        manifest = build_manifest([], seeds=[0], train=2, dev=2, test=2, fewshot=1)
        splits = manifest["tasks"]["intermediate"]["0"]
        splits["test"][0] = copy.deepcopy(splits["train"][0])
        manifest["sha256"] = fingerprint({key: value for key, value in manifest.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()