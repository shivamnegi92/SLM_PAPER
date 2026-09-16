import unittest

from public_benchmarks import arc_item, text_windows


class PublicBenchmarkTests(unittest.TestCase):
    def test_numeric_and_letter_answer_labels(self):
        for labels, key in ((["A", "B"], "B"), ([1, 2], "2")):
            record = {"id": "example", "question": "What is water?", "answerKey": key,
                      "choices": {"label": labels, "text": ["solid", "liquid"]}}
            self.assertEqual(arc_item(record)["correct_index"], 1)

    def test_unknown_answer_fails(self):
        with self.assertRaises(ValueError):
            arc_item({"id": "bad", "question": "Q", "answerKey": "C",
                      "choices": {"label": ["A", "B"], "text": ["one", "two"]}})

    def test_text_windows_are_disjoint(self):
        self.assertEqual(text_windows("abcdefghij", 2, 5), ["abcde", "fghij"])
        with self.assertRaises(ValueError):
            text_windows("short", 2, 5)


if __name__ == "__main__":
    unittest.main()