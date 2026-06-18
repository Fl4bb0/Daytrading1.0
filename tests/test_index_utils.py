import unittest

import numpy as np

from kvant.utils.index_utils import in_split, valid_target_positions


class InSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.val_start = "2025-06-01"
        self.test_start = "2025-07-01"
        self.test_end = "2025-07-08"

    def test_test_split_is_open_ended_without_test_end(self) -> None:
        # No upper bound supplied: anything from test_start onward is "test",
        # matching the original behaviour relied on elsewhere in the codebase.
        far_future = "2025-12-31"
        self.assertTrue(in_split(far_future, "test", self.val_start, self.test_start))

    def test_test_split_excludes_lookahead_buffer_when_test_end_given(self) -> None:
        inside_test = "2025-07-05"
        inside_buffer = "2025-07-08"  # == test_end, exclusive boundary
        before_buffer_end = "2025-07-07"

        self.assertTrue(
            in_split(inside_test, "test", self.val_start, self.test_start, test_end=self.test_end)
        )
        self.assertTrue(
            in_split(before_buffer_end, "test", self.val_start, self.test_start, test_end=self.test_end)
        )
        self.assertFalse(
            in_split(inside_buffer, "test", self.val_start, self.test_start, test_end=self.test_end)
        )

    def test_train_and_val_unaffected_by_test_end(self) -> None:
        in_train = "2025-03-01"
        in_val = "2025-06-15"
        self.assertTrue(
            in_split(in_train, "train", self.val_start, self.test_start, test_end=self.test_end)
        )
        self.assertTrue(
            in_split(in_val, "val", self.val_start, self.test_start, test_end=self.test_end)
        )

    def test_valid_target_positions_unchanged(self) -> None:
        labels = np.array([-1, -1, 0, 1, 2, -1, 0])
        out = valid_target_positions(labels, lookback_L=2)
        self.assertEqual(list(out), [2, 3, 4, 6])


if __name__ == "__main__":
    unittest.main()