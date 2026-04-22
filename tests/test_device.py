from __future__ import annotations

import unittest
from unittest.mock import patch

from kvant.utils.device import resolve_torch_device


class DeviceResolverTests(unittest.TestCase):
    def test_explicit_cpu_is_accepted(self) -> None:
        self.assertEqual(resolve_torch_device("cpu"), "cpu")

    def test_auto_prefers_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_torch_device("auto"), "cuda")

    def test_unavailable_cuda_is_rejected_when_explicit(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_torch_device("cuda")


if __name__ == "__main__":
    unittest.main()
