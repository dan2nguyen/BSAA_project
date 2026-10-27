"""Dependency-light detector checks; run: python -m unittest discover -s tests."""
import unittest
import numpy as np
from databasing.elgendi.detection import detect_ppg_peaks

class TestElgendi(unittest.TestCase):
    def test_synthetic_periodic_pulses(self):
        fs = 100.0
        t = np.arange(3000) / fs
        expected = np.arange(0.5, 29.5, 0.8)
        x = sum(np.exp(-0.5 * ((t - p) / 0.085)**2) for p in expected)
        peaks = detect_ppg_peaks(x, fs)
        self.assertEqual(len(peaks), len(expected))
        np.testing.assert_allclose(peaks/fs, expected, atol=1/fs)

    def test_flat_signal_has_no_peaks(self):
        self.assertEqual(len(detect_ppg_peaks(np.zeros(1000), 100)), 0)

if __name__ == '__main__':
    unittest.main()
