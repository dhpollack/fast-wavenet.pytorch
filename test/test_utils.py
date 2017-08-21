import unittest
import torch
import torchaudio
from wavenet.utils import *
import numpy as np

class Test_mu_law(unittest.TestCase):
    sig, sr = torchaudio.load("data/yesno/raw/waves_yesno/0_0_0_0_1_1_1_1.wav")
    def test1_mu_law_encoding(self):
        quantization_channels = 256
        mu = quantization_channels - 1.
        sig = self.sig.numpy()
        sig /= np.abs(sig).max()
        self.assertTrue(sig.min() >= -1. and sig.max() <= 1.)

        sig_mu = mu_law_encoding(sig, mu)
        print(sig_mu.ptp(), sig_mu.min(), sig_mu.max())
        self.assertTrue(sig_mu.min() >= 0. and sig.max() <= quantization_channels)

        sig_exp = mu_law_expansion(sig_mu, mu)
        print(sig_exp.ptp(),sig_exp.min(), sig_exp.max(), sig_exp.shape)
        self.assertTrue(sig_exp.min() >= -1. and sig_exp.max() <= 1.)

        diff = sig - sig_exp
        mse = np.linalg.norm(diff) / diff.shape[0]
        print(mse, np.isclose(mse, 0., atol=1e-4))

    def test2_prime_factorization(self):
        num = 100
        factors_true = [2, 2, 5, 5]
        factors_calc = list(prime_factors(num))
        self.assertEqual(factors_true, factors_calc, print(factors_calc))

        num = 16000
        factors_true = [2, 2, 2, 2, 2, 2, 2, 5, 5, 5]
        factors_calc = list(prime_factors(num))
        self.assertEqual(factors_true, factors_calc, print(factors_calc))

if __name__ == '__main__':
    unittest.main()
