import unittest

import torchaudio

from wavenet.layers import *

sig, sr = torchaudio.load("data/yesno/raw/waves_yesno/0_0_0_0_1_1_1_1.wav")  # (l, c)
sig.t_()  # (l, c) -> (c, l)
if len(sig.size()) == 2:
    sig.unsqueeze_(0)  # (d=1, c, l)
print("original size: {}".format(sig.size()))
sig, pad_len = dilate(sig, 12)
print("dilate1 size: {} with padding of {}".format(sig.size(), pad_len))
sig, pad_len = dilate(sig, 8)
print("dilate2 size: {} with padding of {}".format(sig.size(), pad_len))
