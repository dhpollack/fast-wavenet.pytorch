import unittest

import torchaudio

from wavenet.layers import *

sig, sr = torchaudio.load("data/yesno/raw/waves_yesno/0_0_0_0_1_1_1_1.wav")
if len(sig.size()) == 2:
    sig.unsqueeze_(0)
print("original size: {}".format(sig.size()))
sig = dilate(sig, 12)
print("dilate1 size: {}".format(sig.size()))
sig = dilate(sig, 8, init_dilation=sig.size(0))
print("dilate2 size: {}".format(sig.size()))
