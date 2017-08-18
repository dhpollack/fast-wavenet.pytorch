import unittest
import torch
import torchaudio # one could replace torchaudio.load with scipy.io.wavfile.read
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from wavenet.layers import *
from wavenet.utils import *
from wavenet.wavenet import FastWaveNet
from test.models import *
import numpy as np

class Test_wavenet(unittest.TestCase):
    self.use_cuda = torch.cuda.is_available()
    def test1_wavenet_mono(self):
        num_samples = 1<<12
        input = Variable(torch.rand(1, 1, num_samples))
        print(input.size())
        m = FastWaveNet(layers=8, # less than non-unique prime factors in input size
                        blocks=2, # number of blocks
                        residual_channels=16,
                        dilation_channels=32,
                        skip_channels=16,
                        quantization_channels=256,
                        input_len=num_samples,
                        kernel_size=2)
        output = m(input)
        print(output.size())
        self.assertEqual(input.size(-1), output.size(-1))
        print(m.sizes)

    def test2_wavenet_stereo(self):
        num_samples = 1<<12
        input = Variable(torch.rand(1, 2, num_samples))
        batch_size, audio_channels, _ = input.size()
        m = FastWaveNet(layers=8,
                        blocks=2, # number of blocks
                        residual_channels=16,
                        dilation_channels=32,
                        skip_channels=16,
                        quantization_channels=256,
                        input_len=num_samples,
                        audio_channels=audio_channels,
                        kernel_size=2)
        output = m(input)
        print(output.size())
        self.assertEqual(input.size(-1), output.size(-1))

    def test3_wavenet_parameter_count(self):
        num_samples = 1<<14
        input = Variable(torch.rand(1, 2, num_samples))
        batch_size, audio_channels, _ = input.size()
        m = FastWaveNet(layers=12,
                        blocks=4, # number of blocks
                        residual_channels=16,
                        dilation_channels=32,
                        skip_channels=16,
                        quantization_channels=256,
                        input_len=num_samples,
                        audio_channels=audio_channels,
                        kernel_size=2)
        output = m(input)
        print(output.size())
        print(m.parameter_count())
        self.assertEqual(input.size(-1), output.size(-1))

    def test4_wavenet_loss(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("install matplotlib for plot of signals")
            plt = None
        # setup inputs and labels
        num_samples = 1<<10
        #for some reason the network has a lot more trouble with the sinewave
        input = torch.linspace(0, 20*np.pi, num_samples)
        input = torch.sin(input)
        #input = torch.rand(1, 1, num_samples) * 2. - 1.
        input = Variable(input.view(1, 1, -1))
        labels = input.data.numpy()
        labels = mu_law_encoding(labels, 256)
        labels = Variable(torch.from_numpy(labels).squeeze().long())
        batch_size, audio_channels, _ = input.size()
        print(input.size(), labels.size())
        # build network and optimizer
        m = FastWaveNet(layers=10,
                        blocks=6, # number of blocks
                        residual_channels=16,
                        dilation_channels=32,
                        skip_channels=16,
                        quantization_channels=256,
                        input_len=num_samples,
                        audio_channels=audio_channels,
                        kernel_size=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
        epochs = 100
        losses = []
        for epoch in range(epochs):
            m.zero_grad()
            output = m(input)
            if epoch == 0:
                print(m.sizes)
            output.squeeze_()
            output = output.t()
            loss = criterion(output, labels)
            losses.append(loss.data[0])
            if epoch % (epochs // 10) == 0:
                print("loss of {} at epoch {}".format(losses[-1], epoch+1))
            loss.backward()
            optimizer.step()
        print("final loss of {} after {} epochs".format(losses[-1], epoch+1))

        if plt is not None:
            input = input.data.float().numpy().ravel()

            output = F.softmax(output.t())
            output = output.max(0)[1].data.float().numpy()
            output = mu_law_expansion(output, 256)
            print(input.shape, output.shape)
            print(input.min(), input.max(), output.min(), output.max())

            f, ax = plt.subplots(2, sharex=True)
            ax[0].plot(input)
            ax[1].plot(output)
            f.savefig("test/test_wavenet_dummy.png")

        # j = i + 1
        for l_i, l_j, in zip(losses[-1:], losses[1:]):
            self.assertTrue(l_j >= l_i)

    def test5_wavenet_music(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("install matplotlib for plot of signals")
            plt = None

        num_samples = 1 << 15

        sig, sr = torchaudio.load("test/data/david.wav")
        sig = sig[:-(sig.size(0)%3):3]
        input = sig[16000:(16000+num_samples)].contiguous()
        # write sample for qualitative test
        torchaudio.save("test/data/david_16000hz_input_sample.wav", input, sr//3)
        input /= torch.abs(input).max()
        assert input.min() >= -1. and input.max() <= 1.
        input = input.view(1, 1, -1)
        input = Variable(input)
        labels = input.data.numpy()
        labels = mu_law_encoding(labels, 256)
        labels = Variable(torch.from_numpy(labels).squeeze().long())

        # build network and optimizer
        m = FastWaveNet(layers=10,
                        blocks=4, # number of blocks
                        residual_channels=16,
                        dilation_channels=32,
                        skip_channels=16,
                        quantization_channels=256,
                        input_len=num_samples,
                        audio_channels=1,
                        kernel_size=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
        epochs = 100
        losses = []
        for epoch in range(epochs):
            m.zero_grad()
            output = m(input)
            output.squeeze_()
            output = output.t()
            loss = criterion(output, labels)
            losses.append(loss.data[0])
            if epoch % (epochs // 10) == 0:
                print("loss of {} at epoch {}".format(losses[-1], epoch+1))
            loss.backward()
            optimizer.step()
        print("final loss of {} after {} epochs".format(losses[-1], epoch+1))

        if plt is not None:
            input = input.data.float().numpy().ravel()

            output = F.softmax(output.t())
            output = output.max(0)[1].data.float().numpy()
            output = mu_law_expansion(output, 256)
            print(input.shape, output.shape)
            print(input.min(), input.max(), output.min(), output.max())

            f, ax = plt.subplots(2, sharex=True)
            ax[0].plot(input)
            ax[1].plot(output)
            f.savefig("test/test_wavenet_audio.png")

        output = torch.from_numpy(output) * (1 << 30)
        output = output.unsqueeze(1).long()
        #output = output.float()

        torchaudio.save("test/data/david_16000hz_output_sample.wav", output, sr//3)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
