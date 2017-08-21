import unittest
import torch
import torchaudio # one could replace torchaudio.load with scipy.io.wavfile.read
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from wavenet.layers import *
from wavenet.utils import *
from wavenet.wavenet import FastWaveNet
from test.models import *
import numpy as np

class Test_wavenet(unittest.TestCase):
    use_cuda = torch.cuda.is_available()
    def test1_wavenet_mono(self):
        num_samples = 1<<12
        input = torch.rand(1, 1, num_samples)
        m = FastWaveNet(layers=8, # less than non-unique prime factors in input size
                        blocks=2, # number of blocks
                        residual_channels=16,
                        dilation_channels=32,
                        skip_channels=16,
                        quantization_channels=256,
                        input_len=num_samples,
                        kernel_size=2)
        if self.use_cuda:
            m, input = m.cuda(), input.cuda()
        input = Variable(input)
        print(input.size())
        # forward pass
        output = m(input)
        # tests on output
        print(output.size())
        self.assertEqual(input.size(-1), output.size(-1))
        print(m.sizes)

    def test2_wavenet_stereo(self):
        # TODO fix with stereo signals.  For now, just split into multiple monos.
        num_samples = 1<<12
        input = torch.rand(1, 2, num_samples)
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
        if self.use_cuda:
            m, input = m.cuda(), input.cuda()
        input = Variable(input)
        # forward pass
        output = m(input)
        # tests on output
        print(input.size(), output.size())
        print(m.sizes)
        self.assertEqual(input.size(-1), output.size(-1))

    def test3_wavenet_dummy(self):
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
        input = input.view(1, 1, -1)
        labels = input.numpy()
        labels = mu_law_encoding(labels, 256)
        labels = torch.from_numpy(labels).squeeze().long()
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
        if self.use_cuda:
            m = m.cuda()
            criterion = criterion.cuda()
            input, labels = input.cuda(), labels.cuda()
        input, labels = Variable(input), Variable(labels)

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

    def test4_wavenet_audio(self):
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
        labels = input.numpy()
        labels = mu_law_encoding(labels, 256)
        labels = torch.from_numpy(labels).squeeze().long()

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

        epochs = 250

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs//3)

        if self.use_cuda:
            m = m.cuda()
            criterion = criterion.cuda()
            input, labels = input.cuda(), labels.cuda()
        input, labels = Variable(input), Variable(labels)

        losses = []
        for epoch in range(epochs):
            scheduler.step()
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
            if self.use_cuda:
                input = input.data.cpu()
            else:
                input = input.data
            input = input.float().numpy().ravel()
            output = F.softmax(output.t())
            if self.use_cuda:
                output = output.data.cpu()
            else:
                output = output.data
            output = output.max(0)[1].float().numpy()
            output = mu_law_expansion(output, 256)
            print(input.shape, output.shape)
            print(input.min(), input.max(), output.min(), output.max())

            f, ax = plt.subplots(2, sharex=True)
            ax[0].plot(input)
            ax[1].plot(output)
            f.savefig("test/test_wavenet_audio.png")

            plt.figure()
            plt.plot(losses)
            plt.savefig("test/test_wavenet_audio_loss.png")

        output = torch.from_numpy(output) * (1 << 30)
        output = output.unsqueeze(1).long()
        #output = output.float()

        torchaudio.save("test/data/david_16000hz_output_sample.wav", output, sr//3)

if __name__ == '__main__':
    unittest.main()
