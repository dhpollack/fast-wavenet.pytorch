import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
from wavenet.layers import *
from test.models import *
import numpy as np

class Test_dilation(unittest.TestCase):
    def test_dilate(self):
        input = Variable(torch.arange(0, 13).view(1, 1, 13))

        dilated, _ = dilate(input, 1)
        self.assertEqual(dilated.size(), (1, 1, 13))
        self.assertEqual(dilated[0, 0, 4].data[0], 4)

        dilated, _ = dilate(input, 2)
        self.assertEqual(dilated.size(), (2, 1, 7))
        self.assertEqual(dilated[1, 0, 2].data[0], 4)

        dilated, _ = dilate(input, 4)
        self.assertEqual(dilated.size(), (4, 1, 4))
        self.assertEqual(dilated[3, 0, 1].data[0], 4)

        dilated, _ = dilate(dilated, 1)
        self.assertEqual(dilated.size(), (1, 1, 16))
        self.assertEqual(dilated[0, 0, 7].data[0], 4)

    def test_dilate_multichannel(self):
        input = Variable(torch.arange(0, 36).view(2, 3, 6))

        dilated, _ = dilate(input, 1)
        self.assertEqual(dilated.size(), (1, 3, 12))
        dilated, _ = dilate(input, 2)
        self.assertEqual(dilated.size(), (2, 3, 6))
        dilated, _ = dilate(input, 4)
        self.assertEqual(dilated.size(), (4, 3, 3))

    def test_dilate_invalid(self):
        input = Variable(torch.arange(0, 36).view(2, 3, 6))

        try:
            dilate(input, 5)
        except AssertionError:
            print("raised AssertionError")

class Test_padding(unittest.TestCase):
    def test_constantpad1d(self):

        # equal padding on all 4 sides
        input = torch.rand(3, 2, 5)
        padding = 1
        m = ConstantPad1d(padding) # m for model
        output = m(input).data
        self.assertEqual(input[0, 0, 0], output[0, padding, padding])
        self.assertTrue(np.all(output[0, :, 0].numpy()==0))
        self.assertTrue(np.all(output[0, :, -1].numpy()==0))
        self.assertTrue(np.all(output[0, 0, :].numpy()==0))
        self.assertTrue(np.all(output[0, -1, :].numpy()==0))

        # unequal padding on dimensions, but equal within dimension
        input = torch.rand(3, 2, 5)
        padding = (1, 2)
        m = ConstantPad1d(padding) # m for model
        output = m(input).data
        self.assertEqual(input[0, 0, 0], output[0, padding[1], padding[0]])
        self.assertTrue(np.all(output[0, :, :padding[0]].numpy()==0))
        self.assertTrue(np.all(output[0, :, -padding[0]:].numpy()==0))
        self.assertTrue(np.all(output[0, :padding[1], :].numpy()==0))
        self.assertTrue(np.all(output[0, -padding[1]:, :].numpy()==0))

        # padding in one dimension, like we'll use for wavenet
        input = torch.rand(3, 2, 5)
        padding = (3, 0, 0, 0)
        m = ConstantPad1d(padding) # m for model
        output = m(input).data
        self.assertTrue(np.all(output[:, :, :padding[0]].numpy()==0))

        # non-zero padding, possibly useful for masking
        input = torch.rand(3, 2, 5)
        padding = (3, 0, 0, 0)
        pad_val = -100
        m = ConstantPad1d(padding, pad_val) # m for model
        output = m(input).data
        self.assertTrue(np.all(output[:, :, :padding[0]].numpy()==pad_val))

class Test_conv1dext(unittest.TestCase):
    def test_ncc(self):
        module = Conv1dExt(in_channels=3,
                           out_channels=5,
                           kernel_size=4)
        rand = Variable(torch.rand(5, 3, 4))
        module._parameters['weight'] = module.weight * module.weight + rand * 1
        ncc = module.normalized_cross_correlation()
        print("ncc:\n{}".format(ncc.data))

class Test_simple_models(unittest.TestCase):
    def test_net_forward(self):

        model = Net()
        print(model)
        self.assertEqual(model.conv1.out_channels, model.conv2.out_channels)
        self.assertEqual(model.conv1.out_channels, model.conv3.in_channels)
        self.assertEqual(model.conv2.out_channels, model.conv3.in_channels)
        self.assertEqual(model.conv3.out_channels, model.conv4.in_channels)

        # simple forward pass
        input = Variable(torch.rand(1, 1, 4) * 2 - 1)
        output = model(input)
        self.assertEqual(output.size(), (1, 2, 4))

        # feature split
        model.conv1.split_feature(feature_i=1)
        model.conv2.split_feature(feature_i=3)
        print(model)
        self.assertEqual(model.conv1.out_channels, model.conv2.out_channels)
        self.assertEqual(model.conv1.out_channels, model.conv3.in_channels)
        self.assertEqual(model.conv2.out_channels, model.conv3.in_channels)
        self.assertEqual(model.conv3.out_channels, model.conv4.in_channels)

        output2 = model(input)

        diff = output - output2

        dot = torch.dot(diff.view(-1), diff.view(-1))
        # should be close to 0
        #self.assertTrue(np.isclose(dot.data[0], 0., atol=1e-2))
        print("mse: ", dot.data[0])

class Test_dilated_queue(unittest.TestCase):
    def test_enqueue(self):
        queue = DilatedQueue(max_length=8, num_channels=3)
        e = torch.zeros((3))
        for i in range(11):
            e = e + 1
            queue.enqueue(e)

        data = queue.data[0, :].data
        #print('data: ', data)
        self.assertEqual(data[0], 9)
        self.assertEqual(data[2], 11)
        self.assertEqual(data[7], 8)

    def test_dequeue(self):
        queue = DilatedQueue(max_length=8, num_channels=1)
        e = torch.zeros((1))
        for i in range(11):
            e = e + 1
            queue.enqueue(e)

        #print('data: ', queue.data)

        for i in range(9):
            d = queue.dequeue(num_deq=3, dilation=2)
            d = d.data # only using values for tests
            #print("dequeue size: {}".format(d.size()))

        self.assertEqual(d[0][0], 5)
        self.assertEqual(d[0][1], 7)
        self.assertEqual(d[0][2], 9)

    def test_combined(self):
        queue = DilatedQueue(max_length=12, num_channels=1)
        e = torch.zeros((1))
        for i in range(30):
            e = e + 1
            queue.enqueue(e)
            d = queue.dequeue(num_deq=3, dilation=4)
            d = d.data
            self.assertEqual(d[0][0], max(i - 7, 0))

'''
class Test_zero_padding(unittest.TestCase):
    def test_end_padding(self):
        x = torch.ones((3, 4, 5))

        p = zero_pad(x, num_pad=5, dimension=0)
        assert p.size() == (8, 4, 5)
        assert p[-1, 0, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=1)
        assert p.size() == (3, 9, 5)
        assert p[0, -1, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=2)
        assert p.size() == (3, 4, 10)
        assert p[0, 0, -1] == 0

    def test_start_padding(self):
        x = torch.ones((3, 4, 5))

        p = zero_pad(x, num_pad=5, dimension=0, pad_start=True)
        assert p.size() == (8, 4, 5)
        assert p[0, 0, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=1, pad_start=True)
        assert p.size() == (3, 9, 5)
        assert p[0, 0, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=2, pad_start=True)
        assert p.size() == (3, 4, 10)
        assert p[0, 0, 0] == 0

    def test_narrowing(self):
        x = torch.ones((2, 3, 4))
        x = x.narrow(2, 1, 2)
        print(x)

        x = x.narrow(0, -1, 3)
        print(x)

        assert False


class Test_wav_files(unittest.TestCase):
    def test_wav_read(self):
        data = wavfile.read('trained_generated.wav')[1]
        print(data)
        # [0.1, -0.53125...
        assert False


class Test_padding(unittest.TestCase):
    def test_1d(self):
        x = Variable(torch.ones((2, 3, 4)), requires_grad=True)

        pad = ConstantPad1d(5, dimension=0, pad_start=False)

        res = pad(x)
        assert res.size() == (5, 3, 4)
        assert res[-1, 0, 0] == 0

        test = gradcheck(ConstantPad1d, x, eps=1e-6, atol=1e-4)
        print('gradcheck', test)

        # torch.autograd.backward(res, )
        res.backward()
        back = pad.backward(res)
        assert back.size() == (2, 3, 4)
        assert back[-1, 0, 0] == 1

    #
    # pad = ConstantPad1d(5, dimension=1, pad_start=True)
    #
    # res = pad(x)
    # assert res.size() == (2, 5, 4)
    # assert res[0, 4, 0] == 0
    #
    # back = pad.backward(res)
    # assert back.size() == (2, 3, 4)
    # assert back[0, 2, 0] == 1


    def test_2d(self):
        pad = ConstantPad2d((5, 0, 0, 0))
        x = Variable(torch.ones((2, 3, 4, 5)))

        res = pad.forward(x)
        print(res.size())
        assert False
'''

def main():
    unittest.main()

if __name__ == '__main__':
    main()
