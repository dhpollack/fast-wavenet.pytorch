# fast-wavenet.pytorch
A PyTorch implementation of fast-wavenet

[fast-wavenet paper](https://arxiv.org/abs/1611.09482)

[tensorflow fast-wavenet implementation](https://github.com/tomlepaine/fast-wavenet)

[yesno dataset](http://openslr.org/1/)

### Notes

This repo is currently incomplete, although I do hope to get back to working on this.  Notably, I don't have an autoregressive fast forward function.

I created a [similar repo](https://github.com/dhpollack/bytenet.pytorch) for bytenet, which is a predecessor to WaveNet.  This repo does have an autoregressive forward function.

### Testing

```sh
python -m test.layers_test 
```
