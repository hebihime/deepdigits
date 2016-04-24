# deepdigits

teaching my computer to read handwritten digits with TensorFlow.

ever since AlphaGo took game 1 off Lee Sedol i can't stop reading about
neural networks. figured the only way to actually understand any of this
is to build something myself, so: the classic MNIST digits dataset.
start small, right?

## setup

TensorFlow isn't on pip yet — you install a wheel straight from google's
storage bucket (this is the linux cpu one for python 2.7):

    sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

the MNIST data downloads itself into MNIST_data/ the first time a
script runs.

## running it

    python softmax.py
    python cnn.py      # the good one

## plan

- [x] softmax regression (the tutorial one) — 92%
- [x] a real convolutional net — 98.6%!
- [ ] draw my own digits and have it guess them

## results

| model | test accuracy |
| ----- | ------------- |
| softmax regression | 92.1% |
| convnet (2 conv + 2 pool) | 98.6% |
