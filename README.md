# deepdigits

teaching my computer to read handwritten digits with TensorFlow.

ever since AlphaGo took game 1 off Lee Sedol i can't stop reading about
neural networks. figured the only way to actually understand any of this
is to build something myself, so: the classic MNIST digits dataset.
start small, right?

## setup

    pip install -r requirements.txt

(2019 edit: this used to be a wheel url from google's storage bucket,
because tensorflow wasn't even on pypi back then. wild.)

the MNIST data downloads itself into MNIST_data/ the first time a
script runs.

## running it

    python softmax.py
    python cnn.py      # the good one

## plan

- [x] softmax regression (the tutorial one) — 92%
- [x] a real convolutional net — 98.6%!
- [ ] draw my own digits and have it guess them
- [ ] port all this to eager mode / tf 2.0 someday

## results

| model | test accuracy |
| ----- | ------------- |
| softmax regression | 92.1% |
| convnet (2 conv + 2 pool) | 98.6% |
| convnet + dropout | 99.1% |
