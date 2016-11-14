from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# copy pasted the whole graph from cnn.py so the checkpoint lines up.
# there is definitely a better way to share this but it works (mostly)

x = tf.placeholder(tf.float32, [None, 784])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(
    tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                 padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                 padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y = tf.nn.softmax(y_logits)

saver = tf.train.Saver()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()
saver.restore(sess, 'checkpoints/digits-final')

image = mnist.test.images[0:1]
guess = sess.run(tf.argmax(y, 1), feed_dict={x: image, keep_prob: 1.0})
print('the network says: %d' % guess[0])
print('the label says:   %d' % mnist.test.labels[0].argmax())

# TODO: load a png i drew myself, that was the whole point of this.
# tried a couple screenshots and it guesses wrong a lot — i think my
# digits aren't centered the way mnist ones are?
