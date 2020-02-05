import tensorflow as tf
import numpy as np

# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# create placeholders
x = tf.placeholder(tf.float32,
                   shape=(None, D_in))
y = tf.placeholder(tf.float32,
                   shape=(None, D_out))

# create Variables for the weights
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# setup training
learning_rate = 1e-6

# forward pass
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# compute loss
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# compute gradient of the loss
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# backward pass
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# execute the graph
with tf.Session() as sess:
    # run the graph once to initialize
    sess.run(tf.global_variables_initializer())

    # create numpy arrays holding 
    # the actual data for x and y
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for t in range(500):
        # execute the graph many times
        loss_value, _, _ = sess.run(
            [loss, new_w1, new_w2],
            feed_dict={x: x_value,
                       y: y_value})
        if t % 100 == 99:
            print(t, loss_value)
