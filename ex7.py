import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def add_layer(inputs, in_size, out_size, nlayers, activation_function=None):
    layers_name="layer%s" % nlayers
    with tf.name_scope(layers_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layers_name+"/Weights", Weights)
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layers_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases


        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layers_name + "/outputs", outputs)
    return outputs

x_data = np.linspace(-1,1,300, dtype=np.float)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_in")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_in")



l1 = add_layer(xs, 1, 10, nlayers=1, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, nlayers=2, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()

write = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        write.add_summary(rs, i)



