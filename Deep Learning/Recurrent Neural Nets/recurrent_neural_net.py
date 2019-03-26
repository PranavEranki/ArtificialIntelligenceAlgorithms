'''
Overview
==========
 - The RNN (Recurent Neural Net) most commonly uses LSTM cells, which is itself a type
 of simple RNN. It takes in an input and generates and output, but then recures the output
 back through the cell and continues per epoch. So basically the outputs recure into inputs
 using an activation function of forget gates, add gates, and output gates.
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_epochs = 3 #ideally 10
n_classes = 10
batch_size = 128

# mnist data: uses 28x28
chunk_size = 28
n_chunks = 28

rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    x= tf.transpose(x, [1,0,2]) #data is being reformatted to what the rnn_cell wants
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size) #LSTM cell that recures for the rnn_size
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost) #with default learning rate 0.001
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',n_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)

'''
Results/Notes
==============
 - Accuracy with 3 epochs was 97.2%, this result is much more efficent
 than the traditional multilayer perceptron (the standard neural net) by
 doing less epochs with similar accuracy

- Most of the code here is the same as deep-net.py (multilayer perceptron), with
the exception of reformatting data around to work with LSTM cells and the RNN.
'''