'''
Overview:
==========
Convolutional Neural Nets (CNNs) consists:
 - Input -> (Convolutions + Pooling) -> Fully Connected Layer -> Output
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 128 #standard batch size

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# -- Dropout -- : caps the amount of neurons firing, synonmous with the human brain
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32) 

# not nescessary to define functions for these
def conv2d(x, W):
    # moving convolution across image in order to not skip any pixels
    # strides moving 1 pixel at a time
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    # 2D, no depth

def maxpool2d(x):
    # convolution simplifying
    # strides moving 2 by 2 at a time to pervent overlap
    # padding as convolution reaches edge of screen 
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                               5x5 Convolution, 1 input, 32 features (outputs)
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               #                          7x7 image * 64 features with 1024 nodes
               'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes])),}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes])),}

    # can method visualization with numpy
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc']) #rectified linear

    # dropout cap of 80% >
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',n_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        '''
        Unless you have a ton of VRAM, the accuracy will load a massive matrix into your memory,
        so to handle this, you have to compute it in batches else you will run out of memory.
        '''
        accuracy_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
        good = 0
        total = 0
        for i in range(int(mnist.test.num_examples/batch_size)):
            testSet = mnist.test.next_batch(batch_size)
            good += accuracy_sum.eval(feed_dict={x: testSet[0], y: testSet[1], keep_prob: 1.0})
            total += testSet[0].shape[0]
        print("test accuracy %g"%(good/total))
        #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

'''
Results/Notes
==============
 - The final accuracy was 97.7% (0% dropout rate), which is only slightly better than the recurrent network model
 this is primarily due to the size of data being used, RNNs typically get by better with less data,
 and CNNs are superior with larger data sets.
 
 - Accuracy: 96.8% (20% dropout rate)

'''

