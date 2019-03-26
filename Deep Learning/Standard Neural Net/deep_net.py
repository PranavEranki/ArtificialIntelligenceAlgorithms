import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#nmist dataset: 60k of 28x28 pixel images of handwritten digits

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) #one comp will be on
'''
One_hot: 10 classes (0-9 digits)
===============================
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500 #could change
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 #batches of 100 features for foreward feeding

# height x width
x = tf.placeholder('float',[None, 784]) #as long as input remains same shape
y = tf.placeholder('float') #28x28 pixels
 
def neural_network_model(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	
	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases': tf.Variable(tf.random_normal([n_classes]))}
	
	# model: (input_data * weights) + biases

	#make sure syntax is correct here >
	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1) #activation func

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	#using cross entropy with logits as the cost function, which will calculate the difference between out and intented 
	
	# minimizing cost function, default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost) # synonomous with stochastic gradient descent

	#cycles of feed foreward and backpropagation
	n_epochs = 10 #becareful if slow computer
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size) #part of helper funcs in TF
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', n_epochs, 'loss: ', epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1)) #using one_hots to test
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

'''
Model Summary
=================
1. Input data of 28x28 pixels is reshaped to 784 input nodes
2. a set of weights and biases are applied to the input data
3. based on (input_data * weights) + biases, an activation function within
each hidden layer decides if a neuron/node will fire or not.
4. data now reaches the output layer 
5. a cost function determines how much 'loss' or difference there is between the output
and the intended output.
6. an optimiziation function then uses backpropagation to adjust weights and biases accordingly
7. The epoch repeats until cost fucntion or 'loss' is minimized to an acceptable value.

Thanks to tensorflow, this whole model was done in less than 100 lines of code.

Results
==============
1. With this model and 10 epochs, final accuracy was ~95%
'''
