import tensorflow as tf 
from sentiment_featureset import create_featuresets_and_labels
import numpy as np

train_x, train_y, test_x, test_y = create_featuresets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500 #could change
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100 #batches of 100 features for foreward feeding

# height x width
x = tf.placeholder('float',[None, len(train_x[0])]) #as long as input remains same shape
y = tf.placeholder('float') #28x28 pixels
 
def neural_network_model(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
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

			i = 0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size

			print('Epoch', epoch+1, 'completed out of', n_epochs, 'loss: ', epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1)) #using one_hots to test
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)

'''
Results/Notes
================
1. Using the sentiment dataset of about 10k samples is very small, typically neural nets
need at least 10 million, maybe more, sample sizes for natural language processing
2. because of this Accuracy fluctuated around 50% give or take.
3. For neural nets to work as intended, they need LOTS of data and good computational resources, 
because of this increasing the dataset 10 fold will provide a more accurate model.
'''
