'''
Overview:
=========
 - tflearn is a tensorflow lib that does high level abstraction layers, similar to Keras
 - extremly simple implementation of very useful deep learning models that would otherwise take
 hours to make.
 - goal of tflearn, keras, sk-flow is all to make simple, concise code for deep learning

 Model: Convolutional Neural Net (CNN) with tflearn
'''


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

#loading data
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

#shaping/formatting data
X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

#input layer
convnet = input_data(shape=[None, 28, 28, 1], name='input')

#convolution 1 (hidden layer 1)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

#convolution 2 (hidden layer 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

#fully connected layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

#output layer and cost+optimization 
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

#creating model
model = tflearn.DNN(convnet)


#training model
model.fit({'input': X},{'targets': Y}, n_epoch=10, validation_set=({'input': X},{'targets': Y}), snapshot_step=500, 
	show_metric=True, run_id='mnist')

#saving only weights here >
model.save('cnn_tflearn.model')

#Done. in 30 lines of code!

# AFTER training, the model can be used to predict new data
model.load('cnn_tflearn.model')
print(model.predict([test_x[1]]))

# Final Accuracy: 98.9%