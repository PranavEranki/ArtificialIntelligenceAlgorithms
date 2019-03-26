import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
n_lines = 10000

def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos,neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:n_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)
				#creates lexicon for lemmatization
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon) # {'the': 52358}

	lexicon2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50: #tinker with values
			lexicon2.append(w)
			#removes useless words like: 'um' 'or' 'and'
	print(len(lexicon2)) #every input vector is going to have 423 elements to it
	return lexicon2

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:n_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])
			# will create a feature set of one-hot array and a pos/neg label
	return featureset

def create_featuresets_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt', lexicon,[1,0])
	features += sample_handling('neg.txt', lexicon,[0,1])
	random.shuffle(features)
	#neural network requires suffling

	features = np.array(features)
	testing_size = int(test_size*len(features))
	train_x = list(features[:,0][:-testing_size:]) #using all 0th elements for x
	train_y = list(features[:,1][:-testing_size:])

	test_x = list(features[:,0][:-testing_size:])
	test_y = list(features[:,1][:-testing_size:])

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_featuresets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)
'''
Notes
============
1. Most of the time, data in the real world requires some sort of preprocesssing to obtain it
in the desired format (numerical)
2. These methods utilze the NLTK library (amazing lib) in order to format the data for use later
by a neural network of some kind.

Overview
============
1. The raw language data is tokenized (split up) using NLTK (natural language tool kit) by word 
into a lexicon (a tokenized set of words or paragraphs)
2. The raw lexicon is then further refined into a more useful set, getting rid of useless words.
3. Then the samples within the lexicon are distributed into a featureset 
4. The featureset and lexicon are used to create proper data (training and testing subsets)
'''
