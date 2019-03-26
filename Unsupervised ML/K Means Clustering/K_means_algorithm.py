import matplotlib.pyplot as plt 
from matplotlib import style 
style.use('ggplot')
import numpy as np 

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1, 0.6],[9,11]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

colors = ["g","r","c","b","k","o"]

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		'''
		k = num of clusters/groups
		tol = (tolerance): how much the centroids move
		max_iter = max num of times the algo with iterate
		'''
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]
			#you could randomize but it ultimatly shoudn't matter
		for i in range(self.max_iter):#you can tweak exact iterations here
			self.classifications = {} #keys are the centoids, changes everytime centroid changes
			
			for i in range(self.k):
				self.classifications[i] = [] #values will be the feature-sets contained 
			for featureset in X: #Note: X is in place of data, it should be data that was passed in earlier
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)
			prev_centroids = dict(self.centroids) #for tolerance 

			for classification in self.classifications:
				pass
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100) > self.tol:
					optimized = False
			if optimized:
				break


	def predict(self, data):
		#clustering so, passing in same data that algo was trained against
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k",
		s=150, linewidths=5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

unknown_data = np.array([[1,3],[5,9],[1,7],[2,9],[6,3]])

for unknown in unknown_data:
	classification = clf.predict(unknown)
	plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)


plt.show()










