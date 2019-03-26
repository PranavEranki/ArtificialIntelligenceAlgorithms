import matplotlib.pyplot as plt 
from matplotlib import style 
import numpy as np 
style.use('ggplot')

class SVM:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1:'r',-1:'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)
	
	def fit(self, data):
		#training // convex optimizing for w and b values
		self.data = data
		# { ||w|| : [w,b]}
		opt_dictionary = {}
		
		transforms = [[1,1],[-1,1],[-1,-1],[1,-1]] #applied to w vector

		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)

		self.max_feature_val = max(all_data)
		self.min_feature_val = min(all_data)
		all_data = None

		#support vector y1(xi*w+b) = 1, or really close to 1 (1.0001)
		step_sizes = [self.max_feature_val * 0.1, self.max_feature_val * 0.01, 
					  self.max_feature_val * 0.001] #after 0.001 it gets kinda expensive
		
		# very very expensive
		b_range_multiple = 5 
		#taking bigger steps with b than w (sake of simplicity, otherwise computing would take forever)
		b_multiple = 5
		latest_optimum = self.max_feature_val*10 #first element of vector W

		for step in step_sizes:
			w = np.array([latest_optimum,latest_optimum])
			optimized = False #due to the convex
			while not optimized:
				#this includes a different stepping for b than it does for w (sake of simplicity)
				for b in np.arange(-1*(self.max_feature_val*b_range_multiple),
										self.max_feature_val*b_range_multiple, step*b_multiple):
					for transformation in transforms:
						w_t = w*transformation
						found_option = True
						#weakest part of the SVM, bc it can only minimize data to a certian degree
						#sequential minimal optimization(SMOs) can fix this a bit, but is still bad for bigdata.
						for i in self.data:
							#must be ran on all data to make sure its optimized.
							#constraint func: yi(xi*w+b) >= 1
							for xi in self.data[i]:
								yi = i
								if not yi*(np.dot(w_t,xi)+b) >= 1:
									found_option = False
									#should be broken once false is found (saves computing)...
						if found_option: 
							opt_dictionary[np.linalg.norm(w_t)] = [w_t,b] #mag of vector

				if w[0] < 0:
					#val = 0 bc vector should be optimized at this point
					optimized = True
					print('Optimized a step.')
				else: 
					w = w - step #subtracting a scalar from the w vector (W[5,5] - <step> = W[5-step,5-step])
			
			norms = sorted([n for n in opt_dictionary]) #vector mags
			opt_choice = opt_dictionary[norms[0]]
			#||w|| : [w,b]
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			latest_optimum = opt_choice[0][0]+step*2 #reseting lat_opt per each step
		for i in self.data:
			for xi in self.data[i]:
				yi = i
				print(xi,':',yi*(np.dot(self.w,xi)+self.b))
				# allows to see if problem is optimized (both classes have a SV of ~1)

	def predict(self, features):
		#sign of func -> scaler value
		#modeling: sign(x*w+b)
		classification = np.sign(np.dot(np.array(features), self.w)+self.b)
		if classification !=0 and self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

		return classification
	
	def visualize(self): #no effect on SVM
		[[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dictionary[i]] for i in data_dictionary]


		def hyperplane(x,w,b,v):
			#hyperplane equation = x*w+b = v
			#func provides hyperplane values
			return (-w[0]*x-b+v) / w[1] 
		data_range = (self.min_feature_val*0.9, self.max_feature_val*1.1)
		hyp_x_min = data_range[0]
		hyp_x_max = data_range[1]

		# positive support vector hyperplane, (x*w+b) = 1
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1) #scalar value given an x
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
		self.ax.plot([hyp_x_min,hyp_x_max], [psv1, psv2], 'k')

		# negative support vector hyperplane, (x*w+b) = -1
		nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1) 
		nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
		self.ax.plot([hyp_x_min,hyp_x_max], [nsv1, nsv2], 'k')

		# decison boundary hyperplane, (x*w+b) = 0
		db1 = hyperplane(hyp_x_min, self.w, self.b, 0) 
		db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
		self.ax.plot([hyp_x_min,hyp_x_max], [db1, db2], 'y--')

		plt.show()


data_dictionary = {-1:np.array([[1,7],
								[2,8], #largest val = 8
								[3,8],]),
				    1:np.array([[5,1],
				    			[6,-1],
				    			[7,3],])} #creates features(X) classes 

support_vector_machine_classifier = SVM()
support_vector_machine_classifier.fit(data=data_dictionary)

prediction_set = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]] 

for p in prediction_set:
	support_vector_machine_classifier.predict(p)
	#once classifier is trained, prediction calculates really fast

support_vector_machine_classifier.visualize()

#To tweak results and computation:
#=================================
#1.) b_range_multiple
# - Increasing range multiple will increase precision at the cost of significantly more computation. 
#2.) step_sizes
# - Increasing the amount of steps and their precision will increase overall precison but also require more computation.
#3.)lastest_optimum and its multiplier 
# - has little effect on the overall precison and computation




