import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import xlrd

'''
Data being used: titanic dataset (titanic.xls)
========================================
Pclass passenger's class (1st-3rd)
survival Survive (0 = no, 1 = yes)
name Name
sex Sex
age Age
sibsp Num of Silings/Spouses aboard
parch Num of parents/children aboard
ticket Ticket Num
fare Passenger's Fare (BPS)
cabin Cabin
embarked Port of Embarkation(C = Cherbourg, Q = Queenstown, S = Southampton)
boat Lifeboat
body Body ID Num
home.dest Home/Destination 
'''
df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True) #this data might taint the outcome 
df.apply(pd.to_numeric, errors='coerce') #coerce / ignore for errors=''
df.fillna(0, inplace=True)

def handle_non_numerical(df):
	columns = df.columns.values

	for column in columns:
		text_digit_vals = {} #ex. {'female', 0}
		def convert_to_int(val):
			return text_digit_vals[val]
		
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist() # all just converting data to numerical
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1
			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numerical(df)

df.drop(['boat'], 1, inplace=True)  #you can tweak the dataset to see what variabes have an impact

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = clf.predict(predict_me) # only for unspervised learning
	if prediction[0] == y[i]:
		correct += 1

print(df.head())
print(correct/len(X))

'''
Since the algorithm is identifying random/arbitary centroids for the clusters,
the accuracy of the model will bounce between 2 numbers.

For Example: accuracy could be 30%, which might sound bad but its really (100-30%) = 70%
and if you run the same model multiple times, the accuracy will hover between 30 and 70, 
because of the randomness in centroid choosing, it might pick 'the wrong centroid' for 
each cluster but the overall algorithm is accurate.
'''

























