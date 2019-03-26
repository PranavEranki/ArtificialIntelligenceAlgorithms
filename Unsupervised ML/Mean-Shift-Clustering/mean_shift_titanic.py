import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import MeanShift
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
original_df = pd.DataFrame.copy(df) #only way to copy data, will be used to compare data

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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan 

for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i] #refs rows in df

n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate
    
print(survival_rates)
#meanShift does have an air of randomness, so results on same data might actually yield different results









