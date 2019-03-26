import pandas as pd 
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle 

style.use('ggplot')

#df: dataframe
#Quandl: online data service
df = quandl.get('WIKI/GOOGL') #working with google's stocks ('WIKI/GOOGL')

#print(df.head()) // raw data set
#point where we determine 'good' features like: open, high, low ... etc
#make features as simply yet effective as possible // find a relationship to model
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']] #adjusted for stock splits
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100 #calculating percent volatility (doesnt care about abt * 100)
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#			price 			x 			x 			x  		
df = df[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']] #features, later pointed out as "not that good feat., rather simple"
# adj. close will not be a label! *i think*
forecast_col = 'Adj. Close' #you can change this later...
df.fillna(-99999, inplace=True) #you cant work with NAN data (not a number) but you shouldn't scrifice data if you can

forecast_out = int(math.ceil(0.01*len(df))) #rounds everything up to the nearest whole, perferable in integer // 0.1 uses data to predict 10 days ago
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


#features: X, labels: y
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) #scalling X before feeding to clasifier (new values, scale them along side other values, but does add to processing time)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) #20% is testing data (eval data)

#clf = LinearRegression(n_jobs=1) #algorithim or classifer // you should look at algoithims documentation i.e. n_jobs or threading
#clf.fit(X_train, y_train)
#with open('linearregression.pickle','wb') as f:
#	pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test) #evaluation data?
forecast_set = clf.predict(X_lately) 
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan 

last_date = df.iloc[-1].name #dates are not features 
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day #next day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day												#provides data visual and dates x-axis
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




















