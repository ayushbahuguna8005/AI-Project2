import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

###Dataset 1
df = np.genfromtxt('DataSet-Release 1/ds1/ds1Info.csv', delimiter=',')
df_alpha_numeric = np.genfromtxt('DataSet-Release 1/ds1/ds1Train.csv', delimiter=',')
df_alpha_numeric_validation = np.genfromtxt('DataSet-Release 1/ds1/ds1Val.csv', delimiter=',')

train_X = df_alpha_numeric[:, :1024]
train_Y = df_alpha_numeric[:, 1024]

naive_bayes_classifier = BernoulliNB()
naive_bayes_classifier.fit(train_X, train_Y)

testX = df_alpha_numeric_validation[:, :1024]
targetTest = df_alpha_numeric_validation[:, 1024]

predictedTarget = naive_bayes_classifier.predict(testX)
print('Dataset1')
print(accuracy_score(predictedTarget, targetTest))

###Dataset 2

df1 = np.genfromtxt('DataSet-Release 1/ds2/ds2Info.csv', delimiter=',')
df_greek = np.genfromtxt('DataSet-Release 1/ds2/ds2Train.csv', delimiter=',')
df_greek_valid = np.genfromtxt('DataSet-Release 1/ds2/ds2Val.csv', delimiter=',')

train_GreekX = df_greek[:, :1024]
train_GreekY = df_greek[:, 1024]

testGreekX = df_greek_valid[:, :1024]
testGreekTarget = df_greek_valid[:, 1024]

naive_bayes_classifier.fit(train_GreekX, train_GreekY)
predictGreek = naive_bayes_classifier.predict(testGreekX)
print('Dataset 2')
print(accuracy_score(testGreekTarget, predictGreek))
