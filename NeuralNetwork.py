import numpy as np

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

###Dataset 1
df = np.genfromtxt('DataSet-Release 1/ds1/ds1Info.csv', delimiter=',')
df_alpha_numeric = np.genfromtxt('DataSet-Release 1/ds1/ds1Train.csv', delimiter=',')
df_alpha_numeric_validation = np.genfromtxt('DataSet-Release 1/ds1/ds1Val.csv', delimiter=',')

train_X = df_alpha_numeric[:, :1024]
train_Y = df_alpha_numeric[:, 1024]

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(513), random_state=0)
classifier.fit(train_X, train_Y)

trainPred = classifier.predict(train_X)
print('Dataset1 training accuracy')
print(accuracy_score(predictedTarget, targetTest))

testX = df_alpha_numeric_validation[:, :1024]
targetTest = df_alpha_numeric_validation[:, 1024]


predictedTarget = classifier.predict(testX)
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

classifier2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(155, 54), random_state=0)
classifier2.fit(train_GreekX, train_GreekY)
predictGreek = classifier2.predict(testGreekX)
print('Dataset 2')
print(accuracy_score(testGreekTarget, predictGreek))
