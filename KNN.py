import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

###Dataset 1
df = np.genfromtxt('DataSet-Release 1/ds1/ds1Info.csv', delimiter=',')
df_alpha_numeric = np.genfromtxt('DataSet-Release 1/ds1/ds1Train.csv', delimiter=',')
df_alpha_numeric_validation = np.genfromtxt('DataSet-Release 1/ds1/ds1Val.csv', delimiter=',')

train_X = df_alpha_numeric[:, :1024]
train_Y = df_alpha_numeric[:, 1024]

# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(train_X, train_Y)

testX = df_alpha_numeric_validation[:, :1024]
targetTest = df_alpha_numeric_validation[:, 1024]

print('dataset 1')
train_results = []
test_results = []
neighbors = list(range(1, 31))

for n in neighbors:
   model = KNeighborsClassifier(n_neighbors=n)
   model.fit(train_X, train_Y)

   train_pred = model.predict(train_X)

   acc = accuracy_score(train_pred, train_Y)
   train_results.append(acc)

   y_predTest = model.predict(testX)

   acc = accuracy_score(targetTest, y_predTest)
   test_results.append(acc)

line1, = plt.plot(neighbors, train_results, 'b', label='Train Accuracy')
line2, = plt.plot(neighbors, test_results,'r', label='Test Accuracy')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.interactive(False)
plt.ylabel('Accuracy')
plt.xlabel('number of neighbors')
plt.show()
#
# predictedTarget = neigh.predict(testX)
# print('Dataset 1')
# print(accuracy_score(predictedTarget, targetTest))

###Dataset 2

# df1 = np.genfromtxt('DataSet-Release 1/ds2/ds2Info.csv', delimiter=',')
# df_greek = np.genfromtxt('DataSet-Release 1/ds2/ds2Train.csv', delimiter=',')
# df_greek_valid = np.genfromtxt('DataSet-Release 1/ds2/ds2Val.csv', delimiter=',')
#
#
# train_GreekX = df_greek[:, :1024]
# train_GreekY = df_greek[:, 1024]
#
# testGreekX = df_greek_valid[:, :1024]
# testGreekTarget = df_greek_valid[:, 1024]
#
# # neigh2 = KNeighborsClassifier(n_neighbors=6)
# # neigh2.fit(train_GreekX, train_GreekY)
# # predictGreek = neigh2.predict(testGreekX)
# # print('Dataset 2')
# # print(accuracy_score(testGreekTarget, predictGreek))
#
# train_results = []
# test_results = []
# neighbors = list(range(1, 31))
#
# for n in neighbors:
#    model = KNeighborsClassifier(n_neighbors=n)
#    model.fit(train_GreekX, train_GreekY)
#
#    train_predGreek = model.predict(train_GreekX)
#
#    acc = accuracy_score(train_predGreek, train_GreekY)
#    train_results.append(acc)
#
#    y_predTestGreek = model.predict(testGreekX)
#
#    acc = accuracy_score(testGreekTarget, y_predTestGreek)
#    test_results.append(acc)
#
# line1, = plt.plot(neighbors, train_results, 'b', label='Train Accuracy')
# line2, = plt.plot(neighbors, test_results,'r', label='Test Accuracy')
#
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.interactive(False)
# plt.ylabel('Accuracy')
# plt.xlabel('number of neighbors')
# plt.show()