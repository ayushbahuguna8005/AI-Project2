import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

def maxDepth(train_X, train_Y, testX, targetTest):
    max_depths = np.linspace(1, 50, 50, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        dt = tree.DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(train_X, train_Y)

        train_pred = dt.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = dt.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(max_depths, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('Tree depth')
    plt.show()

def minSampleSplits(train_X, train_Y, testX, targetTest):
    min_samples_splits = np.linspace(0.1, 0.5, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        dt = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
        dt.fit(train_X, train_Y)

        train_pred = dt.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = dt.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('min samples split')
    plt.show()

def minSampleLeaf(train_X, train_Y, testX, targetTest):
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        dt = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        dt.fit(train_X, train_Y)

        train_pred = dt.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = dt.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('min samples leaf')
    plt.show()


def maxFeatures(train_X, train_Y, testX, targetTest):
    max_features = list(range(1, train_X.shape[1]))
    train_results = []
    test_results = []
    for max_feature in max_features:
        dt = tree.DecisionTreeClassifier(max_features=max_feature)
        dt.fit(train_X, train_Y)

        train_pred = dt.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = dt.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(max_features, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(max_features, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('max features')
    plt.show()


###Dataset 1
df = np.genfromtxt('DataSet-Release 1/ds1/ds1Info.csv', delimiter=',')
df_alpha_numeric = np.genfromtxt('DataSet-Release 1/ds1/ds1Train.csv', delimiter=',')
df_alpha_numeric_validation = np.genfromtxt('DataSet-Release 1/ds1/ds1Val.csv', delimiter=',')

train_X = df_alpha_numeric[:, :1024]
train_Y = df_alpha_numeric[:, 1024]

testX = df_alpha_numeric_validation[:, :1024]
targetTest = df_alpha_numeric_validation[:, 1024]

# maxDepth(train_X, train_Y, testX, targetTest)
# minSampleSplits(train_X, train_Y, testX, targetTest)
# minSampleLeaf(train_X, train_Y, testX, targetTest)
# maxFeatures(train_X, train_Y, testX, targetTest)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(train_X, train_Y)
joblib.dump(clf, 'dt1classifier.joblib')

# scores = cross_val_score(clf, train_X, train_Y, cv = 10)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Above two lines generated output: Accuracy: 0.28 (+/- 0.06)

clfTarget = clf.predict(testX)
# with open('DSTDS1.csv', 'wb') as file:

# x = np.genfromtxt('DSTDS1.csv', delimiter=',')
# print(x)
print('Dataset1')
print(accuracy_score(clfTarget, targetTest))


###Dataset 2
df1 = np.genfromtxt('DataSet-Release 1/ds2/ds2Info.csv', delimiter=',')
df_greek = np.genfromtxt('DataSet-Release 1/ds2/ds2Train.csv', delimiter=',')
df_greek_valid = np.genfromtxt('DataSet-Release 1/ds2/ds2Val.csv', delimiter=',')

train_GreekX = df_greek[:, :1024]
train_GreekY = df_greek[:, 1024]

testGreekX = df_greek_valid[:, :1024]
testGreekTarget = df_greek_valid[:, 1024]

# maxDepth(train_GreekX, train_GreekY, testGreekX, testGreekTarget)
# minSampleSplits(train_GreekX, train_GreekY, testGreekX, testGreekTarget)
# minSampleLeaf(train_GreekX, train_GreekY, testGreekX, testGreekTarget)
# maxFeatures(train_GreekX, train_GreekY, testGreekX, testGreekTarget)

clf2 = tree.DecisionTreeClassifier(criterion='entropy')
# # scores = cross_val_score(clf2, train_GreekX, train_GreekY, cv = 10)
# # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# # Above two lines generated output: Accuracy: 0.74 (+/- 0.03)
#
clf2.fit(train_GreekX, train_GreekY)
joblib.dump(clf2, 'dt2classifier.joblib')
clfTarget = clf2.predict(testGreekX)
print('Dataset 2')
print(accuracy_score(testGreekTarget, clfTarget))

