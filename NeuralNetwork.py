import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def max_iter(train_X, train_Y, testX, targetTest):
    iterations = np.linspace(200, 1000, 100, endpoint=True)
    train_results = []
    test_results = []
    for i in iterations:
        # nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(int(neurons)), random_state=0)
        nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(125), random_state=0, max_iter=int(i))
        nn.fit(train_X, train_Y)

        train_pred = nn.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = nn.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(iterations, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(iterations, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('number of iterations')
    plt.show()

def numberOfNeurons(train_X, train_Y, testX, targetTest):
    num_neurons = np.linspace(1, 200, 200, endpoint=True)
    train_results = []
    test_results = []
    for neurons in num_neurons:
        # nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(int(neurons)), random_state=0)
        nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(int(neurons), 125), random_state=0)
        nn.fit(train_X, train_Y)

        train_pred = nn.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = nn.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(num_neurons, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(num_neurons, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('number of neurons')
    plt.show()


# ###Dataset 1
df = np.genfromtxt('DataSet-Release 1/ds1/ds1Info.csv', delimiter=',')
df_alpha_numeric = np.genfromtxt('DataSet-Release 1/ds1/ds1Train.csv', delimiter=',')
df_alpha_numeric_validation = np.genfromtxt('DataSet-Release 1/ds1/ds1Val.csv', delimiter=',')

train_X = df_alpha_numeric[:, :1024]
train_Y = df_alpha_numeric[:, 1024]

testX = df_alpha_numeric_validation[:, :1024]
targetTest = df_alpha_numeric_validation[:, 1024]
# numberOfNeurons(train_X, train_Y, testX, targetTest)

# max_iter(train_X, train_Y, testX, targetTest)

classifier = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(110), random_state=0, max_iter=440)
classifier.fit(train_X, train_Y)

joblib.dump(classifier, 'nn1classifier.joblib')

predictedTarget = classifier.predict(testX)
print('Dataset1 training accuracy')
print(accuracy_score(predictedTarget, targetTest))


###Dataset 2
df1 = np.genfromtxt('DataSet-Release 1/ds2/ds2Info.csv', delimiter=',')
df_greek = np.genfromtxt('DataSet-Release 1/ds2/ds2Train.csv', delimiter=',')
df_greek_valid = np.genfromtxt('DataSet-Release 1/ds2/ds2Val.csv', delimiter=',')

train_GreekX = df_greek[:, :1024]
train_GreekY = df_greek[:, 1024]

testGreekX = df_greek_valid[:, :1024]
testGreekTarget = df_greek_valid[:, 1024]

# numberOfNeurons(train_GreekX, train_GreekY, testGreekX, testGreekTarget)
# max_iter(train_GreekX, train_GreekY, testGreekX, testGreekTarget)
classifier2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(125), random_state=0, max_iter=250)
classifier2.fit(train_GreekX, train_GreekY)

joblib.dump(classifier2, 'nn2classifier.joblib')

predictGreek = classifier2.predict(testGreekX)
print('Dataset 2')
print(accuracy_score(testGreekTarget, predictGreek))
