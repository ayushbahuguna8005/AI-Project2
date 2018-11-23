import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

def tweakAlpha(train_X, train_Y, testX, targetTest):
    alphas = np.linspace(0.1, 5, 20, endpoint=True)
    train_results = []
    test_results = []
    for alpha in alphas:
        naive_bayes_classifier = BernoulliNB(alpha=alpha)
        naive_bayes_classifier.fit(train_X, train_Y)

        train_pred = naive_bayes_classifier.predict(train_X)

        trainAcc = accuracy_score(train_Y, train_pred)
        train_results.append(trainAcc)

        y_predTest = naive_bayes_classifier.predict(testX)

        testAcc = accuracy_score(targetTest, y_predTest)
        test_results.append(testAcc)

    line1, = plt.plot(alphas, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(alphas, test_results, 'r', label='Test Accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.interactive(False)
    plt.ylabel('Accuracy')
    plt.xlabel('Alpha')
    plt.show()


###Dataset 1
df = np.genfromtxt('DataSet-Release 1/ds1/ds1Info.csv', delimiter=',')
df_alpha_numeric = np.genfromtxt('DataSet-Release 1/ds1/ds1Train.csv', delimiter=',')
df_alpha_numeric_validation = np.genfromtxt('DataSet-Release 1/ds1/ds1Val.csv', delimiter=',')

train_X = df_alpha_numeric[:, :1024]
train_Y = df_alpha_numeric[:, 1024]

testX = df_alpha_numeric_validation[:, :1024]
targetTest = df_alpha_numeric_validation[:, 1024]

# tweakAlpha(train_X, train_Y, testX, targetTest)

# naive_bayes_classifier = MultinomialNB(fit_prior=False)
# naive_bayes_classifier = MultinomialNB(fit_prior=True)


naive_bayes_classifier = BernoulliNB(fit_prior=False)
naive_bayes_classifier.fit(train_X, train_Y)

joblib.dump(naive_bayes_classifier, 'nb1classifier.joblib')

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

# tweakAlpha(train_GreekX, train_GreekY, testGreekX, testGreekTarget)

naive_bayes_classifier2 = BernoulliNB(fit_prior=False)
naive_bayes_classifier2.fit(train_GreekX, train_GreekY)

joblib.dump(naive_bayes_classifier2, 'nb2classifier.joblib')

predictGreek = naive_bayes_classifier2.predict(testGreekX)
print('Dataset 2')
print(accuracy_score(testGreekTarget, predictGreek))
