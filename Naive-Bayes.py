import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_csv('DataSet-Release 1/ds1/ds1Info.csv')
df_alpha_numeric = pd.read_csv('DataSet-Release 1/ds1/ds1Train.csv')
df_alpha_numeric_validation = pd.read_csv('DataSet-Release 1/ds1/ds1Val.csv')

df1 = pd.read_csv('DataSet-Release 1/ds2/ds2Info.csv')
df_greek = pd.read_csv('DataSet-Release 1/ds2/ds2Train.csv')
df_greek_valid = pd.read_csv('DataSet-Release 1/ds2/ds2Val.csv')


#test-target split
train_X = df_alpha_numeric.ix[:, :1024]
train_Y = df_alpha_numeric.ix[:, 1024]

classifier = BernoulliNB()
classifier.fit(train_X, train_Y)

testX = df_alpha_numeric_validation.ix[:, :1024]
targetTest = df_alpha_numeric_validation.ix[:, 1024]

predictedTarget = classifier.predict(testX)

print(accuracy_score(targetTest, predictedTarget))

#---------------------------------------------------------------
train_GreekX = df_greek.ix[:, :1024]
train_GreekY = df_greek.ix[:, 1024]

testGreekX = df_greek_valid.ix[:, :1024]
testGreekTarget = df_greek_valid.ix[:, 1024]
classifier.fit(train_GreekX, train_GreekY)
predictGreek = classifier.predict(testGreekX)
print(accuracy_score(testGreekTarget, predictGreek))


#---------------------------------------------------------------
clf = tree.DecisionTreeClassifier()
clf.fit(train_X, train_Y)
clfTarget = clf.predict(testX)
print(accuracy_score(targetTest, clfTarget))

