import numpy as np
import pandas as pd
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


testX = df_alpha_numeric_validation.ix[:, :1024]
targetTest = df_alpha_numeric_validation.ix[:, 1024]

#---------------------------------------------------------------
# max_depths = np.linspace(1, 32, 32, endpoint=True)
# min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

clf = tree.DecisionTreeClassifier(criterion="gini")
# for min_samples_split in min_samples_splits:
#    dt = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
#    dt.fit(train_X, train_Y)
# clf = tree.DecisionTreeClassifier(max_depth=max_depths)
clf.fit(train_X, train_Y)
clfTarget = clf.predict(testX)
print(accuracy_score(targetTest, clfTarget))


train_GreekX = df_greek.ix[:, :1024]
train_GreekY = df_greek.ix[:, 1024]

testGreekX = df_greek_valid.ix[:, :1024]
testGreekTarget = df_greek_valid.ix[:, 1024]

clf.fit(train_GreekX, train_GreekY)
clfTarget = clf.predict(testGreekX)
print(accuracy_score(testGreekTarget, clfTarget))