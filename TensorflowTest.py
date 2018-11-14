import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

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

#######

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X.values, train_Y.values, epochs=100)

test_loss, test_acc = model.evaluate(testX, targetTest)

print('Accuracy: ',test_acc)

# testX = df_alpha_numeric_validation.ix[:, :1024]
# targetTest = df_alpha_numeric_validation.ix[:, 1024]