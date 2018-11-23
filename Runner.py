import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def read_val_File(dataset):
    if dataset == 1:
        val = np.genfromtxt('DataSet-Release 2/ds1/ds1Val.csv', delimiter=',')
    else:
        val = np.genfromtxt('DataSet-Release 2/ds2/ds2Val.csv', delimiter=',')
    return val


def read_test_File(dataset):
    if dataset == 1:
        test = np.genfromtxt('DataSet-Release 2/ds1/ds1Test.csv', delimiter=',')
    else:
        test = np.genfromtxt('DataSet-Release 2/ds2/ds2Test.csv', delimiter=',')
    return test

def write_Prediction_To_File(predicted,fileName):
    with open(fileName, 'w') as f:
        for i in range(len(predicted)):
            f.write('%d,%d\n' % (i + 1, predicted[i]))

while True:
    option = input('1 for Naive Bayes Dataset 1'
                   '\n2 for Naive Bayes Dataset 2'
                   '\n3 for Decision Tree Dataset 1'
                   '\n4 for Decision Tree Dataset 2'
                   '\n5 for Neural Networks Dataset 1'
                   '\n6 for Neural Networks Dataset 2'
                   '\n7 for exit\n')
    if option == '1':
        print('option 1')
        nb1 = joblib.load('nb1classifier.joblib')
        val = read_val_File(1)
        test = read_test_File(1)
        predictionVal = nb1.predict(val[:, :1024])
        predictionTest = nb1.predict(test)
        print('Accuracy: ', accuracy_score(predictionVal, val[:, 1024]))
        write_Prediction_To_File(predictionVal, 'ds1Val-nb.csv')
        write_Prediction_To_File(predictionTest, 'ds1Test-nb.csv')

    elif option == '2':
        print('option 2')
        nb2 = joblib.load('nb2classifier.joblib')
        val = read_val_File(2)
        test = read_test_File(2)
        predictionVal = nb2.predict(val[:, :1024])
        predictionTest = nb2.predict(test)
        print('Accuracy: ', accuracy_score(predictionVal, val[:, 1024]))
        write_Prediction_To_File(predictionVal, 'ds2Val-nb.csv')
        write_Prediction_To_File(predictionTest, 'ds2Test-nb.csv')



    elif option == '3':
        print('option 3')
        dt1 = joblib.load('dt1classifier.joblib')
        val = read_val_File(1)
        test = read_test_File(1)
        predictionVal = dt1.predict(val[:, :1024])
        predictionTest = dt1.predict(test)
        print('Accuracy: ', accuracy_score(predictionVal, val[:, 1024]))
        write_Prediction_To_File(predictionVal, 'ds1Val-dt.csv')
        write_Prediction_To_File(predictionTest, 'ds1Test-dt.csv')




    elif option == '4':
        print('option 4')
        dt2 = joblib.load('dt2classifier.joblib')
        val = read_val_File(2)
        test = read_test_File(2)
        predictionVal = dt2.predict(val[:, :1024])
        predictionTest = dt2.predict(test)
        print('Accuracy: ', accuracy_score(predictionVal, val[:, 1024]))
        write_Prediction_To_File(predictionVal, 'ds2Val-dt.csv')
        write_Prediction_To_File(predictionTest, 'ds2Test-dt.csv')



    elif option == '5':
        print('option 5')
        nn = joblib.load('nn1classifier.joblib')
        val = read_val_File(1)
        test = read_test_File(1)
        predictionVal = nn.predict(val[:, :1024])
        predictionTest = nn.predict(test)
        print('Accuracy: ', accuracy_score(predictionVal, val[:, 1024]))
        write_Prediction_To_File(predictionVal, 'ds1Val-3.csv')
        write_Prediction_To_File(predictionTest, 'ds1Test-3.csv')


    elif option == '6':
        print('option 6')
        nn = joblib.load('nn2classifier.joblib')
        val = read_val_File(2)
        test = read_test_File(2)
        predictionVal = nn.predict(val[:, :1024])
        predictionTest = nn.predict(test)
        print('Accuracy: ', accuracy_score(predictionVal, val[:, 1024]))
        write_Prediction_To_File(predictionVal, 'ds2Val-3.csv')
        write_Prediction_To_File(predictionTest, 'ds2Test-3.csv')

    elif option == '7':
        print('Exiting')
        break
    else:
        print('Wrong choice')