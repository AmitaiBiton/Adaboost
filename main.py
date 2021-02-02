import numpy as np
from sklearn.datasets import load_digits, make_friedman1
import  random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.preprocessing import StandardScaler
class Decision_stump:
    threshold = 0
    check = 0
    def __init__(self,plus_average,minus_average):
        self.threshold = (plus_average+minus_average)/2
        if plus_average > minus_average:
            self.check = 1
        else:
            self.check = -1
    def decide(self,value):
        if self.check > 0:
            if value > self.threshold:
                return 1
            else:
                return -1
        else:
            if value < self.threshold:
                return 1
            else:
                return -1



# return the avarage for class 0ne and avarage for class two per feature
def find_threshold(X,y):
    plus_count = 0
    minus_count = 0
    plus_amount = 0
    minus_amount = 0
    for i in range(len(y)):
        if y[i] == 1:
            plus_count = plus_count + X[i]
            plus_amount = plus_amount + 1
        else:
            minus_count = minus_count + X[i]
            minus_amount = minus_amount + 1
    return Decision_stump(plus_count/plus_amount, minus_count/minus_amount)


def adaboost(X,y,h,wrong_preds):
    w = []
    H = []
    for i in X:
        w.append(1/len(X))
    for i in range(10):
        errors = []
        for j in range(len(h)):
            error = 0
            for k in range(len(X)):
                if k in wrong_preds[j]:
                    error = error + w[k]
            errors.append(error)
        min_error = min(errors)
        min_error_index = errors.index(min_error)
        alpha = (np.log((1-min_error)/min_error)/2)
        H.append((min_error_index , alpha))
        for k in range(len(X)):
            if k in wrong_preds[min_error_index]:
                w[k] = (w[k]/min_error)/2
            else:
                w[k] = (w[k]/(1-min_error))/2
    return H


# predict Y using adaboost
# after we achieve H_Dict
# using H(x)  = sign(alpha*h(x)+.....)
def predict(H,h,X):
    preds = [0]*len(X)
    for index,alpha in H:
        for xi in range(len(X.T[index])):
            preds[xi] = preds[xi] + alpha* h[index].decide(X.T[index][xi])
    for i in range(len(preds)):
        preds[i] = np.sign(preds[i])
    return preds

# return the data only for class 0 and 1
def getDataFor2Classes(X ,Y) :
    cur_X = []
    cur_y = []
    for i in range(len(Y)):
        if y_digits[i] < 2:
            if Y[i] == 0:
                Y[i] = -1
            cur_X.append(X[i])
            cur_y.append(Y[i])
    X = np.array(cur_X)
    y = np.array(cur_y)
    return X , y

# build H by calculate avarge = avg0 + avg1 /2 for any feature
# and H will hols 64 places for 64 feature
def buildH(X ,y):
    h = []
    for i in range(len(X.T)):
        h.append(find_threshold(X.T[i], y))
    return h


# return the index that H have mistake on
# list of  list
# list[0] hold the mistake on feature 0 that H "say class 0 and y =1"
def H_Mistake(h , X_train , y_train):
    h_mistake = []
    for i in range(len(h)):
        list = []
        for j in range(len(X_train.T[i])):
            if y_train[j] != h[i].decide(X_train.T[i][j]):
                list.append(j)
        h_mistake.append(list)
    return h_mistake

# predict y by using one h only to build vector Y
# using sign to know if 1 or -1
def calculateyByH(X_test , h):
    preds = []
    for row in X_test:
        list = []
        for i in range(len(row)):
            list.append(h[i].decide(row[i]))
        preds.append(np.array(list))
    Y_pred = []
    for row in preds:
        pred = np.sign(np.sum(row))
        if pred == 0:
            pred = 1
        Y_pred.append(pred)
    return Y_pred








if __name__ == '__main__':


    X_digits, y_digits = load_digits(return_X_y=True)

    newX, newY = getDataFor2Classes(X_digits, y_digits)
    sc = StandardScaler()
    newX = sc.fit_transform(newX)  # normalize data
    X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3, random_state=45)

    # -------------STEP A-------------#
    h = buildH(X_train , y_train)

    # -------------STEP B-------------#
    h_mistake = H_Mistake(h , X_train ,y_train)

    # -------------STEP C-------------#
    y_perd = calculateyByH(X_test ,h)
    confusion_matrix = pd.crosstab(y_test, np.array(y_perd), rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.title("H - Equal Weights" + " \n Accuracy: {}".format(metrics.accuracy_score(y_test, y_perd)))
    plt.show()
    # -------------STEP D-------------#

    H_dict = adaboost(X_train, y_train, h, h_mistake)

    fin_preds = predict(H_dict, h, X_test)
    fin_preds = np.array(fin_preds)
    confusion_matrix = pd.crosstab(y_test, fin_preds, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.title("H - Adaboost" + " \n Accuracy: {}".format(metrics.accuracy_score(y_test, fin_preds)))
    plt.show()
