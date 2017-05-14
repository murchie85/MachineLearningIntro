#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = svm.SVC(kernel="linear")

print("attempting to fit")
clf.fit(features_train, labels_train)

print("attempting to predict")
predict = clf.predict(features_test)

print("working out accuracy")
acc = accuracy_score(predict, labels_test)
def submitAccuracy():
    return acc
print(acc)





#########################################################


