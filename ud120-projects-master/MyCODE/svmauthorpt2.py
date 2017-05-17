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
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#only 1% of the features, but over 88% the performance?  Not too shabby!
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel='rbf')
#FIT AND TRAIN
print("attempting to fit")
t0 = time() #START TIMER
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" #PRINT

#PREDICT
print("attempting to predict")
t0 = time() #START TIMER
predict = clf.predict(features_test)
print "Predicting Time:", round(time()-t0, 3), "s" #PRINT

#ACCURACY
print("working out accuracy")
acc = accuracy_score(predict, labels_test)
def submitAccuracy():
    return acc
print(acc)





#########################################################


