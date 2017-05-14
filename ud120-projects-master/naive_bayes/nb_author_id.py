#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess


import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# the classifier
clf = GaussianNB()


# -------------------TRAIN-------------------
t0 = time() #START TIMER
clf.fit(features_train, labels_train) #FIT
print "training time:", round(time()-t0, 3), "s" #PRINT

GaussianNB(priors=None)


# -------------------PREDICT-------------------
t0 = time() #START TIMER
pred = clf.predict(features_test)
print (pred) #TI
print "Predicting Time:", round(time()-t0, 3), "s" #PRINT


accuracy = accuracy_score(pred, labels_test)

print ("accuracy is ", accuracy)






#########################################################


