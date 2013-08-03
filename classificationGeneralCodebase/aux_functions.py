from sklearn.svm import SVC
import time
import scipy as sp
import numpy as np

def classif_RBF(features_train, features_test, labels_train, labels_test, gamma, c):
    print "Starting classification for G=",gamma, ", c = ", c
    tic = time.time()
    clf = SVC(gamma=gamma, C=c)
    clf.fit(features_train, labels_train) #[:1960][:]
    score = clf.score(features_test, labels_test) #[:13841][:]
    print "selected score for G=",gamma, ", c = ", c, "is: ", score
    print "time taken:", round(time.time() - tic), "seconds"
