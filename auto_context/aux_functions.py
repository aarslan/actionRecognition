import h5py
import hmax
from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
import scipy as sp
import glob
import numpy as np
from scipy import io
import tables as ta
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl


def confidence_par(allLearners,ii, dada):
    thisLab = allLearners[ii]:
    res = np.zeros([dada.shape[0]])
    for jj, thisLearner in enumerate(thisLab):
        for hh, thisEstimator in enumerate(thisLearner):
            #multiply the predictions with the weight of the learner
            res = res+thisEstimator.predict(dada)*thisLearner.estimator_weights_[hh]

    lab_confidence_perii = res
    print "time taken to produce confidence:", round(time.time() - tic,2), "seconds"
    #import ipdb;ipdb.set_trace()
    return lab_confidence_perii, ii
