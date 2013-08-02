#!/usr/bin/env python
"""string"""

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
from sklearn import preprocessing
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl
from pymatlab.matlab import MatlabSession
import classify_data_monkey as mk
import auto_context_demo as ac

N_LIM = 1441
N_ESTIM = 15
learning_rate = 0.00002
Sample_N = 200
N_RUNS = 10
N_LAB = 15
CLF = 'adaboost'#'randomforest'

#------------------------------------------------------------------------------#

def main():
    confidence_test = np.random.random_integers(1, 40000, (4000,5))

    tic = time.time()
    new_CF_35 = ac.get_contextual_matlab(confidence_test, 35)
    new_CF_75 = ac.get_contextual_matlab(confidence_test, 75)
    new_CF_35 = new_CF_35[:, np.sum(new_CF_35,axis= 0) != 0]
    new_CF_75 = new_CF_75[:,np.sum(new_CF_35,axis= 0) != 0]
    
    print "time taken old way:", round(time.time() - tic,2), "seconds"
    
    tic = time.time()
    old_CF_35 = ac.get_contextual(confidence_test, 35)
    old_CF_75 = ac.get_contextual(confidence_test, 75)
    print "time taken new way:", round(time.time() - tic,2), "seconds"
    import ipdb;ipdb.set_trace()
    plt.matshow(new_CF_35, aspect = 'auto')
    
    plt.matshow(old_CF_35, aspect = 'auto')
    plt.show()
    
    import ipdb;ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()
