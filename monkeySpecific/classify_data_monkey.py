#!/usr/bin/env python
"""string"""

import h5py
import hmax
from hmax.classification import kernel
#from shogun import Kernel, Classifier, Features
from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
import scipy as sp
import numpy as np
from scipy import io
import tables as ta
from sklearn.svm import SVC, LinearSVC
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
from aux_functions import classif_RBF


l_cats = sp.array(['crouch', 'crouchrock', 'drink', 'groom_sit', 'motor', 'move',
                   'rock_sit', 'rock_stand', 'sit', 'sitdown', 'sitturn', 'situp',
                   'stand', 'standdown', 'standfull', 'standfull_walk', 'standup',
                   'swing', 'tic', 'unused', 'walk'], dtype='|S17')


REGULARIZATION_VALUE = 1E4
N_SAMPLES = 15# 571741    %GUZEL SONUC 7 sample, 100 feat gamma=0.000001
N_FEATURES  = 1441 #1000
N_LIM = 50
l_c = [1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2]
l_g = pow(2,np.linspace(-15, -5, 7))
#------------------------------------------------------------------------------#
def getMonkeySplits(table_fname, splitNo, n_samples = N_SAMPLES, n_features = N_FEATURES):

    
    h5_tr = ta.openFile(table_fname + str(splitNo) + '_train.h5', mode = 'r')
    table_tr = h5_tr.root.input_output_data.readout
    
    h5_te = ta.openFile(table_fname + str(splitNo) + '_test.h5', mode = 'r')
    table_te = h5_te.root.input_output_data.readout
    import ipdb; ipdb.set_trace()
    print 'Converting arrays to sp'
    features_train = sp.array(table_tr.cols.features)#[:,:N_LIM]
    labels_train = sp.array(table_tr.cols.label)
    
    features_test = sp.array(table_te.cols.features)#[:,:N_LIM]
    labels_test = sp.array(table_te.cols.label)

#    features_train = sp.array(features_train, dtype = 'uint8')
#    features_test = sp.array(features_test, dtype = 'uint8')
#    labels_train = sp.array(labels_train)
#    labels_test = sp.array(labels_test)
    print 'Converted'
    
    table_tr.flush()
    table_te.flush()
    h5_tr.close()
    h5_te.close()
    print "feature loading completed"
    return features_train , labels_train, features_test, labels_test

#------------------------------------------------------------------------------#
def svm_cla_sklearn(features_train, features_test, labels_train, labels_test):
    """docstring for svm_sklearn"""

    print "zscore features and generating the normalized dot product kernel"
    tic = time.time()
    features_train_prep, mean_f, std_f = features_preprocessing(features_train)
    features_test_prep, mean_f, std_f  = features_preprocessing(features_test, mean_f, std_f)
    #print "time taken to zscore data is:", time.time() - tic , "seconds"
    
    featSize = np.shape(features_train_prep)
    print 'using %d samp, %d feats' % (featSize[0], featSize[1])
    
    
    for c in l_c:
        tic = time.time()
        clf = SVC(C=c) #gamma 1=> 0.027 verdi 10 ==> ~0.03
        clf.fit(features_train_prep, labels_train) #[:1960][:]
        #import ipdb; ipdb.set_trace()
        score = clf.score(features_test_prep, labels_test) #[:13841][:]
        #score = clf.score(features_test_prep, labels_test)
        print "score for C,",c, "is: ", score
        print "time taken:", time.time() - tic, "seconds"
        import ipdb; ipdb.set_trace()

#------------------------------------------------------------------------------#
def svm_cla_sklearn_feat_sel(features_train, features_test, labels_train, labels_test):
    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, RFECV
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import zero_one_loss
    
    #features_train = sp.array(features_train, dtype = 'uint8')
    #features_test = sp.array(features_test, dtype = 'uint8')
    
    print "zscore features"
    tic = time.time()
    features_train, mean_f, std_f = features_preprocessing(features_train)
    features_test, mean_f, std_f  = features_preprocessing(features_test, mean_f, std_f)
    print "time taken to zscore data is:", round(time.time() - tic) , "seconds"
    
    featSize = np.shape(features_train)
    selector = LinearSVC(C=0.0005, penalty="l1", dual=False).fit(features_train, labels_train)

    print 'Starting with %d samp, %d feats, keeping %d' % (featSize[0], featSize[1], (np.shape(selector.transform(features_train)))[1])
    print 'classifying'
    
    features_train = selector.transform(features_train)
    features_test = selector.transform(features_test)
    #import ipdb; ipdb.set_trace()
    mem = Memory(cachedir='tmp')
    classif_RBF2 = mem.cache(classif_RBF)

    c = l_c[0]
    Parallel(n_jobs=7)(delayed(classif_RBF2)(features_train, features_test, labels_train, labels_test, g, c) for g in l_g)
    #import ipdb; ipdb.set_trace()

    print "Starting CONTROL classification for c = ", c
    tic = time.time()
    clf = SVC(C=c)
    clf.fit(features_train, labels_train) #[:1960][:]
    score = clf.score(features_test, labels_test) #[:13841][:]
    print "selected CONTROL score for c = ", c, "is: ", score
    print "time taken:", time.time() - tic, "seconds"

#------------------------------------------------------------------------------#
def features_preprocessing(features, mean_f = None, std_f = None):

    features = sp.array(features, dtype = 'float64')

    if mean_f is None:
        mean_f = features.mean(0)
        std_f  = features.std(0)

    features -= mean_f
    # avoid zero division
    std_f[std_f == 0] = 1
    features /= std_f

    return features, mean_f, std_f

#------------------------------------------------------------------------------#
def main():
    
    parser = argparse.ArgumentParser(description="""This file does this and that \n
            usage: python ./classify_data.py --n_samples 10 --n_features 100 --features_fname ./bla.mat --labels_fname ./bla1.mat""")
    parser.add_argument('--n_features', type=int, default = N_FEATURES, help="""string""")
    parser.add_argument('--n_samples', type=int, default = N_SAMPLES, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""") ##THIS IS THE BASE NAME, PARTS WILL BE ADDED IN THE CODE
    parser.add_argument('--split_no', type=int, default = 1, help="""string""")
    args = parser.parse_args()

    table_fname = args.table_fname
    n_features = args.n_features
    n_samples = args.n_samples
    splitNo = args.split_no
    
    
    features_train , labels_train, features_test, labels_test = getMonkeySplits(table_fname, splitNo, n_samples, n_features)
    #svm_cla_sklearn(features_train, features_test, labels_train, labels_test)
    svm_cla_sklearn_feat_sel(features_train, features_test, labels_train, labels_test)

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

