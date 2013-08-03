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
#from pymatlab.matlab import MatlabSession
import classify_data_monkey as mk
import auto_context_demo as ac

N_ESTIM = 10
learning_rate = 1.
Sample_N = 300
N_RUNS = 5
N_LAB = 35 #35
CLF = 'adaboost'#'randomforest' #
N_FEATURES = 1441

#------------------------------------------------------------------------------#

def strip_features(level, features):
    
    import platform
    if platform.node() != 'g6':
        mat_path = '/Users/aarslan/Desktop/myFeats.mat'
    else:
        mat_path = '/home/aarslan/prj/data/monkey_new_ClassifData/myFeats.mat'

    
    ff = sp.io.loadmat(mat_path)
    all_sets = ff.keys()
    all_sets.sort()
    all_sets = all_sets[3:]
    take = np.squeeze(ff[str(all_sets[level])])
    features = features[:,np.append(take, range(-23,-1))]
    print 'stripped data have now ', str(features.shape[1]), ' features'
    return features


#------------------------------------------------------------------------------#

def main():
    parser = argparse.ArgumentParser(description="""This file does this and that """)
    parser.add_argument('--table_path', type=str, help="""string""") ##THIS IS THE BASE NAME, PARTS WILL BE ADDED IN THE CODE
    parser.add_argument('--mat_path', type=str, default = '0', help="""string""")
    parser.add_argument('--split_no', type=int, default = 1, help="""string""")
    args = parser.parse_args()
    
    table_path = args.table_path
    mat_path = args.mat_path
    splitNo = args.split_no
    
    orig_feats , orig_labels, test_feats, test_labels = mk.get_monkey_splits_lim(table_path, splitNo, 1000,
                                                                                 N_FEATURES,
                                                                                 contig_labels = True, n_lab = N_LAB)
    #import ipdb; ipdb.set_trace()
    
    #orig_labels = mk.groupLabels(orig_labels)
    #test_labels = mk.groupLabels(test_labels)
    
    orig_feats = strip_features(7, orig_feats)
    test_feats = strip_features(7, test_feats)
    #import ipdb; ipdb.set_trace()
    le = preprocessing.LabelEncoder()
    le.fit(orig_labels)
    orig_labels = le.transform(orig_labels)
    test_labels = le.transform(test_labels)
    
    orig_feats= orig_feats.astype(np.float64)
    small_scaler = preprocessing.StandardScaler()
    orig_feats = small_scaler.fit_transform(orig_feats)
    
    print 'FIRST ROUND: training with original features'
    allLearners_orig, used_labels = ac.train_adaboost(orig_feats,orig_labels,learning_rate, N_LAB, N_RUNS, N_ESTIM, Sample_N)
    #allLearners_orig, used_labels = ac.train_randomforest(orig_feats,orig_labels, N_LAB, N_RUNS, N_ESTIM, Sample_N)

    #tic= time.time()
    #confidence_orig= ac.compute_confidence(allLearners_orig, orig_feats, CLF)
    #print "time taken to produce SERIAL confidence:", round(time.time() - tic,2), "seconds"
    #import ipdb; ipdb.set_trace()
    #tic= time.time()
    confidence_orig= ac.compute_confidence_par(allLearners_orig, orig_feats, CLF)
    #print "time taken to produce PARALLEL confidence:", round(time.time() - tic,2), "seconds"
    #import ipdb; ipdb.set_trace()
    
    
    print 'Getting contextual features'
    orig_CF_35 = ac.get_contextual(confidence_orig, 35) #yeni = orig_CF_75[:,np.squeeze([np.sum(orig_CF_75,axis=0)!= 0])]
    orig_CF_75 = ac.get_contextual(confidence_orig, 75)
    orig_CF_110 = ac.get_contextual(confidence_orig, 110)
    CF_feats = np.concatenate([orig_CF_35,orig_CF_75, orig_CF_110], axis = 1)
    #CF_feats = orig_CF_75
    #import ipdb; ipdb.set_trace()
    big_scaler = preprocessing.StandardScaler()
    rich_feats = np.concatenate([orig_feats, CF_feats], axis=1)
    #import ipdb; ipdb.set_trace()
    rich_feats = big_scaler.fit_transform(rich_feats)
    print 'SECOND ROUND: training with original and contextual features'
    allLearners_rich, dumb = ac.train_adaboost(rich_feats, orig_labels, learning_rate, N_LAB, N_RUNS, N_ESTIM, Sample_N)
    #allLearners_rich, dumb = ac.train_randomforest(rich_feats, orig_labels, N_LAB, N_RUNS, N_ESTIM, Sample_N)

    print 'Computing confidence for the test features'
    test_feats= test_feats.astype(np.float64)
    test_feats  = small_scaler.transform(test_feats)
    confidence_test = ac.compute_confidence_par(allLearners_orig, test_feats, CLF)
    
    print 'Getting contextual features'
    test_CF_35 = ac.get_contextual(confidence_test, 35)
    test_CF_75 = ac.get_contextual(confidence_test, 75)
    test_CF_110 = ac.get_contextual(confidence_test, 110)
    test_CF_feats = np.concatenate([test_CF_35, test_CF_75, test_CF_110], axis = 1)
    #test_CF_feats = test_CF_75
    
    rich_test_feats = np.concatenate([test_feats, test_CF_feats], axis=1)

    print 'Computing confidence for the test and contextual features'
    rich_test_feats = big_scaler.transform(rich_test_feats)
    confidence_rich_test = ac.compute_confidence_par(allLearners_rich, rich_test_feats, CLF)
    pred = np.argmax(confidence_rich_test, axis=1)

#    pred_sur = mk.groupLabels(le.inverse_transform(pred))
#    test_labels_sur = mk.groupLabels(le.inverse_transform(test_labels))
    
    pred_sur = le.inverse_transform(pred)
    test_labels_sur = le.inverse_transform(test_labels)
    
    
    cm = confusion_matrix(test_labels_sur, pred_sur)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float16').T
    print 'the mean across the diagonal is ' + str(np.mean(norm_cm.diagonal()))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_cm, interpolation='nearest')
    fig.colorbar(cax)

    ACTIONS = np.unique(test_labels_sur)
    ax.set_xticks(range(-1,len(ACTIONS)))
    ax.set_yticks(range(-1,len(ACTIONS)))
    ax.set_xticklabels(['']+list(ACTIONS), rotation='vertical')
    ax.set_yticklabels(['']+list(ACTIONS))
    ax.axis('image')

    plt.show()
    import ipdb; ipdb.set_trace()
    
    confidence_rich_train = ac.compute_confidence_par(allLearners_rich, rich_feats, CLF)
    pred_train = np.argmax(confidence_rich_train, axis=1)
    
    pred_train_sur = mk.groupLabels(le.inverse_transform(pred_train))
    train_labels_sur = mk.groupLabels(le.inverse_transform(orig_labels))
    
    #pred_train_sur = le.inverse_transform(pred_train)
    #train_labels_sur = le.inverse_transform(orig_labels)
    
    cm = confusion_matrix(train_labels_sur, pred_train_sur)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float16').T
    print 'the mean across the diagonal FOR TRAINING is ' + str(np.mean(norm_cm.diagonal()))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_cm, interpolation='nearest')
    fig.colorbar(cax)
    
    ax.set_xticks(range(-1,len(ACTIONS)))
    ax.set_yticks(range(-1,len(ACTIONS)))
    ax.set_xticklabels(['']+list(ACTIONS), rotation='vertical')
    ax.set_yticklabels(['']+list(ACTIONS))
    ax.axis('image')
    
    plt.show()

    import ipdb;ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()
