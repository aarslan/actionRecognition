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
import pickle

N_ESTIM = 40
learning_rate = 1.
N_SAMPLES = 5000
N_RUNS = 5
N_LAB = 55
CLF = 'adaboost'#'randomforest' #
N_FEATURES = 700

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
def get_bfast_splits(table_fname, settings, n_samples = N_SAMPLES, n_features = N_FEATURES, n_lab = N_LAB, contig_labels = True):
    
    h5_all = ta.openFile(table_fname, mode = 'r')
    table_all = h5_all.root.input_output_data.readout

    train_p = settings['train_p']
    test_p = settings['test_p']
    cams = settings['cameras']

    #import ipdb; ipdb.set_trace()
    #row_train = [table_all.where("(partiNames == '%s') & (camNames == '%s')" % (ll, nn) ) for ll in train_p for nn in cams ]
    labels_train = []
    features_train = []
    labels_test = []
    features_test = []
    
    pbar = start_progressbar(len(train_p), str(len(train_p))+ ' training participants' )
    for jj,pat in enumerate(train_p):
        for cam in cams:
            labels_train = labels_train + [row['label'] for row in
                                           table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
            features_train = features_train+ [row['features'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
        update_progressbar(pbar, jj)
    end_progressbar(pbar)

    pbar = start_progressbar(len(train_p), str(len(train_p))+ ' testing participants' )
    for jj,pat in enumerate(test_p):
        for cam in cams:
            labels_test = labels_test + [row['label'] for row in
                                           table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
            features_test = features_test+ [row['features'] for row in table_all.readWhere("(partiNames == '%s') & (camNames == '%s')" % (pat, cam)) ]
        update_progressbar(pbar, jj)
    end_progressbar(pbar)

    tic = time.time()
    #uniqLabels = np.intersect1d(np.unique(labels_train), np.unique(labels_test))
    uniqLabels = np.unique(labels_train)
    #KILL UNUSED
    uniqLabels=uniqLabels[uniqLabels!='SIL']
    uniqLabels = uniqLabels[:n_lab]
    print 'using ',str(len(uniqLabels)),' labels in total'

    labels_train = np.array(labels_train)
    selector = np.zeros_like(labels_train, dtype= 'bool')
    for uL in uniqLabels:
        selector = np.squeeze(selector|[labels_train == uL])
    labels_train = labels_train[selector]
    features_train = np.array(features_train)[selector,:n_features]

    labels_test = np.array(labels_test)
    selector = np.zeros_like(labels_test, dtype= 'bool')
    for uL in uniqLabels:
        selector = np.squeeze(selector|[labels_test == uL])
    labels_test = labels_test[selector]
    features_test = np.array(features_test)[selector,:n_features]
    print "Loaded features converted in ", round(time.time() - tic) , "seconds"
    
    table_all.flush()
    h5_all.close()
    return features_train , labels_train, features_test, labels_test

#------------------------------------------------------------------------------#

def main():
    parser = argparse.ArgumentParser(description="""This file does this and that """)
    parser.add_argument('--table_path', type=str, help="""string""")
    args = parser.parse_args()
    
    table_path = args.table_path
    
    settings = {
        'train_p' :  [ 'P03', 'P04', 'P05','P06', 'P07', 'P08', 'P09', 'P10',
                      'P11', 'P12', 'P13', 'P14', 'P15','P16', 'P17', 'P18', 'P19', 'P20',
                      'P21', 'P22', 'P23', 'P24', 'P25','P26', 'P27', 'P28', 'P29', 'P30',
                      'P31','P32','P33','P34','P35','P36','P37','P38','P39','P40',
                      'P41','P42','P43','P44','P45', 'P46', 'P47', 'P48', 'P49',
                      'P50', 'P51', 'P52', 'P53', 'P54'],
        'test_p': [ 'P16'], #
                'cameras' : ['webcam01', 'webcam02', 'cam01', 'cam02','stereo01', 'stereo02', 'webcam01mirr', 'webcam02mirr', 'cam01mirr', 'cam02mirr','stereo01mirr', 'stereo02mirr']
                }
    
    orig_feats , orig_labels, test_feats, test_labels = get_bfast_splits(table_path, settings, 3000,
                                                                                 N_FEATURES,
                                                                                 contig_labels = True, n_lab = N_LAB)
    

    selector_path = '/home/aarslan/oldumulan'
    if not os.path.exists(selector_path):
        feats,labs = ac.get_multi_sets(orig_feats, orig_labels, np.unique(orig_labels), 3000)
        tic = time.time()
        selector = LinearSVC(C=0.000006, penalty="l1", dual=False).fit(feats, labs)
        print "time taken to score data is:", round(time.time() - tic) , "seconds"
        container = {}
        container['selector'] = selector
        import ipdb; ipdb.set_trace()
        pickle.dump(container, open(selector_path, 'wb'))
    else:
        import ipdb; ipdb.set_trace()
        container = pickle.load(open(selector_path, 'rb'))
        container['selector'].transform(orig_feats)

    import ipdb; ipdb.set_trace()
    le = preprocessing.LabelEncoder()
    le.fit(orig_labels)
    orig_labels = le.transform(orig_labels)
    test_labels = le.transform(test_labels)
    
    #orig_feats= orig_feats.astype(np.float64)
    small_scaler = preprocessing.StandardScaler()
    orig_feats = small_scaler.fit_transform(orig_feats)
    
    print 'FIRST ROUND: training with original features'
    allLearners_orig, used_labels = ac.train_adaboost(orig_feats,orig_labels,learning_rate, N_LAB, N_RUNS, N_ESTIM, N_SAMPLES)
    #allLearners_orig, used_labels = ac.train_randomforest(orig_feats,orig_labels, N_LAB, N_RUNS, N_ESTIM, Sample_N)

    confidence_orig= ac.compute_confidence_par(allLearners_orig, orig_feats, CLF)

    
    print 'Getting contextual features'
    #orig_CF_35 = ac.get_contextual(confidence_orig, 35) #yeni = orig_CF_75[:,np.squeeze([np.sum(orig_CF_75,axis=0)!= 0])]
    orig_CF_75 = ac.get_contextual(confidence_orig, 75)
    #orig_CF_110 = ac.get_contextual(confidence_orig, 110)
    #CF_feats = np.concatenate([orig_CF_75, orig_CF_110], axis = 1)
    CF_feats = orig_CF_75
    #import ipdb; ipdb.set_trace()
    big_scaler = preprocessing.StandardScaler()
    rich_feats = np.concatenate([orig_feats, CF_feats], axis=1)
    #import ipdb; ipdb.set_trace()
    rich_feats = big_scaler.fit_transform(rich_feats)
    print 'SECOND ROUND: training with original and contextual features'
    allLearners_rich, dumb = ac.train_adaboost(rich_feats, orig_labels, learning_rate, N_LAB, N_RUNS, N_ESTIM, N_SAMPLES)

    print 'Computing confidence for the test features'
    test_feats= test_feats.astype(np.float64)
    test_feats  = small_scaler.transform(test_feats)
    confidence_test = ac.compute_confidence(allLearners_orig, test_feats, CLF)
    
    print 'Getting contextual features'
    #test_CF_35 = ac.get_contextual(confidence_test, 35)
    test_CF_75 = ac.get_contextual(confidence_test, 75)
    #test_CF_110 = ac.get_contextual(confidence_test, 110)
    #test_CF_feats = np.concatenate([test_CF_75, test_CF_110], axis = 1)
    test_CF_feats = test_CF_75
    
    rich_test_feats = np.concatenate([test_feats, test_CF_feats], axis=1)

    print 'Computing confidence for the test and contextual features'
    rich_test_feats = big_scaler.transform(rich_test_feats)
    confidence_rich_test = ac.compute_confidence(allLearners_rich, rich_test_feats, CLF)
    pred = np.argmax(confidence_rich_test, axis=1)

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
    
    confidence_rich_train = ac.compute_confidence(allLearners_rich, rich_feats, CLF)
    pred_train = np.argmax(confidence_rich_train, axis=1)
    
    pred_train_sur = mk.groupLabels(le.inverse_transform(pred_train))
    train_labels_sur = mk.groupLabels(le.inverse_transform(orig_labels))
    
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
