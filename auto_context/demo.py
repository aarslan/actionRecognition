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
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl

#from sklearn import datasets
from pymatlab.matlab import MatlabSession


N_LIM = 20
N_ESTIM = 15 #ANALOGUOUS TO MAXITER IN MTALAB: boosting iterations
learning_rate = 1.
Sample_N = 1000 #Shengping's
N_RUNS = 10;
N_LAB = 13
N_JOBS = 8

#------------------------------------------------------------------------------#
def load_training(table_path, splitNo, trainOrTest):
    print 'loading ' + trainOrTest + ' features'
    h5 = ta.openFile(table_path + trainOrTest +'_' + str(splitNo) + '.h5', mode = 'r')
    table = h5.root.input_output_data.readout
    import ipdb;ipdb.set_trace()
    tic = time.time()
    features = sp.array(table.cols.features)#[:,:N_LIM]
    print "time taken:", time.time() - tic, "seconds"
    labels = sp.array(table.cols.label)
    print 'features loaded'
    return features, labels

#------------------------------------------------------------------------------#
def load_training_mats(mat_path, splitNo, trainOrTest):
    myData = [];
    labels = [];
    names = [];
    labFiles = glob.glob(mat_path + trainOrTest + '/*_labels_double.mat')
    
    features = np.array([])
    labels = np.array([])
    labFiles = [x[0:-18] for x in labFiles]
    print 'loading ' + trainOrTest + ' features'
    tic = time.time()
    
    for myFile in labFiles:
        dd = sp.io.loadmat(myFile+'_xavier_features.mat')['positon_features']
        ll = sp.io.loadmat(myFile+'_labels_double.mat')['labels_double']
        features = np.concatenate([x for x in [features, dd] if x.size > 0],axis=1)
        labels = np.concatenate([x for x in [labels, ll] if x.size > 0],axis=0)
    
            #import ipdb;ipdb.set_trace()

    features = np.array(features.T, dtype='float64')
    labels = labels[:,0]

    print "time taken:", round(time.time() - tic,2), "seconds"
    print str(features.shape[0]), ' features loaded'
    return features, labels

#------------------------------------------------------------------------------#
def train_adaboost(features, labels):
    uniqLabels = np.unique(labels)
    print 'TAKING ONLY ', str(N_LAB), ' LABELS FOR SPEED '
    uniqLabels = uniqLabels[:N_LAB]
    
    allLearners = []
    for targetLab in uniqLabels:
        print 'processing for label ', str(targetLab)
        runs=[]
        #import ipdb;ipdb.set_trace()
        for rrr in xrange(N_RUNS):
            #import ipdb;ipdb.set_trace()
            feats,labs = get_binary_sets(features, labels, targetLab)
            #print 'fitting stump'
            #import ipdb;ipdb.set_trace()
            baseClf = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
            baseClf.fit(feats, labs)
            ada_real = AdaBoostClassifier( base_estimator=baseClf, learning_rate=learning_rate,
                                      n_estimators=N_ESTIM,
                                      algorithm="SAMME.R")
            #import ipdb;ipdb.set_trace()
            runs.append(ada_real.fit(feats, labs))
        allLearners.append(runs)
    
    return allLearners
#------------------------------------------------------------------------------#
def get_binary_sets(features, labels, targetLab):
    #import ipdb;ipdb.set_trace()
    trainPos_idx = np.where(labels == targetLab)[0]
    trainNeg_idx = np.where(labels != targetLab)[0]
    #import ipdb;ipdb.set_trace()
    if len(trainPos_idx)<Sample_N/2:
        sample_id = np.array(trainPos_idx);
        sample_id = np.concatenate((sample_id, trainNeg_idx[random.sample(range(0,len(trainNeg_idx)), Sample_N-len(trainPos_idx))]));
    else:
        sample_id = np.array(trainPos_idx[random.sample(range(0,len(trainPos_idx)), Sample_N/2)])
        sample_id = np.concatenate((sample_id, trainNeg_idx[random.sample(range(0,len(trainNeg_idx)), Sample_N/2)]));
        
    feats = np.array(features[sample_id,:], dtype='float64');
    labs = np.array(labels[sample_id], dtype='int8');
    posInd = labs==targetLab
    negInd = labs!=targetLab
    labs[posInd] = 1;
    labs[negInd] = -1; #convert labels of positive samples to +1 and labels of negative samples to -1
    #import ipdb;ipdb.set_trace()
    return feats,labs

#------------------------------------------------------------------------------#
def compute_confidence(allLearners, dada):
    #import ipdb;ipdb.set_trace()
    lab_confidence = np.zeros([dada.shape[0], len(allLearners)])
    tic = time.time()
    #import ipdb;ipdb.set_trace()
    pbar = start_progressbar(len(allLearners), '%i producing weighted outputs' % len(allLearners))
    
    for ii,thisLab in enumerate(allLearners):
        res = np.zeros([dada.shape[0]])
        for jj, thisLearner in enumerate(thisLab):
            for hh, thisEstimator in enumerate(thisLearner):
                #multiply the predictions with the weight of the learner
                res = res+thisEstimator.predict(dada)*thisLearner.estimator_weights_[hh]
        lab_confidence[:,ii] = res
        update_progressbar(pbar, ii)
    end_progressbar(pbar)
    print "time taken to produce confidence:", round(time.time() - tic,2), "seconds"
    #import ipdb;ipdb.set_trace()
    return lab_confidence
#------------------------------------------------------------------------------#
def compute_confidence_par(allLearners, dada):

    lab_confidence = np.zeros([dada.shape[0], len(allLearners)])
    tic = time.time()
    #import ipdb;ipdb.set_trace()
    print 'producing weighted outputs IN PARALLEL'
    
    mem = Memory(cachedir='tmp')
    classif_RBF2 = mem.cache(confidence_par)
    
    c = l_c[0]
    r = Parallel(n_jobs=N_JOBS)(delayed(confidence_par)(allLearners,ii,dada) for ii in enumerate(allLearners))
    res, iis = zip(*r)
    
    for t,y in enumerate(iis):
        lab_confidence[:,y] = res[t]
    
    print "time taken to produce confidence:", round(time.time() - tic,2), "seconds"
    #import ipdb;ipdb.set_trace()
    return lab_confidence

#------------------------------------------------------------------------------#

def get_contextual(conf,Wsz):
    #% Compute temporal context features for multi-class action recognition
    #% from action classes confidence scores, result
    #% of prior learning iteration using a one vs all binary classifier
    #%  cf = ctxtFeat(conf,Wsz);
    #%
    #% INPUTS
    #%  conf   - [numFrames x numClasses] array containing confidence scores
    #%             result of learning
    #%  Wsz    - Window size (in frames) at which to compute features
    #% OUTPUTS
    #%  cf     - [numFrames x numCtxtFeatures] context features values
    #%             numCtxtFeatures = (16*(numClasses^2))
    #import ipdb;ipdb.set_trace()
    
    nEx,nBhv = conf.shape
    nCF = (5*(pow(nBhv,2)))+(pow(nBhv,2))
    cf = np.zeros((nEx,nCF),dtype='float32')
    
    cf[:,0:nBhv] = conf
    kk=nBhv
    for ii in range(0,nBhv):
        for jj in range(1,nBhv):
            cf[:,kk] = conf[:,ii]-conf[:,jj]
            kk=kk+1;
    orig=cf[:,0:kk]
    origKK=kk

    #import ipdb;ipdb.set_trace()
    import ipdb;ipdb.set_trace()
    #REST OF FEATURES
    for ii in xrange(nEx):
        cf1 = computeAll(orig,ii,Wsz)
        try:
            cf[ii,origKK:origKK+len(cf1)] = cf1
        except:
            import ipdb;ipdb.set_trace()
            'wot'
    import ipdb;ipdb.set_trace()
    print 'WARNING: DID YOU CHECK IF THERES EMPTY CONTEXT FEATURES?'
    cf = normalizeWF(cf)
    return cf

def computeAll(orig,ii,wsz):
    nEx,nBhv2 = orig.shape
    wsz = (wsz-1)/2
    import ipdb;ipdb.set_trace()
    window = orig[max(0,ii-wsz):min(nEx,ii+wsz),:]
    ctr = compute1(window, nBhv2)
    cf1 = ctr
    return cf1

def compute1(window,nBhv2):
    cf1 = np.append(np.mean(window,0),
                    [window[-1,:]-window[0,:],
                    np.max(window),
                    np.min(window),
                    np.var(window)])
    import ipdb;ipdb.set_trace()
    return cf1

def normalizeWF(wf):
    #    Normalize values per feature
    import ipdb;ipdb.set_trace()
    wf2 = np.array(wf)
    wf3 = np.array(wf)
    mn = np.min(wf,0)
    for f in xrange(len(wf[0])):
        wf[:,f] = wf[:,f]-mn[f]
        mx = np.max(wf[:,f],0)
        #import ipdb;ipdb.set_trace()
        if mx>0 :
            wf[:,f] = np.round((wf[:,f]/mx)*255)
    wf = np.array(wf, dtype='uint8');
    import ipdb;ipdb.set_trace()
    return wf
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#

def get_contextual_matlab(conf,Wsz):
    
    import platform
    if platform.node() != 'g6':
        dataPath = '/Users/aarslan/Brown/auto_context'
    else:
        dataPath = '/home/aarslan/prj/data/auto_context'

    print 'computing context features for windows size: ', str(Wsz)
    tic = time.time()
    session = MatlabSession()
    session.run('cd ', dataPath)
    session.putvalue('conf',conf)
    session.putvalue('Wsz',np.array([Wsz], dtype='float64'))
    session.run('B = ctxtFeat(conf, Wsz)')
    cf = session.getvalue('B')
    session.close()
    np.array(cf, dtype='uint8')
    print "Context feature for ", str(Wsz), " size windows took ", round(time.time() - tic,2), "seconds"
    return cf

def main():
    """
        This is where the magic happens
        """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
        usage: python ./classify_data.py --n_samples 10 --n_features 100 --features_fname ./bla.mat --labels_fname ./bla1.mat""")
    parser.add_argument('--table_path', type=str, help="""string""") ##THIS IS THE BASE NAME, PARTS WILL BE ADDED IN THE CODE
    parser.add_argument('--mat_path', type=str, default = '0', help="""string""")
    parser.add_argument('--split_no', type=int, default = 1, help="""string""")
    args = parser.parse_args()
    
    table_path = args.table_path
    mat_path = args.mat_path
    splitNo = args.split_no
    
    if mat_path == '0':
        orig_feats,orig_labels = load_training(table_path, splitNo, 'train')
    else:
        orig_feats,orig_labels = load_training_mats(mat_path, splitNo, 'train')

    orig_feats= orig_feats.astype(np.float64)
    allLearners_orig = train_adaboost(orig_feats,orig_labels)
    confidence_orig = compute_confidence(allLearners_orig, orig_feats)
    #confidence_orig = compute_confidence_par(allLearners_orig, orig_feats)

    orig_CF_75 = get_contextual_matlab(confidence_orig, 75)
    orig_CF_185 = get_contextual_matlab(confidence_orig, 185)
    orig_CF_615 = get_contextual_matlab(confidence_orig, 615)
    CF_feats = np.concatenate([orig_CF_75,orig_CF_185,orig_CF_615], axis = 1)

    rich_feats = np.concatenate([orig_feats,CF_feats], axis=1)
    allLearners_rich = train_adaboost(rich_feats,orig_labels)

    if mat_path == '0':
        test_feats,test_labels = load_training(table_path, splitNo, 'test')
    else:
        test_feats,test_labels = load_training_mats(mat_path, splitNo, 'test')

    test_feats= test_feats.astype(np.float64)
    confidence_test = compute_confidence(allLearners_orig, test_feats)
    #confidence_test_par = compute_confidence_par(allLearners_orig, test_feats)

    
    test_CF_75 = get_contextual_matlab(confidence_test, 75)
    test_CF_185 = get_contextual_matlab(confidence_test, 185)
    test_CF_615 = get_contextual_matlab(confidence_test, 615)
    test_CF_feats = np.concatenate([test_CF_75, test_CF_185, test_CF_615], axis = 1)
    rich_test_feats = np.concatenate([test_feats, test_CF_feats], axis=1)
    
    confidence_rich_test = compute_confidence(allLearners_rich, rich_test_feats)
    #confidence_rich_test = compute_confidence_par(allLearners_rich, rich_test_feats)
    pred = np.argmax(confidence_rich_test, axis=1)

    testUnique = np.unique(test_labels)[:N_LAB]

    import ipdb;ipdb.set_trace()

    used_labs = np.sum([test_labels == lab for lab in testUnique],0)
    truth = test_labels[used_labs.astype('bool')]
    pred2 = testUnique[pred][used_labs.astype('bool')]

    import ipdb;ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()
