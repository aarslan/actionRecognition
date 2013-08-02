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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from  matplotlib import pyplot as plt
from joblib import Parallel, Memory, delayed
import time
import argparse
import pylab as pl
from multiprocessing import Process

#from sklearn import datasets
#from pymatlab.matlab import MatlabSession

ACTIONS = ['approach','walk_away','circle','chase','attack','copulation','drink','eat','clean','human','sniff','up','other']


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
def train_adaboost(features, labels, learning_rate, n_lab, n_runs, n_estim, n_samples):
    uniqLabels = np.unique(labels)
    print 'Taking ', str(n_lab), ' labels'
    uniqLabels = uniqLabels[:n_lab]
    used_labels = uniqLabels
    pbar = start_progressbar(len(uniqLabels), 'training adaboost for %i labels' %len(uniqLabels))
    allLearners = []
    for yy ,targetLab in enumerate(uniqLabels):
        runs=[]
        for rrr in xrange(n_runs):
            #import ipdb;ipdb.set_trace()
            feats,labs = get_binary_sets(features, labels, targetLab, n_samples)
            #print 'fitting stump'
            #import ipdb;ipdb.set_trace()
            baseClf = DecisionTreeClassifier(max_depth=1, min_samples_leaf=4, min_samples_split=4)
            baseClf.fit(feats, labs)
            ada_real = AdaBoostClassifier( base_estimator=baseClf, learning_rate=learning_rate,
                                      n_estimators=n_estim,
                                      algorithm="SAMME.R")
            #import ipdb;ipdb.set_trace()
            runs.append(ada_real.fit(feats, labs))
        allLearners.append(runs)
        update_progressbar(pbar, yy)
    end_progressbar(pbar)
    
    return allLearners, used_labels

#------------------------------------------------------------------------------#
def train_randomforest(features, labels, n_lab, n_runs, n_estim, n_samples):
    
    uniqLabels = np.unique(labels)
    print 'TAKING ONLY ', str(n_lab), ' LABELS FOR SPEED '
    print "using random forests"
    uniqLabels = uniqLabels[:n_lab]
    used_labels = uniqLabels
    
    allLearners = []
    #import ipdb;ipdb.set_trace()
    for rrr in xrange(n_runs):
        #import ipdb;ipdb.set_trace()
        feats,labs = get_multi_sets(features, labels, used_labels, n_samples)
        #import ipdb;ipdb.set_trace()
        rfclf = RandomForestClassifier(n_estimators=n_estim, max_depth=1, min_samples_split=1, random_state=0)
        #import ipdb;ipdb.set_trace()
        allLearners.append(rfclf.fit(feats, labs))
    
    return allLearners, used_labels

#------------------------------------------------------------------------------#
def get_binary_sets(features, labels, targetLab, Sample_N):
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
def get_multi_sets(features, labels, used_labels, Sample_N):
    from sklearn.cross_validation import train_test_split
    #size = (Sample_N*used_labels.shape[0])/float(labels.shape[0])
    #feats, null1, labs, null2 = train_test_split(features, labels, train_size=size)
    
    labels_train = []
    
    #nn, bins, patches = pl.hist(labels, len(used_labels))
    #Sample_N = min(nn)
    cnt = 0
    #import ipdb;ipdb.set_trace()
    features_train = np.zeros(((Sample_N*used_labels.shape[0]), features.shape[1]), dtype = 'float')
    for myLab in used_labels:
        allExemplars = np.where(labels == myLab)[0]
        try:
            selInd = np.random.choice(allExemplars, Sample_N, replace=False)
        except ValueError:
            #import ipdb;ipdb.set_trace()
            selInd = np.random.choice(allExemplars, Sample_N, replace=True)
        uzun = len(selInd)
        labels_train = labels_train + list(labels[selInd])
        #import ipdb;ipdb.set_trace()
        features_train[cnt:cnt+uzun,:] = features[selInd,:]
        cnt = cnt+uzun

    feats = features_train[features_train[:,1]!=0, :]
    labs = np.array(labels_train)
    if labs.shape[0] != feats.shape[0]:
        raise ValueError('the label and feat dimensions in get_multi_set dont match')

    return feats,labs

#------------------------------------------------------------------------------#
def compute_confidence(allLearners, dada, classifierType):
    #import ipdb;ipdb.set_trace()
    
    tic = time.time()
    #import ipdb;ipdb.set_trace()
    
    if classifierType == 'adaboost':
        lab_confidence = np.zeros([dada.shape[0], len(allLearners)], dtype='float64')
        pbar = start_progressbar(len(allLearners), '%i producing weighted outputs' % len(allLearners))
        for ii,thisLab in enumerate(allLearners):
            res = np.zeros([dada.shape[0]], dtype='float64')
            for jj, thisLearner in enumerate(thisLab):
                my_weights = thisLearner.estimator_weights_
                #tic = time.time()
                for hh, thisEstimator in enumerate(thisLearner):
                    res = res+thisEstimator.predict(dada)*my_weights[hh]
                    #import ipdb;ipdb.set_trace()
            lab_confidence[:,ii] = np.float64(res)
            update_progressbar(pbar, ii)
        end_progressbar(pbar)
    
    if classifierType == 'randomforest':
        #import ipdb;ipdb.set_trace()
        lab_confidence = np.zeros((dada.shape[0],len(allLearners[0].classes_)), dtype='float64')
        pbar = start_progressbar(len(allLearners), '%i producing weighted outputs' % len(allLearners[0].classes_))
        for ii, thisRun in enumerate(allLearners):
            lab_confidence +=  thisRun.predict_proba(dada)
            update_progressbar(pbar, ii)
        end_progressbar(pbar)

    return lab_confidence

#------------------------------------------------------------------------------#
def compute_confidence_par2(allLearners, dada):
    from multiprocessing import Process
    nthreads = len(allLearners)
    def worker(allLearners,dada, outdict):
        for n, thisLab in enumerate(allLearners):
            outdict[n] = confidante(thisLab, dada)
    
    threads = []
    outs = [{} for i in range(nthreads)]
    
    for i in range(nthreads):
        # Create each thread, passing it its chunk of numbers to factor
        # and output dict.
        t = Process(target=worker,
                             args=(allLearners,dada,outs[i]))
        threads.append(t)
        t.start()
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    # Merge all partial output dicts into a single dict and return it
    return {k: v for out_d in outs for k, v in out_d.iteritems()}

#------------------------------------------------------------------------------#

def get_contextual(conf,Wsz):
    tic = time.time()

    nEx,nBhv = conf.shape
    nCF = (5*(pow(nBhv,2)))+(pow(nBhv,2))
    #cf = np.zeros((nEx,nCF),dtype='float32')
    cf = conf
    kk=nBhv
    for ii in range(0,nBhv):
        for jj in range(1,nBhv):
            cf = np.concatenate((cf, np.array([conf[:,ii]-conf[:,jj]]).T),axis=1)
    cf = cf[:,np.sum(cf,axis=0)!=0]
    orig=cf

    #REST OF FEATURES
    winFeats =[]
    for ii in xrange(nEx):
        #import ipdb;ipdb.set_trace()
        winFeats.append(compute_windowed(orig,ii,Wsz))
    cf = np.concatenate((cf, np.array(winFeats)),axis=1)
    #cf = normalize(cf)
    return cf
#------------------------------------------------------------------------------#

def compute_windowed(orig,ii,wsz):
    nEx,nBhv2 = orig.shape
    wsz = (wsz-1)/2
    window = orig[max(0,ii-wsz):min(nEx,ii+wsz),:]
    
    cf1 = np.append(np.mean(window,0),
                    [window[-1,:]-window[0,:],
                     np.max(window, axis=0),
                     np.min(window, axis=0),
                     np.var(window, axis=0)])
    if any(np.isinf(cf1)):
        import ipdb;ipdb.set_trace()
    return cf1
#------------------------------------------------------------------------------#
def normalize(raw):
    high = 255.0
    low = 0.0
    mins = np.min(raw, axis=0)
    maxs = np.max(raw, axis=0)
    rng = maxs - mins
    scaled_points = high - (((high - low) * (maxs - raw)) / rng)
    return scaled_points
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
    session.run('cd '+ dataPath)
    session.putvalue('conf',np.squeeze(np.array([conf], dtype='float64')))
    session.putvalue('Wsz',np.array([Wsz], dtype='float64'))
    session.run('B = ctxtFeat(conf, Wsz)')
    cf = session.getvalue('B')
    session.close()
    np.array(cf, dtype='uint8')
    print "Context feature for ", str(Wsz), " size windows took ", round(time.time() - tic,2), "seconds"
    return cf
#------------------------------------------------------------------------------#
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
    allLearners_orig, used_labels = train_adaboost(orig_feats,orig_labels)
#    tic = time.time()
#    confidence_orig = compute_confidence_new(allLearners_orig, orig_feats)
#    print "time taken new way:", round(time.time() - tic,2), "seconds"
    tic = time.time()
    confidence_orig= compute_confidence(allLearners_orig, orig_feats)
    print "time taken old way:", round(time.time() - tic,2), "seconds"

    #confidence_orig = compute_confidence_par(allLearners_orig, orig_feats)
            
    #import ipdb;ipdb.set_trace()

    orig_CF_75 = get_contextual(confidence_orig, 75)

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


    cm = confusion_matrix(truth, pred2)
    norm_cm = np.divide(cm.T,sum(cm.T), dtype='float64').T
    print 'the mean across the diagonal is ' + str(np.mean(norm_cm.diagonal()))
    #    pl.matshow(norm_cm)
    #    pl.colorbar()
    #    pl.show()

    #alpha = ['ABC', 'DEF', 'GHI', 'JKL']

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
