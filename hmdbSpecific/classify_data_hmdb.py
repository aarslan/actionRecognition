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


l_cats = sp.array(['brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs','dive', 'draw_sword','dribble','drink','eat','fall_floor','fencing','flic_flac','golf','handstand','hit','hug', 'jump','kick','kick_ball','kiss','laugh','pick','pour','pullup','punch','push','pushup','ride_bike', 'ride_horse','run','shake_hands','shoot_ball','shoot_bow','shoot_gun','sit','situp','smile','smoke', 'somersault','stand','swing_baseball','sword','sword_exercise','talk','throw','turn','walk','wave'], dtype='|S17')


REGULARIZATION_VALUE = 1E4
N_SAMPLES = 15# 571741    %GUZEL SONUC 7 sample, 100 feat gamma=0.000001
N_FEATURES  = 10000 #1000
l_c = [1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2]
l_g = pow(2,np.linspace(-15, -5,16))
#------------------------------------------------------------------------------#
def get_HMDB_splits(table_fname, vidName, vidMode, n_samples = N_SAMPLES, n_features = N_FEATURES):

    h5 = ta.openFile(table_fname, mode = 'r')
    table = h5.root.input_output_data.readout
    
    l_features = table.cols.features
    l_index  = table.cols.frame_index
    l_labels = table.cols.label
    l_aviNames = table.cols.aviNames
    #    assert(2*n_samples < n_samples_total)

    trVidName = [];
    teVidName = [];
    noVidName = [];
    for j in range(0, len(vidMode)-1):
        innerVidMode = vidMode[j]
        #import ipdb; ipdb.set_trace()
        trVidName = trVidName + [vidName[j][i] for i, x in enumerate(innerVidMode) if x == '1']
        teVidName = teVidName + [vidName[j][i] for i, x in enumerate(innerVidMode) if x == '2']
        noVidName = noVidName + [vidName[j][i] for i, x in enumerate(innerVidMode) if x == '0']
    
    features_train =[]
    labels_train = []
    
    features_test = []
    labels_test = []

#import ipdb; ipdb.set_trace()
    
    exctCnt = 0
    pbar = start_progressbar(len(trVidName), '%i train features' % (len(trVidName)))
    for i, vid in enumerate(trVidName):
        tempLabels = [row['label'] for row in table.where("aviNames == vid")]
        try:
            selInd = random.sample(range(0,len(tempLabels)), n_samples)
        except ValueError:
            selInd = range(0,len(tempLabels))
            exctCnt = exctCnt+1
        labels_train = labels_train + [tempLabels[gg] for gg in selInd]
        tempFeatures = [row['features'][:][:n_features] for row in table.where("aviNames == vid")]
        features_train = features_train + [tempFeatures[gg] for gg in selInd]
        update_progressbar(pbar, i)
        
    end_progressbar(pbar)
    
    print'finished with %i exceptions' % (exctCnt)
    
    pbar = start_progressbar(len(teVidName), '%i test features' % (len(teVidName)))
    for i, vid in enumerate(teVidName):
        labels_test = labels_test + [row['label'] for row in table.where("aviNames == vid")]
        features_test = features_test + [row['features'][:][:n_features] for row in table.where("aviNames == vid")]
        update_progressbar(pbar, i)
    end_progressbar(pbar)

    print 'Converting arrays to sp'
    features_train = sp.array(features_train, dtype = 'uint8')
    features_test = sp.array(features_test, dtype = 'uint8')
    labels_train = sp.array(labels_train)
    labels_test = sp.array(labels_test)
    print 'Converted'
    
    table.flush()
    h5.close()
    print "feature loading completed"
    return features_train , labels_train, features_test, labels_test

#------------------------------------------------------------------------------#
def svm_cla_sklearn(features_train, features_test, labels_train, labels_test):
    """docstring for svm_sklearn"""

    features_train = sp.array(features_train, dtype = 'uint8')
    features_test = sp.array(features_test, dtype = 'uint8')

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
    
    features_train = sp.array(features_train, dtype = 'uint8')
    features_test = sp.array(features_test, dtype = 'uint8')
    
    print "zscore features"
    tic = time.time()
    features_train, mean_f, std_f = features_preprocessing(features_train)
    features_test, mean_f, std_f  = features_preprocessing(features_test, mean_f, std_f)
    print "time taken to zscore data is:", round(time.time() - tic) , "seconds"
    
    featSize = np.shape(features_train)
    selector = LinearSVC(C=0.0007, penalty="l1", dual=False).fit(features_train, labels_train)

    print 'Starting with %d samp, %d feats, keeping %d' % (featSize[0], featSize[1], (np.shape(selector.transform(features_train)))[1])
    print 'classifying'
    
    features_train = selector.transform(features_train)
    features_test = selector.transform(features_test)
    #import ipdb; ipdb.set_trace()
    mem = Memory(cachedir='tmp')
    classif_RBF2 = mem.cache(classif_RBF)

    c = l_c[0]
    Parallel(n_jobs=8)(delayed(classif_RBF2)(features_train, features_test, labels_train, labels_test, g, c) for g in l_g)
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

def parse_HMDB_splits(splitPath, splitNo):
    import glob
    files = glob.glob( splitPath + '*'+ str(splitNo) +'.txt')
    vidMode = []
    vidName = []
    for ff in range(0,len(files)):
        #inexing'i falan yap burda
        fName = files[ff]
        lines = [line.strip() for line in open(fName)]
        things= [sp.partition(' ') for sp in lines]
        vidMode.append([])
        vidName.append([])
        for ll in things:
            vidName[ff].append(ll[0][0:-4])
            vidMode[ff].append(ll[2])
    
    print '%d videos were selected' % (len(vidMode))
    return (vidName, vidMode)


#------------------------------------------------------------------------------#
def main():
    
    parser = argparse.ArgumentParser(description="""This file does this and that \n
            usage: python ./classify_data.py --n_samples 10 --n_features 100 --features_fname ./bla.mat --labels_fname ./bla1.mat""")
    parser.add_argument('--n_features', type=int, default = N_FEATURES, help="""string""")
    parser.add_argument('--n_samples', type=int, default = N_SAMPLES, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""") ##THIS IS THE BASE NAME, PARTS WILL BE ADDED IN THE CODE
    args = parser.parse_args()

    table_fname = args.table_fname
    n_features = args.n_features
    n_samples = args.n_samples
    
    import platform
    if platform.node() != 'g6':
        splitPath = '/Users/aarslan/Desktop/hmdb_ClassifData/testTrainMulti_7030_splits/'
    else:
        splitPath = '/home/aarslan/prj/data/hmdb_ClassifData/testTrainMulti_7030_splits/'
    vidName, vidMode = parse_HMDB_splits(splitPath, 1)
    
    features_train , labels_train, features_test, labels_test = get_HMDB_splits(table_fname, vidName, vidMode, n_samples, n_features)
    #svm_cla(features_train, features_test, labels_train, labels_test)
    svm_cla_sklearn_feat_sel(features_train, features_test, labels_train, labels_test)
    #svm_cla_sklearn_feat_sel(features_train, features_test, labels_train, labels_test)

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

