#!/usr/bin/env python

import tables as ta
import scipy as sp
import numpy as np
import commands
import scipy
from scipy import random
from hmax.tools.utils import start_progressbar, end_progressbar, update_progressbar
from sklearn.svm import SVC, LinearSVC
import argparse
from scipy import io
import time
import h5py
import auto_context_demo as ac
import pickle
import os.path
#h5py._errors.unsilence_errors()

N_PARTS = 20    #HMDB 10
N_FEATURES_TOTAL = 500 #HMDB 1000
N_SAMPLES = 1000 #5453533 #HMDB 571741 #10000

#------------------------------------------------------------------------------#
def create_empty_table(table_fname, feature_len):
    class images(ta.IsDescription):
        frame_index  = ta.Int32Col(shape = (1))
        features     = ta.UInt8Col(shape = (feature_len))
        label        = ta.StringCol(128)
        camNames     = ta.StringCol(32)
        actNames     = ta.StringCol(64)
        partiNames   = ta.StringCol(4)
    
    
    h5    = ta.openFile(table_fname, mode = 'w', title='list of images')
    group = h5.createGroup("/", 'input_output_data', 'images information')
    table = h5.createTable(group, 'readout', images, "readout example")
    #pp = table.row
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def read_data_files(features_name):
    """docstring for read_mat_file"""
    
    print "reading features"
    tic = time.time()
    f = h5py.File(features_name, 'r')
    ff = f["myData"]
    features = np.array(ff, dtype='uint8').T
    print "time taken :", time.time() - tic, 'seconds'
    
    return features
#------------------------------------------------------------------------------#
def read_meta_files(labels_fname, camname_fname, actname_fname, partiname_fname):
    
    print "reading participant names"
    tic = time.time()
    partiNames = np.squeeze(io.loadmat(partiname_fname)['myPartis'])
    partiNames_items = np.squeeze(io.loadmat(partiname_fname+'_items')['myPartis_items'])
    print "time taken :", time.time() - tic, 'seconds'
    
    print "reading labels"
    tic = time.time()
    labels = np.squeeze(io.loadmat(labels_fname)['myLabels'])
    labels_items = np.squeeze(io.loadmat(labels_fname+'_items')['myLabels_items'])
    print "time taken :", time.time() - tic, 'seconds'
    
    print "reading camera names"
    tic = time.time()
    camNames = np.squeeze(io.loadmat(camname_fname)['myCams'])
    camNames_items = np.squeeze(io.loadmat(camname_fname+'_items')['myCams_items'])
    print "time taken :", time.time() - tic, 'seconds'

    print "reading action names"
    tic = time.time()
    actNames = np.squeeze(io.loadmat(actname_fname)['myActs'])
    actNames_items = np.squeeze(io.loadmat(actname_fname+'_items')['myActs_items'])
    print "time taken :", time.time() - tic, 'seconds'

    return labels, camNames, actNames, partiNames
#------------------------------------------------------------------------------#
def read_meta_files_items(labels_fname, camname_fname, actname_fname, partiname_fname):
    
    partiNames_items = io.loadmat(partiname_fname+'_items', squeeze_me=True)['myPartis_items']
    labels_items = io.loadmat(labels_fname+'_items', squeeze_me=True)['myLabels_items']
    camNames_items = io.loadmat(camname_fname+'_items', squeeze_me=True)['myCams_items']
    actNames_items = io.loadmat(actname_fname+'_items', squeeze_me=True)['myActs_items']
    
    partiNames_items= np.array([str(x) for x in partiNames_items])
    labels_items=np.array([str(x) for x in labels_items])
    camNames_items=np.array([str(x) for x in camNames_items])
    actNames_items=np.array([str(x) for x in actNames_items])
    
    return labels_items, camNames_items, actNames_items, partiNames_items
#------------------------------------------------------------------------------#
def feature_selector(features, labels):
    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, RFECV
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import zero_one_loss
    tic = time.time()
    selector = LinearSVC(C=0.000008, penalty="l1", dual=False).fit(features, labels)
    print "time taken to score data is:", round(time.time() - tic) , "seconds"
    return selector

#------------------------------------------------------------------------------#
def populate_table(table_fname, features, labels, camNames, actNames, partiNames):
    
    n_samples = labels.shape[0]
    pbar = start_progressbar(n_samples, '%i features to Pytable' % (n_samples))
    
    h5 = ta.openFile(table_fname, mode = 'a')
    table = h5.root.input_output_data.readout
    pp = table.row
    
    for i in xrange(n_samples):
        pp['frame_index'] = i
        pp['features']    = features[i, :]
        pp['label']       = labels[i]
        #pp['aviNames']    = aviNames[i][0:-4]
        pp['camNames']   = camNames[i]
        pp['actNames']   = actNames[i]
        pp['partiNames'] = partiNames[i]
        pp.append()
        update_progressbar(pbar, i)
    
    end_progressbar(pbar)
    # save everything in the file and close it
    table.cols.camNames.createIndex()
    table.cols.actNames.createIndex()
    table.cols.partiNames.createIndex()
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def main():
    """
        
        """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
        usage: python ./file.py 11 --bla 10  blabla""")
    parser.add_argument('--data_path', type=str, help="""this is the path for all the data files""", default = '/Users/aarslan/Desktop/bfast_ClassifData/')
    parser.add_argument('--features_basename', type=str, help="""string""", default = 'myData_v2_slide_len1_part')
    parser.add_argument('--labels_fname', type=str, help="""string""", default = 'myLabels')
    parser.add_argument('--table_fname', type=str, help="""string""", default = 'selected.h5')
    parser.add_argument('--camname_fname', type=str, help="""string""", default = 'myCams')
    parser.add_argument('--actname_fname', type=str, help="""string""", default= 'myActs')
    parser.add_argument('--partiname_fname', type=str, help="""string""", default= 'myPartis')
    args = parser.parse_args()
    
    data_path = args.data_path
    features_basename = data_path + args.features_basename
    labels_fname =  data_path + args.labels_fname
    table_fname = data_path + args.table_fname
    camname_fname = data_path + args.camname_fname
    actname_fname = data_path + args.actname_fname
    partiname_fname = data_path + args.partiname_fname
    
    labels, camNames, actNames, partiNames = read_meta_files(labels_fname, camname_fname, actname_fname, partiname_fname)
    labels_items, camNames_items, actNames_items, partiNames_items = read_meta_files_items(labels_fname, camname_fname, actname_fname, partiname_fname)
    
    selector_path = data_path+'featureSelectors'
    if not os.path.exists(selector_path):
        selectors = {}
        for pp in range(1,20):
            feature_name = features_basename + str(pp)+ '.mat'
            features = read_data_files(feature_name)
            features_small, labels_small = ac.get_multi_sets(features, labels, np.unique(labels), N_SAMPLES)
            selector = feature_selector(features_small, labels_small)
            print 'selected features: ',str(sum(sum(selector.coef_) != 0))
            selectors[feature_name] = selector
        pickle.dump(selectors, open(selector_path, 'wb'))
    else: #if not os.path.exists(table_fname):
        print 'already found a selector'
        stuff = np.empty((labels.shape[0], 1000), dtype = 'uint8')
        selectors = pickle.load(open(selector_path, 'rb'))
        feature_names = selectors.keys()
        cur_ind = 0
        total_len = 0
        for fn in feature_names:
            print 'loading',fn
            features = read_data_files(fn)
            features_trans = selectors[fn].transform(features)
            features = []
            len = features_trans.shape[1]
            stuff[:,cur_ind:cur_ind+len] = features_trans
            cur_ind=cur_ind+len
            total_len += len
            print 'appended',str(len), 'features'
        #import ipdb; ipdb.set_trace()
        stuff = stuff[:,:total_len]
#    else:
#        h5 = ta.openFile(table_fname, mode = 'a')
#        table = h5.root.input_output_data.readout
#        l_labels = table.cols.label
#        l_features = table.cols.features
#        labels_train = []
#        stuff = np.empty(l_features.shape)
#
##        pbar = start_progressbar(l_features.shape[0], '%i test features' % (l_features.shape[0]))
##        for gg, bb in range(0,l_features.shape[0]):
##            stuff[gg,:]=l_features[gg]
##            update_progressbar(pbar, gg)
##        end_progressbar(pbar)
##
##        import ipdb; ipdb.set_trace()
##        armut = [aa for aa in l_labels]
##        
##        import ipdb; ipdb.set_trace()
#
#        features_small, labels_small = ac.get_multi_sets(l_features, l_labels, np.unique(labels), N_SAMPLES)
#        import ipdb; ipdb.set_trace()

    labels = labels_items[np.int64(labels)-1]
    partis = partiNames_items[np.int64(partiNames)-1]
    cams   = camNames_items[np.int64(camNames)-1]
    acts   = actNames_items[np.int64(actNames)-1]
    import ipdb; ipdb.set_trace()

    create_empty_table(table_fname, total_len)
    populate_table(table_fname, stuff, labels, cams, acts, partis)
    

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

