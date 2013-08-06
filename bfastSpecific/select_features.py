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
#h5py._errors.unsilence_errors()

N_PARTS = 20    #HMDB 10
N_FEATURES_TOTAL = 500 #HMDB 1000
N_SAMPLES = 100000 #5453533 #HMDB 571741 #10000

#------------------------------------------------------------------------------#
def create_empty_table(table_fname):
    
    class images(ta.IsDescription):
        frame_index  = ta.Int32Col(shape = (1))
        features     = ta.UInt8Col(shape = (N_FEATURES_TOTAL*N_PARTS))
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
def read_data_files(features_basename, part):
    """docstring for read_mat_file"""
    
    print "reading features"
    tic = time.time()
    f = h5py.File(features_basename + str(part)+ '.mat', 'r')
    ff = f["myData"]
    features = np.array(ff, dtype='uint8').T
    print "time taken :", time.time() - tic, 'seconds'
    
    return features
#------------------------------------------------------------------------------#
def read_meta_files(labels_fname, camname_fname, actname_fname, partiname_fname):
    
    print "reading participant names"
    tic = time.time()
    partiNames = np.squeeze(io.loadmat(partiname_fname)['myPartis'])
    print "time taken :", time.time() - tic, 'seconds'
    
    print "reading labels"
    tic = time.time()
    labels = np.squeeze(io.loadmat(labels_fname)['myLabels'])
    print "time taken :", time.time() - tic, 'seconds'
    
    print "reading camera names"
    tic = time.time()
    camNames = np.squeeze(io.loadmat(camname_fname)['myCams'])
    print "time taken :", time.time() - tic, 'seconds'

    print "reading action names"
    tic = time.time()
    actNames = np.squeeze(io.loadmat(actname_fname)['myActs'])
    print "time taken :", time.time() - tic, 'seconds'

    return labels, camNames, actNames, partiNames
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
def feature_selector(features, labels):
    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, RFECV
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import zero_one_loss
    tic = time.time()
    selector = LinearSVC(C=0.00001, penalty="l1", dual=False).fit(features[:N_SAMPLES,:], labels[:N_SAMPLES])
    print "time taken to zscore data is:", round(time.time() - tic) , "seconds"
    return selector

#------------------------------------------------------------------------------#
def main():
    """
        
        """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
        usage: python ./file.py 11 --bla 10  blabla""")
    parser.add_argument('--data_path', type=str, help="""this is the path for all the data files""", default = '/Users/aarslan/Desktop/bfast_ClassifData/')
    parser.add_argument('--features_basename', type=str, help="""string""", default = 'myData_v2_slide_len1_part1')
    parser.add_argument('--labels_fname', type=str, help="""string""", default = 'myLabels.mat')
    parser.add_argument('--table_fname', type=str, help="""string""", default = 'selected.h5')
    parser.add_argument('--camname_fname', type=str, help="""string""", default = 'myCams.mat')
    parser.add_argument('--actname_fname', type=str, help="""string""", default= 'myActs.mat')
    parser.add_argument('--partiname_fname', type=str, help="""string""", default= 'myPartis.mat')
    args = parser.parse_args()
    
    data_path = args.data_path
    features_basename = data_path + args.features_basename
    labels_fname =  data_path + args.labels_fname
    table_fname = data_path + args.table_fname
    camname_fname = data_path + args.camname_fname
    actname_fname = data_path + args.actname_fname
    partiname_fname = data_path + args.partiname_fname
    
    labels, camNames, actNames, partiNames = read_meta_files(labels_fname, camname_fname, actname_fname, partiname_fname)
    selectors = {}
    for pp in range(1,20):
        features = read_data_files(features_basename, pp)
        selector = feature_selector(features, labels)
        print 'selected features: ',str(sum(sum(selector.coef_) != 0))
        selectors[features_basename] = selector
        import ipdb; ipdb.set_trace()
    import ipdb; ipdb.set_trace()
    np.io.savemat(data_path+'featureSelectors.mat',selectors)
    create_empty_table(table_fname)
    populate_table(table_fname, features, labels, camNames, actNames, partiNames)

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

