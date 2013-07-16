#!/usr/bin/env python

import tables   as ta
import scipy    as sp
import commands
import scipy
from scipy import random
from hmax.tools.utils import start_progressbar, end_progressbar, update_progressbar
import argparse
from scipy import io
import time
import h5py

h5py._errors.unsilence_errors()

N_PARTS = 15    #HMDB 10
N_FEATURES_TOTAL = 500 #HMDB 1000
N_SAMPLES = 5453533 #HMDB 571741 #10000

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
def read_mat_files(features_basename, labels_fname, camname_fname, actname_fname, partiname_fname):
    """docstring for read_mat_file"""
    
    print "reading features"
    tic = time.time()
    
    f = h5py.File(features_basename + '_part1.mat', 'r')
    ff = f["myData"]
    #import ipdb; ipdb.set_trace()
    features = sp.array(ff).T
    for nn in range(2,N_PARTS+1):
        f = h5py.File(features_basename + '_part' + str(nn)+ '.mat', 'r')
        ff = f["myData"]
        temp = sp.array(ff).T
        features = sp.append(features, temp,1)
        print nn
    
    print "time taken :", time.time() - tic, 'seconds'

    print "reading participant names"
    tic = time.time()
    partiNames = io.loadmat(partiname_fname)['myPartis']
    partiNames = sp.array([str(partiNames[i][0][0]) for i in xrange(partiNames.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'

    print "reading labels"
    tic = time.time()
    labels = io.loadmat(labels_fname)['labels']
    labels = sp.array([str(labels[i][0][0]) for i in xrange(labels.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
    
    print "reading camera names"
    tic = time.time()
    camNames = io.loadmat(camname_fname)['myCams']
    camNames = sp.array([str(camNames[i][0][0]) for i in xrange(camNames.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'

    
    print "reading action names"
    tic = time.time()
    actNames = io.loadmat(actname_fname)['myActs']
    actNames = sp.array([str(actNames[i][0][0]) for i in xrange(actNames.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
            


    # few sanity checks
    #assert(not features.isnan().any())
    #assert(not features.isinf().any())
    
    return features, labels, camNames, actNames, partiNames
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
    parser.add_argument('--data_path', type=str, help="""this is the path for all the data files""")
    parser.add_argument('--features_basename', type=str, help="""string""")
    parser.add_argument('--labels_fname', type=str, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""")
    parser.add_argument('--camname_fname', type=str, help="""string""", default = '')
    parser.add_argument('--actname_fname', type=str, help="""string""", default= '')
    parser.add_argument('--partiname_fname', type=str, help="""string""", default= '')
    args = parser.parse_args()
    
    data_path = args.data_path
    features_basename = data_path + args.features_basename
    labels_fname =  data_path + args.labels_fname
    table_fname = data_path + args.table_fname
    camname_fname = data_path + args.camname_fname
    actname_fname = data_path + args.actname_fname
    partiname_fname = data_path + args.partiname_fname
    

    features, labels, camNames, actNames, partiNames = read_mat_files(features_basename, labels_fname, camname_fname, actname_fname, partiname_fname)
    create_empty_table(table_fname)
    populate_table(table_fname, features, labels, camNames, actNames, partiNames)

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

