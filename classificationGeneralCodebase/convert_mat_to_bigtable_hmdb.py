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

N_PARTS = 10
N_FEATURES_TOTAL = 1000
N_SAMPLES = 571741 #10000

#------------------------------------------------------------------------------#
def create_empty_table(table_fname):

    class images(ta.IsDescription):
        frame_index  = ta.Int32Col(shape = (1))
        features     = ta.UInt8Col(shape = (N_FEATURES_TOTAL*N_PARTS))
        label        = ta.StringCol(128)
        aviNames     = ta.StringCol(256)
        camname      = ta.StringCol(128)

    h5    = ta.openFile(table_fname, mode = 'w', title='list of images')
    group = h5.createGroup("/", 'input_output_data', 'images information')
    table = h5.createTable(group, 'readout', images, "readout example")
    #pp = table.row
    table.flush()
    h5.close()
#------------------------------------------------------------------------------#
def read_mat_files(features_basename, labels_fname, names_fname, camname_fname, actname_fname, partiname_fname):
    """docstring for read_mat_file"""

    print "reading features"
    tic = time.time()
    
    f = h5py.File(features_basename + '_part1.mat', 'r')
    ff = f["myData"]
    features = sp.array(ff).T
    for nn in range(2,N_PARTS+1):
        f = h5py.File(features_basename + '_part' + str(nn)+ '.mat', 'r')
        ff = f["myData"]
        temp = sp.array(ff).T
        features = sp.append(features, temp,1)
        print nn

    print "time taken :", time.time() - tic, 'seconds'
    

    print "reading labels"
    tic = time.time()
    labels = io.loadmat(labels_fname)['labels']
    labels = sp.array([str(labels[i][0][0]) for i in xrange(labels.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
    
    print "reading file names"
    tic = time.time()
    aviNames = io.loadmat(names_fname)['myNames']
    aviNames = sp.array([str(aviNames[i][0][0]) for i in xrange(aviNames.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'

    # few sanity checks
    #assert(not features.isnan().any())
    #assert(not features.isinf().any())

    return features, labels, aviNames
#------------------------------------------------------------------------------#
def populate_table(table_fname, features, labels, aviNames):

    n_samples = labels.shape[0]
    pbar = start_progressbar(n_samples, '%i features to Pytable' % (n_samples))

    h5 = ta.openFile(table_fname, mode = 'a')
    table = h5.root.input_output_data.readout
    pp = table.row

    for i in xrange(n_samples):
        pp['frame_index'] = i
        pp['features']    = features[i, :]
        pp['label']       = labels[i]
        pp['aviNames']    = aviNames[i][0:-4]
        pp.append()
        update_progressbar(pbar, i)

    end_progressbar(pbar)
    # save everything in the file and close it
    table.cols.aviNames.createIndex()
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def main():
    """

    """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
            usage: python ./file.py 11 --bla 10  blabla""")
    parser.add_argument('--features_basename', type=str, help="""string""")
    parser.add_argument('--labels_fname', type=str, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""")
    parser.add_argument('--aviname_fname', type=str, help="""string""")
    args = parser.parse_args()

    features_basename = args.features_basename
    labels_fname = args.labels_fname
    table_fname = args.table_fname
    names_fname = args.aviname_fname
    

    features, labels, aviNames = read_mat_files(features_basename, labels_fname, names_fname, camname_fname, actname_fname, partiname_fname)
    #features = sp.array(255*sp.random.rand(N_FEATURES_TOTAL, 3*N_SAMPLES), dtype = 'int').T
    #labels = sp.random.randint(low = 0, high = 49, size = 3*N_SAMPLES)

    create_empty_table(table_fname)
    populate_table(table_fname, features, labels, aviNames)

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

