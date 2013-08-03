#!/usr/bin/env python

import tables   as ta
import scipy    as sp
import commands
import numpy as np
from scipy import random
from hmax.tools.utils import start_progressbar, end_progressbar, update_progressbar
import argparse
from scipy import io
import time
import h5py


#------------------------------------------------------------------------------#
def create_empty_table(table_fname):
    
    class images(ta.IsDescription):
        features     = ta.Float64Col(shape = (1441))
        label        = ta.StringCol(32)
    
    h5    = ta.openFile(table_fname, mode = 'w', title='list of images')
    group = h5.createGroup("/", 'input_output_data', 'images information')
    table = h5.createTable(group, 'readout', images, "readout example")
    #pp = table.row
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def read_mat_files(features_basename, labels_fname):
    """docstring for read_mat_file"""
    
    print "reading features"
    tic = time.time()
    f = h5py.File(features_basename, 'r')
    ff = f["myData"]
    features = sp.array(ff).T
    print "time taken :", time.time() - tic, 'seconds'

    print "reading labels"
    tic = time.time()
    labels = io.loadmat(labels_fname)['labels']
    
    labels = sp.array([str(labels[i][0][0]) for i in xrange(labels.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
    
    return features, labels
#------------------------------------------------------------------------------#
def populate_table(table_fname, features, labels):
    n_samples = labels.shape[0]
    pbar = start_progressbar(n_samples, '%i features to Pytable' % (n_samples))
    
    h5 = ta.openFile(table_fname, mode = 'a')
    table = h5.root.input_output_data.readout
    pp = table.row
    
    for i in xrange(n_samples):
        pp['features']    = features[i,:]
        pp['label']       = labels[i]
        pp.append()
        update_progressbar(pbar, i)
    
    end_progressbar(pbar)
    # save everything in the file and close it
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def main():
    """
        
        """
    parser = argparse.ArgumentParser(description="""This file does this and that \n
        usage: python ./file.py 11 --bla 10  blabla""")
    #parser.add_argument('--features_basename', type=str, help="""string""")
    parser.add_argument('--table_basename', type=str, help="""string""")
    parser.add_argument('--data_path', type=str, help="""this is the path for all the data files""")
    args = parser.parse_args()
    
    data_path = args.data_path
    table_basename =  data_path + args.table_basename
    
#    #this is test
#    #A
#    featuresA, labelsA = read_mat_files(data_path+'/a/myData.mat', data_path + '/a/labels.mat')
#    #B and the rest
#    featuresB, labelsB = read_mat_files(data_path+ '/b/myData.mat', data_path + '/b/labels.mat')
#    featuresC, labelsC = read_mat_files(data_path+ '/c/myData.mat', data_path + '/c/labels.mat')
#    featuresD, labelsD = read_mat_files(data_path+ '/d/myData.mat', data_path + '/d/labels.mat')
#    
#    import ipdb; ipdb.set_trace()
#    create_empty_table(table_basename+'_train.h5')
#    populate_table(table_basename+'_train.h5', np.concatenate((featuresA, featuresB, featuresC, featuresD )), np.concatenate((labelsA, labelsB, labelsC, labelsD)))
#
#    #EF
#    featuresE, labelsE = read_mat_files(data_path+ '/e/myData.mat', data_path+ '/e/labels.mat')
#    featuresF, labelsF = read_mat_files(data_path+ '/f/myData.mat', data_path+ '/f/labels.mat')
#    create_empty_table(table_basename+ '_test.h5')
#    
#    populate_table(table_basename+ '_test.h5', np.concatenate((featuresE,featuresF)) ,  np.concatenate((labelsE, labelsF)))
    
    
    
    #this is test
    #A
    featuresA, labelsA = read_mat_files(data_path+'/a/myData.mat', data_path + '/a/labels.mat')
    #B and the rest
    featuresB, labelsB = read_mat_files(data_path+ '/b/myData.mat', data_path + '/b/labels.mat')
    featuresE, labelsE = read_mat_files(data_path+ '/e/myData.mat', data_path + '/e/labels.mat')
    
    import ipdb; ipdb.set_trace()
    create_empty_table(table_basename+'_train.h5')
    populate_table(table_basename+'_train.h5', np.concatenate((featuresA, featuresB, featuresE)), np.concatenate((labelsA, labelsB, labelsE)))
    
    #EF
    featuresC, labelsC = read_mat_files(data_path+ '/c/myData.mat', data_path+ '/c/labels.mat')
    create_empty_table(table_basename+ '_test.h5')
    
    populate_table(table_basename+ '_test.h5', featuresC ,  labelsC)
    
    
    import ipdb; ipdb.set_trace()
#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

