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


#------------------------------------------------------------------------------#
def create_empty_table(table_fname):
    
    class images(ta.IsDescription):
        features     = ta.Float64Col(shape = (20))
        label        = ta.Int32Col(shape=(1))
        names        = ta.StringCol(128)
    
    h5    = ta.openFile(table_fname, mode = 'w', title='list of images')
    group = h5.createGroup("/", 'input_output_data', 'images information')
    table = h5.createTable(group, 'readout', images, "readout example")
    #pp = table.row
    table.flush()
    h5.close()

#------------------------------------------------------------------------------#
def read_mat_files(features_basename, labels_fname, names_fname):
    """docstring for read_mat_file"""
    
    print "reading features"
    tic = time.time()
    f = h5py.File(features_basename, 'r')
    ff = f["myData"]
    features = sp.array(ff).T
    print "time taken :", time.time() - tic, 'seconds'

#    print "reading labels"
#    tic = time.time()
#    labels = io.loadmat(labels_fname)['labels']
#    labels = sp.array([str(labels[i][0][0]) for i in xrange(labels.shape[0])])
#    print "time taken :", time.time() - tic, 'seconds'
#    import ipdb; ipdb.set_trace()
    
    
    print "reading labels"
    tic = time.time()
    labels = io.loadmat(labels_fname)['labels']
    labels = sp.array([labels[i] for i in xrange(labels.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
    import ipdb; ipdb.set_trace()
    
    print "reading names"
    tic = time.time()
    names = io.loadmat(names_fname)['names']
    names = sp.array([str(names[i][0][0]) for i in xrange(names.shape[0])])
    print "time taken :", time.time() - tic, 'seconds'
    import ipdb; ipdb.set_trace()
    
    return features, labels, names
#------------------------------------------------------------------------------#
def populate_table(table_fname, features, labels, names):
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
    parser.add_argument('--features_basename', type=str, help="""string""")
    parser.add_argument('--labels_fname', type=str, help="""string""")
    parser.add_argument('--names_fname', type=str, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""")
    parser.add_argument('--data_path', type=str, help="""this is the path for all the data files""")
    args = parser.parse_args()
    
    data_path = args.data_path
    features_basename = data_path + args.features_basename
    labels_fname = data_path + args.labels_fname
    names_fname = data_path + args.names_fname
    table_fname =  data_path + args.table_fname
    
    features, labels, names = read_mat_files(features_basename, labels_fname, names_fname)
    create_empty_table(table_fname)
    populate_table(table_fname, features, labels, names)


#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

