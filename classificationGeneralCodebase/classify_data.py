#!/usr/bin/env python
"""string"""

import h5py
import scipy as sp

import hmax
from hmax.classification import kernel
#from shogun import Kernel, Classifier, Features
from hmax.tools.utils import start_progressbar, update_progressbar, end_progressbar
import scipy as sp
import numpy as np
from scipy import io
import tables as ta
from sklearn.svm import SVC

import time
import argparse

l_cats = sp.array(['SIL', 'add_saltnpepper', 'add_teabag', 'butter_pan', 'crack_egg',
'cut_bun', 'cut_fruit', 'cut_orange', 'fry_egg', 'fry_pancake',
'peel_fruit', 'pour_cereals', 'pour_coffee', 'pour_dough2pan',
'pour_egg2pan', 'pour_flour', 'pour_juice', 'pour_milk', 'pour_oil',
'pour_sugar', 'pour_water', 'put_bunTogether', 'put_egg2plate',
'put_fruit2bowl', 'put_pancake2plate', 'put_toppingOnTop',
'smear_butter', 'spoon_flour', 'spoon_powder', 'spoon_sugar',
'squeeze_orange', 'stir_cereals', 'stir_coffee', 'stir_dough',
'stir_egg', 'stir_fruit', 'stir_milk', 'stir_tea', 'take_bowl',
'take_butter', 'take_cup', 'take_eggs', 'take_glass', 'take_knife',
'take_plate', 'take_squeezer', 'take_topping', 'walk_in', 'walk_out'],
dtype='|S17')


REGULARIZATION_VALUE = 1E4
N_SAMPLES = 10000
N_FEATURES  = 500
l_c = [1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2]
#------------------------------------------------------------------------------#
def split_data_from_table(table_fname, n_samples = N_SAMPLES, n_features = N_FEATURES):

    h5 = ta.openFile(table_fname, mode = 'r')
    table = h5.root.input_output_data.readout

    l_features = table.cols.features
    l_index  = table.cols.frame_index
    l_labels = table.cols.label
    import ipdb; ipdb.set_trace()
    
    n_samples_total = len(l_labels)
    assert(2*n_samples < n_samples_total)

    import warnings ; warnings.warn("""have something that takes a split depending on classes to keep it balanced""")

    #TODO: have a balanced split on each class
    ind_total = sp.random.permutation(n_samples_total)
    ind_train = ind_total[:n_samples]
    ind_test = ind_total[n_samples:2*n_samples]
    sp.array([(ind_train == test).any() for test in ind_test]).any()
    print "checked that train and test do not overlap"

    """
    features_train = features.T[ind_train, :n_features]
    labels_train = labels[ind_train]
    features_test = features.T[ind_test, :n_features]
    labels_test = labels[ind_test]
    """

    features_train = sp.zeros((n_samples, n_features), dtype = 'uint8')
    features_test = sp.zeros((n_samples, n_features), dtype = 'uint8')
    labels_train = []
    labels_test = []

    pbar = start_progressbar(len(ind_train), '%i train features' % (len(ind_train)))
    for i, ind in enumerate(ind_train):
        features_train[i] = l_features[ind][:n_features]
        labels_train.append(l_labels[ind])
        update_progressbar(pbar, i)
    end_progressbar(pbar)

    pbar = start_progressbar(len(ind_test), '%i test features' % (len(ind_test)))
    for i, ind in enumerate(ind_test):
        features_test[i] = l_features[ind][:n_features]
        labels_test.append(l_labels[i])
        update_progressbar(pbar, i)
    end_progressbar(pbar)

    labels_train = sp.array(labels_train)
    labels_test = sp.array(labels_test)

    table.flush()
    h5.close()

    return features_train , labels_train, features_test, labels_test
#------------------------------------------------------------------------------#
def svm_cla_sklearn(features_train, features_test, labels_train, labels_test):
    """docstring for svm_sklearn"""

    features_train = sp.array(features_train, dtype = 'float32')
    features_test = sp.array(features_test, dtype = 'float32')

    print "zscore features and generating the normalized dot product kernel"
    tic = time.time()
    features_train_prep, mean_f, std_f = features_preprocessing(features_train)
    features_test_prep, mean_f, std_f  = features_preprocessing(features_test, mean_f, std_f)
    print "time taken to zscore data is:", time.time() - tic , "seconds"

    for c in l_c:
        tic = time.time()
        clf = SVC(gamma = 1, C=c)
        #clf = SVC(gamma=2, C=1),
        clf.fit(features_train_prep, labels_train)
        score = clf.score(features_test_prep, labels_test)
        print "score for C,",c, "is: ", score
        print "time taken:", time.time() - tic, "seconds"
    import ipdb; ipdb.set_trace()

#------------------------------------------------------------------------------#
def features_preprocessing(features, mean_f = None, std_f = None):

    features = sp.array(features, dtype = 'float32')

    if mean_f is None:
        mean_f = features.mean(0)
        std_f  = features.std(0)

    features -= mean_f
    # avoid zero division
    std_f[std_f == 0] = 1
    features /= std_f

    return features, mean_f, std_f

#------------------------------------------------------------------------------#
def euclidien(features1, features2 = None):
    """
    Builds a similarity matrix based ont eh euclidien distance
    """
    if features2 is None:
        features2 = features1

    nfeat1 = len(features1)
    nfeat2 = len(features2)

    # go
    kernelmatrix = sp.empty((nfeat1, nfeat2), dtype="float")

    if features1 is features2:

        # set up progress bar
        n_iter = 0
        niter = nfeat1 * (nfeat2+1) / 2
        pbar = start_progressbar(niter, "Kernel Train")

        for ifeat1, feat1 in enumerate(features1):

            a_2 = (feat1**2.).sum()

            for ifeat2, feat2 in enumerate(features2):

                if ifeat1 == ifeat2:
                    kernelmatrix[ifeat1, ifeat2] = 0

                # mattrix symmetric, do only top triangle
                elif ifeat1 > ifeat2:

                    a_b = sp.dot(feat1, feat2.T)

                    b_2 = (feat2**2.).sum()
                    dist = (a_2 - 2 *a_b + b_2)

                    # since kernel matrix is symmetric
                    kernelmatrix[ifeat1, ifeat2] = dist
                    kernelmatrix[ifeat2, ifeat1] = dist

                    update_progressbar(pbar, n_iter+1)
                    n_iter += 1
    else:

        # set up progress bar
        n_iter = 0
        niter = nfeat1 * nfeat2
        pbar = start_progressbar(niter, "Kernel Test")

        for ifeat1, feat1 in enumerate(features1):

            a_2 = (feat1**2.).sum()

            for ifeat2, feat2 in enumerate(features2):

                a_b = sp.dot(feat1, feat2.T)

                b_2 = (feat2**2.).sum()
                dist = (a_2 - 2 *a_b + b_2)

                kernelmatrix[ifeat1, ifeat2] = dist

                pbar = update_progressbar(pbar, n_iter)
                n_iter += 1

    end_progressbar(pbar)

    return kernelmatrix

# ------------------------------------------------------------------------------
def dot_product(features1,
                features2 = None):

    """
    generates kernel based on the dot product similarity measure

    input :
    features1 : vectors representing the rows in the matrix
    features2 : vectors representing the columns in the matrix

    output :
    out : the similarity matrix

    """

    if features2 is None:
        features2 = features1

    npoints1 = features1.shape[0]
    npoints2 = features2.shape[0]

    features1.shape = npoints1, -1
    features2.shape = npoints2, -1

    ndims = features1.shape[1]
    assert(features2.shape[1] == ndims)

    if ndims < DOT_MAX_NDIMS:
        out = sp.dot(features1, features2.T)
    else:
        out = sp.dot(features1[:, :DOT_MAX_NDIMS],
                     features2[:, :DOT_MAX_NDIMS].T)
        ndims_done = DOT_MAX_NDIMS
        while ndims_done < ndims:
            out += sp.dot(features1[:, ndims_done:ndims_done+DOT_MAX_NDIMS],
                          features2[:, ndims_done:ndims_done+DOT_MAX_NDIMS].T)
            ndims_done += DOT_MAX_NDIMS

    return out

# ------------------------------------------------------------------------------
def ndot_product(features1,
                 features2 = None):

    """
    generates kernel based on normalized dot product

    input :
    features1 : vectors representing the rows in the matrix
    features2 : vectors representing the columns in the matrix

    output :
    out : similarity matrix

    """

    features1.shape = features1.shape[0], -1
    features1 = features1/sp.sqrt((features1**2.).sum(1))[:, None]

    if features2 is None:
        features2 = features1
    else:
        features2.shape = features2.shape[0], -1
        features2 = features2/sp.sqrt((features2**2.).sum(1))[:, None]

    out = sp.dot(features1, features2.T)

    return out

#------------------------------------------------------------------------------#
def load_data(features_fname, labels_fname):

    tic = time.time()
    print "Loading features and labels"
    f = h5py.File(features_fname)
    import ipdb; ipdb.set_trace()
    ff = f["myData"]
    features = sp.array(ff).T
    import ipdb; ipdb.set_trace()
    labels = io.loadmat('labels_fname')['labels']
    labels = [str(labels[i][0][0]) for i in xrange(labels.shape[0])]
    print "making sure there is no NaN or Inf in the features"
    assert(not features.isnan().any())
    assert(not features.isinf().any())
    print "time taken to load data: ", time.time() - tic, 'seconds'
    print "Done"

    return features, labels
#------------------------------------------------------------------------------#
def split_data(features, labels, n_samples = N_SAMPLES, n_features = N_FEATURES):
    """docstring for split_data"""

    n_samples_total = labels.shape[0]
    assert(2*n_samples < n_samples_total)

    import warnings ; warnings.warn("""have something that takes a split depending on classes to keep it balanced""")

    #TODO: have a balanced split on each class
    ind_total = sp.random.permutation(n_samples_total)
    ind_train = ind_total[:n_samples]
    ind_test = ind_total[n_samples:2*n_samples]
    sp.array([(ind_train == test).any() for test in ind_test]).any()
    print "checked that train and test do not overlap"

    features_train = features.T[ind_train, :n_features]
    labels_train = labels[ind_train]
    features_test = features.T[ind_test, :n_features]
    labels_test = labels[ind_test]

    return features_train, labels_train, features_test, labels_test

#------------------------------------------------------------------------------#
def svm_cla(features_train, features_test, labels_train, labels_test):

    print "zscore features and generating the normalized dot product kernel"
    tic = time.time()
    features_train_prep, mean_f, std_f = features_preprocessing(features_train)
    features_test_prep, mean_f, std_f  = features_preprocessing(features_test, mean_f, std_f)
    print "time taken to zscore data is:", time.time() - tic , "seconds"
    tic = time.time()
    k2_train = ndot_product(features_train_prep)
    k2_test  = ndot_product(features_train_prep, features_test_prep)
    print "time taken to generate the kernels is:", time.time() - tic , "seconds"
    print 'Done\n'

    # SVM CLASSIFICATION
    print 'Classification C2 in process...'
    results = []
    for c in l_c:
        print "For C= ", c
        tic = time.time()
        classifier = build_classifier(k2_train, labels_train, c)
        results   += [classify_data(k2_test, classifier, labels_test)]
        print "time taken: ", time.time() - tic, 'seconds'
        print "classification results: " , results
    print 'Done'

    return results
#------------------------------------------------------------------------------#
def build_classifier(kernel_train, train_labels, regularization = REGULARIZATION_VALUE):
    """
    docstring
    """
    assert((kernel_train != np.NaN).all())
    assert((kernel_train != np.Inf).all())

    alphas = {}
    support_vectors = {}
    biases = {}
    customkernel = Kernel.CustomKernel()

    kernel_train = sp.array(kernel_train, dtype = 'float64')

    customkernel.set_full_kernel_matrix_from_full(kernel_train)

    cat_index = {}

    # -- train
    categories = sp.unique(train_labels)
    if categories.size == 2:
        categories = [categories[0]]
    for icat, cat in enumerate(categories):
        ltrain = sp.zeros((train_labels.size))
        ltrain[train_labels != cat] = -1
        ltrain[train_labels == cat] = +1
        ltrain = sp.array(ltrain, dtype = 'float64')
        current_labels = Features.Labels(ltrain)
        svm = Classifier.LibSVM(regularization,
                                customkernel,
                                current_labels)
        assert(svm.train())
        alphas[cat] = svm.get_alphas()
        svs = svm.get_support_vectors()
        support_vectors[cat] = svs
        biases[cat] = svm.get_bias()
        cat_index[cat] = icat

    classifier = {}
    classifier['alphas']          = alphas
    classifier['support_vectors'] = support_vectors
    classifier['biases']          = biases
    classifier['cat_index']       = cat_index
    classifier['categories']       = categories

    return classifier

#------------------------------------------------------------------------------#
def classify_data(kernel_test, classifier, test_labels):
    """
    docstring
    """

    alphas          = classifier['alphas']
    support_vectors = classifier['support_vectors']
    biases          = classifier['biases']
    cat_index       = classifier['cat_index']
    categories      = classifier['categories']
    n_test = len(test_labels)

    pred = sp.zeros((n_test))
    distances = sp.zeros((n_test, len(categories)))
    for icat, cat in enumerate(categories):
        for point in xrange(n_test):
            index_sv = support_vectors[cat]
            resp     = sp.dot(alphas[cat], kernel_test[index_sv, point]) + biases[cat]
            distances[point, icat] = resp

    balanced_accuracy = 0.0
    import warnings ; warnings.warn("MAKE SURE THE BALANCED ACCURACY IS GOOD")

    pred = distances.argmax(1)
    gtt = sp.array([cat_index[e] for e in test_labels]).astype("int")
    balanced_accuracy = (pred == gtt).sum() / float(len(pred))
    accuracy = 100.*balanced_accuracy / float(len(categories))

    print distances.shape
    print "Classification accuracy on test data (%):", accuracy

    return accuracy
#------------------------------------------------------------------------------#
def main():

    parser = argparse.ArgumentParser(description="""This file does this and that \n
            usage: python ./classify_data.py --n_samples 10 --n_features 100 --features_fname ./bla.mat --labels_fname ./bla1.mat""")
    parser.add_argument('--n_features', type=int, default = N_FEATURES, help="""string""")
    parser.add_argument('--n_samples', type=int, default = N_SAMPLES, help="""string""")
    parser.add_argument('--table_fname', type=str, help="""string""")
    args = parser.parse_args()

    table_fname = args.table_fname
    n_features = args.n_features
    n_samples = args.n_samples

    #features, labels = load_data(features_fname, labels_fname)
    #features = sp.array(255*sp.random.rand(N_FEATURES, 3*N_SAMPLES), dtype = 'int')
    #labels = sp.random.randint(low = 0, high = 49, size = 3*N_SAMPLES)

    features_train , labels_train, features_test, labels_test = split_data_from_table(table_fname, n_samples, n_features)
    #features_train , labels_train, features_test, labels_test = split_data(features, labels, n_samples, n_features)

    #svm_cla(features_train, features_test, labels_train, labels_test)
    svm_cla_sklearn(features_train, features_test, labels_train, labels_test)

#------------------------------------------------------------------------------#
if __name__=="__main__":
    main()

