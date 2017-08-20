# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:34:36 2017

@author: bbishop
"""

import cPickle
import numpy as np

def loadWaveData(path):
    f = open(path)
    #maybe unecessary, could just load the tuple and return that.
    #leaving it for purpose of seeing what data is loaded
    training_data, validation_data, test_data = cPickle.load(f)
    data = (training_data, validation_data, test_data)
    EPOCH_LENGTH, NUM_NOTES = cPickle.load(f)
    params = (EPOCH_LENGTH, NUM_NOTES)
    f.close()
    
    print len(validation_data[0])
    print len(test_data[0])
    return (data, params)
    
def loadWrapper(filename):
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``loadWaveData``, but the format is more
    convenient for use in our implementation of neural networks.
    
    We first call loadWaveData, then convert the inputs to numpy arrays
    and the results to vectors, which are the size of the number of notes
    in the data. They are vectors with a 0 entry everywhere except for the index
    corresponding to the note of the result, which has a 1 entry. This is only
    done for training data, because validation and test data just needs the index,
    not a vector.

    """
    loaded_data = loadWaveData(filename)
    tr_d, va_d, te_d = loaded_data[0]
    EPOCH_LENGTH, NUM_NOTES = loaded_data[1]
    
    training_inputs = [np.reshape(x, (EPOCH_LENGTH, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y, NUM_NOTES) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    validation_inputs = [np.reshape(x, (EPOCH_LENGTH, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    
    if len(te_d) == 2:
        test_inputs = [np.reshape(x, (EPOCH_LENGTH, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])
    else:
        test_data = [np.reshape(x, (EPOCH_LENGTH, 1)) for x in te_d]
    
    data = (training_data, validation_data, test_data)
    params = (EPOCH_LENGTH, NUM_NOTES)
    return (data, params)
    
def vectorized_result(j, NUM_NOTES):
    """
    Return a NUM_NOTES-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a note
    into a corresponding desired output from the neural
    network.
    """
    e = np.zeros((NUM_NOTES, 1))
    e[j] = 1.0
    return e