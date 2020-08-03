# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:38:56 2017

@author: bbishop
"""

from __future__ import division
import wave, struct
import sys, os, cPickle
import random

def main():
    """
    Converts a directory of .wav files to a tuple of lists, which correspond
    to data intended to be run through a neural network.
    """
    test_data=None
    if len(sys.argv) < 3:
        print "Not enough inputs recognized. Usage is: \
        python wave_converter.py <directory> <desired file for data to be stored> <custom test data>\n \
        (custom test data is optional. Otherwise, the data will be split into 70% \
        training data, 15% validation data, and 15% test data.)"
        sys.exit(0)
    elif len(sys.argv) > 3:
        print "Too many inputs recognized. Usage is: \
        python wave_converter.py <directory> <desired file for data to be stored> <custom test data>\n \
        (custom test data is optional. Otherwise, the data will be split into 70% \
        training data, 15% validation data, and 15% test data.)"
        sys.exit(1)
        
    DIRECTORY = sys.argv[1]
    DESTINATION_FILE = sys.argv[2]

    EPOCH_LENGTH = 882  #corresponds to 50 nps. .wav files run at 44100 samples a second.
    NOTE_TABLE = {}    
    """
    NOTE_TABLE = {  # an 's' after a note corresponds to a sharp (#).
    "a4": 0, \
    "b4": 1, \
    "c4": 2, \
    "d3": 3, \
    "e3": 4, \
    "f3": 5, \
    "g3": 6, \
    "as4": 7, \
    "cs4": 8, \
    "ds3": 9, \
    "fs3": 10, \
    "gs3": 11, \
    "d4": 12}
    """
    PERCENT_TRAINING = 0.7  #percentage of data to be used as training data
    
    full_data = loadAllWaves(DIRECTORY, EPOCH_LENGTH, NOTE_TABLE)
    
    NUM_NOTES = len(NOTE_TABLE)
    param_info = (EPOCH_LENGTH, NUM_NOTES, NOTE_TABLE)
    split_data = separateData(full_data, len(full_data), PERCENT_TRAINING)    
    
    print "storing data in " + DESTINATION_FILE
    destfile = open(DESTINATION_FILE,'wb')
    cPickle.dump(split_data, destfile)      #dump the tuple (training_data, validation_data, test_data)
    #CHANGED
    cPickle.dump(param_info, destfile)      #dump the tuple (epoch_len, num_notes)
    destfile.close()
    print "storing complete. Network ready to test."

def loadAllWaves(directory, epoch_len, note_table):
    """
    Convert all .wav files in a given directory to python lists.
    Inputs:
        directory - the directory containing the .wav files
        epoch_len - the desired number of samples per data point
        note_table - a dict that enumerates all of the notes being read in
    Output:
        a list of (list, int) tuples, corresponding to a list of .wav
        data points and the note they represent
    """
    
    full_data = list()
    print "Loading all files from directory " + directory + ":"
    count = 0
    for filename in os.listdir(directory):
        #NOTE: files should all begin with the note the .wav file contains;
        # i.e. "c.wav" or "as_derp.wav"
        if filename.endswith(".wav"):
            print "Loading " + filename
            note = filename[0:3].replace("_","").lower()    #look for the desired note
            #CHANGED NEXT 2 LINES
            note_table[note] = count
            d = waveToList(directory+"/"+filename, epoch_len, count)
            full_data += d
            count += 1
    print str(count) + " wav files loaded successfully."
    print str(len(full_data)) + " data points found."
    return full_data

def waveToList(path, epoch_len, note):
    """
    Read in the binary strings from the two .wav channels and convert them
    to (list, int) tuples.
    Inputs:
        path - the filename being loaded
        epoch_len - the desired number of samples per data point
        note - the note being played in the .wav file
    Ouput:
        a list of (list, int) tuples, corresponding to a list of .wav
        data points and the note they represent
    """
    frames = loadWave(path)
    ch1, ch2 = zip(*frames)     #unpack the tuple into the separate channels
    #avg_channels = [sum(x)/2.0 for x in zip(ch1, ch2)]
    #currently, I only use the 1st channel. this could change in the future
    sep_ch, num_entries = spliceFrames(ch1, epoch_len)
    sep_ch = sep_ch[:-1]        #ignore the last entry, because it more than likely
    num_entries -= 1            # isn't the same length as the other entries
    
    if note != None:
        note_vector = [note] * num_entries
        data = zip(sep_ch, note_vector)
    else:
        data = sep_ch
    return data
    
def loadWave(f):
    """
    load the wave file f into a list by unpacking it into a struct,
    then putting that tuple into a list,
    Inputs:
        f - the .wav filename being unpacked
    Output:
        a list of tuples, which are samples of the .wav file that have been normalized
    """
    waveF = wave.open(f, 'r')
    frames = list()
    
    max_length = waveF.getnframes()      #total number of frames in wave file
    
    for i in range(0, max_length):
        waveData = waveF.readframes(1)
        data = struct.unpack("hh", waveData)    # "hh" means unpack the data into two 16-bit signed ints
        n_data = (data[0]/32768.0, data[1]/32768.0)     #division to normalize data
        frames.append(n_data)
        
    return frames
    
def spliceFrames(channel, epoch):
    """
    Separate a .wav file into epochs. We do this because training on a single
    sample would not be wise for training the network, since a single note can
    vary wildly in terms of what the .wav sample returns. Therefore, we use a
    certain period (the epoch) as a data point in the network.
    Inputs:
        channel - a single channel from the .wav file
        epoch - the length of the epoch (number of samples in a data point)
    Output:
        a list of lists, which is a list of the data points
    """
    total_frames = len(channel)
    x = [list(channel[i:i + epoch]) for i in range(0, total_frames, epoch)]
    return (x, len(x))

def separateData(data, num_entries, PERCENT_TRAINING):
    """
    separate data into training, test, and validation groups.
    """
    #randomize data first
    if (PERCENT_TRAINING > 0.9 or PERCENT_TRAINING < 0.1):
        print "Percentage of training data not in range. please choose a value between 0.1 and 0.9."
        sys.exit(1)
    
    print "Separating data into training, validation, and test sets..."

    PERCENT_VALIDATION = (1 - PERCENT_TRAINING)/2.0
    PERCENT_TEST = PERCENT_VALIDATION
    
    random.shuffle(data)
    break1 = int(PERCENT_TRAINING*num_entries)
    break2 = break1 + int(PERCENT_TEST*num_entries)
    training_data = data[:break1]
    validation_data = data[break1:break2]
    test_data = data[break2:]
    
    tr = convertToTuple(training_data)
    va = convertToTuple(validation_data)
    te = convertToTuple(test_data)
    return (tr, va, te)
    
def convertToTuple(data):
    """
    Convert the list of lists into a tuple of lists. This is necessary for
    unpacking the data after it is stored in a .pkl file.
    Inputs:
        data - a list of tuples, where tuple[0] is a list and tuple[1] is an int
    Output:
        a tuple containing a list of lists and a list of ints
    """
    a = list()
    b = list()
    for data_pt in data:
        a.append(data_pt[0])
        b.append(data_pt[1])
    return (a, b)
    
if __name__ == "__main__":
    main()