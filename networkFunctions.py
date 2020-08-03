# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:02:29 2017

@author: bbishop

This file 
"""

from __future__ import division
import sys
import network2
import wave_loader
import operator
import pyaudio, wave
import numpy as np

def main():
    #createNetwork(sys.argv[1], sys.argv[2])
    testNetwork(sys.argv[1], sys.argv[2], sys.argv[3])

def createNetwork(filename, dest_filename):
    all_data = wave_loader.loadWrapper(filename)
    
    net = optimize_network(all_data)
    net.save(dest_filename)
    
def optimize_network(data):
    training_data, validation_data, test_data = data[0]
    EPOCH_LENGTH, NUM_NOTES, NOTE_TABLE = data[1]
    
    max_neurons = NUM_NOTES*5
    num_steps = 10
    neuron_step = int((max_neurons / num_steps) + 0.5)
    #neuron_range = range(max_neurons)
    #neuron_range = neuron_range[0:max_neurons:neuron_step]
    neuron_range = [200, 225]
    best_accuracy = -1
    best_num_neurons = -1
    best_net = network2.Network([EPOCH_LENGTH, NUM_NOTES], cost=network2.CrossEntropyCost)
    for cnt in neuron_range:
        print "testing network with {} neurons.".format(cnt)
        net = network2.Network([EPOCH_LENGTH, cnt, NUM_NOTES], cost=network2.CrossEntropyCost)
        e_c, e_a, t_c, t_a = net.SGD(training_data, 60, 10, 1.5,
                lmbda = 0.5,
                evaluation_data=validation_data,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                monitor_training_accuracy=False,
                monitor_training_cost=False)
        if (max(e_a) > best_accuracy):
            best_accuracy = max(e_a)
            best_net = net
            best_num_neurons = cnt
    print "Best network contained {} neurons.".format(best_num_neurons)
    print "Best accuracy was {}".format(best_accuracy)
    return best_net
    
def testNetwork(filename, testData, otherData):
    net = network2.loadNetwork(filename)
    data = wave_loader.loadTest(testData)
    
    test_results = net.testUnkownData(data)
    
    #all_data = wave_loader.loadWrapper(otherData)
    #EPOCH_LENGTH, NUM_NOTES, NOTE_TABLE = all_data[1]
    
    NOTE_TABLE = ["a3", \
    "a4", \
    "a5", \
    "a6", \
    "as3", \
    "as4", \
    "as5", \
    "as6", \
    "b3", \
    "b4", \
    "b5", \
    "b6", \
    "c3", \
    "c4", \
    "c5", \
    "c6", \
    "cs3", \
    "cs4", \
    "cs5", \
    "cs6", \
    "d2", \
    "d3", \
    "d4", \
    "d5", \
    "d6", \
    "ds2", \
    "ds3", \
    "ds4", \
    "ds5", \
    "e2", \
    "e3", \
    "e4", \
    "e5", \
    "f2", \
    "f3", \
    "f4", \
    "f5", \
    "fs2", \
    "fs3", \
    "fs4", \
    "fs5", \
    "g2", \
    "g3", \
    "g4", \
    "g5", \
    "gs2", \
    "gs3", \
    "gs4", \
    "gs5", \
    "noise"]
    #NOTE_TABLE = ["a", "b", "c", "d8", "e", "f", "g", "as", "cs", "ds", "fs", "gs" , "d", "STRUM", "CHANGE"]
    #NOTE_TABLE = ["c", "e", "g", "Cmaj"]
    #recognized_notes = {}
    #for r in test_results:
    """
        cur_note = NOTE_TABLE[r]
        if cur_note in recognized_notes:
            recognized_notes[cur_note] += 1
        else: 
            recognized_notes[cur_note] = 1
    sorted_notes = sorted(recognized_notes.items(), key=operator.itemgetter(1), reverse=True)
    print "recognized notes, sorted by frequency: "
    for sn in sorted_notes:
        print str(sn[0]) + ": " + str(sn[1])
    """
        #print NOTE_TABLE[r]
    meh, mehi = partitionNotes(test_results)
    print "starting with {}".format(NOTE_TABLE[mehi[-1]])
    
    CHUNK = 882
    wf = wave.open(testData, 'rb')
    
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    data = wf.readframes(CHUNK)
    index = 0
    #last_seen = meh[0][0]
    while (data != ''):
        stream.write(data)
        data = wf.readframes(CHUNK)
        
        try:
            if index in mehi:
                print "Changed to {}".format(NOTE_TABLE[mehi[index]])
            #print "{0}, {1}".format(NOTE_TABLE[meh[index][0]], NOTE_TABLE[test_results[index]])
        except IndexError:
            print "Index out of range"
            
        #last_seen = meh[index]
        index += 1
    
    stream.stop_stream()
    stream.close()
    
    p.terminate()
    
def partitionNotes(note_vector):
    """
    Partition a set of notes into groupings, trying to identify outliers
    that could mean the strum of a note occuring.
    """
    output = list()
    change_indices = {}
    BUFFER_SIZE = 7
    STRUM = 13
    CHANGE = 14
    #add the first notes in the buffer range with no check
    for i in range(BUFFER_SIZE):
        output.append((note_vector[i], False))
    
    #establish the first note as the most occurring note in the initial Buffer
    cur_note = max(set(note_vector[0:BUFFER_SIZE]), key=note_vector[0:BUFFER_SIZE].count)
    change_indices[-1] = cur_note#uhh why did I use the last index here? Nvm, it's a dictionary so I can look up the start note
    for index, note in enumerate(note_vector[BUFFER_SIZE:], BUFFER_SIZE):
        if (note == cur_note):
            output.append((note, False))
        else:
            #detected a change
            #FIXME will have end of array index errors
            if (index + BUFFER_SIZE < len(note_vector)):
                #make sure we have a range to check for
                buffer_check = note_vector[index:index+BUFFER_SIZE]
                next_common = max(set(buffer_check), key=buffer_check.count)
                if (next_common == cur_note):
                    #looks like we found either a strum or a slight error
                    #output[index] = STRUM
                    output.append((note, False))
                else:
                    #we found a change of notes (or we're playing really fast)
                    #output[index] = CHANGE
                    output.append((note, True))
                    change_indices[index] = next_common
                    cur_note = next_common
            else:
                #at the end of the song, so don't worry about small changes
                output.append((note, False))
    return output, change_indices
    
if __name__ == "__main__":
    main()