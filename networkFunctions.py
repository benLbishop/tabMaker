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

def main():
    #createNetwork(sys.argv[1], sys.argv[2])
    testNetwork(sys.argv[1], sys.argv[2])

def createNetwork(filename, dest_filename):
    all_data = wave_loader.loadWrapper(filename)
    
    net = optimize_network(all_data)
    net.save(dest_filename)
    
def optimize_network(data):
    training_data, validation_data, test_data = data[0]
    EPOCH_LENGTH, NUM_NOTES = data[1]
    
    max_neurons = NUM_NOTES*5
    num_steps = 10
    neuron_step = int((max_neurons / num_steps) + 0.5)
    neuron_range = range(max_neurons)
    neuron_range = neuron_range[0:max_neurons:neuron_step]
    
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
    
def testNetwork(fil, filename):
    net = network2.loadNetwork(filename)
    all_data = wave_loader.loadWrapper(fil)
    data = all_data[0][2]
    
    test_results = net.testUnkownData(data)
    recognized_notes = {}
    NOTE_TABLE = ["a", "b", "c", "d", "e", "f", "g", "as", "cs", "ds", "fs", "gs" , "d8"]
    
    for r in test_results:
        cur_note = NOTE_TABLE[r]
        if cur_note in recognized_notes:
            recognized_notes[cur_note] += 1
        else: 
            recognized_notes[cur_note] = 1
    sorted_notes = sorted(recognized_notes.items(), key=operator.itemgetter(1), reverse=True)
    print "recognized notes, sorted by frequency: "
    for sn in sorted_notes:
        print str(sn[0]) + ": " + str(sn[1])
    
    
if __name__ == "__main__":
    main()