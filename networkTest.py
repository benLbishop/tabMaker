# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:03:32 2017

@author: bbishop
"""

import Network
import network2
#import mnist_loader
import wave_loader
import operator

NOTE_TABLE = ["a", "b", "c", "d", "e", "f", "g", "as", "cs", "ds", "fs", "gs" , "d8"]   

path = 'convertedFiles/s2.pkl'
all_data = wave_loader.loadWrapper(path)

training_data, validation_data, test_data = all_data[0]
EPOCH_LENGTH, NUM_NOTES = all_data[1]
"""
hidden_sizes = [40, 80, 120, 160, 200, 240, 280]
for h in hidden_sizes:
    print "\nNum hidden nodes: " + str(h) + "\n"
    net = Network.Network([EPOCH_LENGTH, h, NUM_NOTES])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""
print "\nTesting Updated Network\n"
#net2 = network2.Network([EPOCH_LENGTH, 20, NUM_NOTES], cost=network2.CrossEntropyCost)
learning_rates = [1.5, 2.0,2.5, 3.0]
lmbda_vals = [0.2, 0.6, 1.0, 1.4, 5.0]


net2 = network2.Network([EPOCH_LENGTH, 30, 70, NUM_NOTES], cost=network2.CrossEntropyCost)
net2.SGD(training_data, 60, 10, 1.5,
        lmbda = 0.5,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=False,
        monitor_training_cost=False)

test_results = net2.testUnkownData(test_data)
recognized_notes = {}

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
