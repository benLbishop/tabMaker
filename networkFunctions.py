# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:02:29 2017

@author: bbishop

This file 
"""

import sys
import network2
import wave_loader

def createNetwork(filename):
    all_data = wave_loader.loadWrapper(filename)
    
    training_data, validation_data, test_data = all_data[0]
    EPOCH_LENGTH, NUM_NOTES = all_data[1]
    
    net = network2.Network([EPOCH_LENGTH NUM_NOTES], cost=network2.CrossEntropyCost)
    
def optimize_network(filename):
    net = network2.loadNetwork(filename)
    
    

