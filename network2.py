# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:43:26 2017

@author: bbishop
"""
import json
import random
import sys

import numpy as np

class CrossEntropyCost(object):
    
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1 - y)*np.log(1 - a)))
        
    @staticmethod
    def delta(z, a, y):
        return (a - y)
        
        
class QuadraticCost(object):
    
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a - y)**2
    
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoidPrime(z)
        

class Network(object):
    
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.defaultWeightInitializer()
        self.cost = cost
        self.NUM_OUTPUTS = sizes[-1]
        
    def defaultWeightInitializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) \
        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def largeWeightInitializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) \
        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, eta, lmbda, len(training_data))
            #print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(self.accuracy(evaluation_data, n_data))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Epoch {0}: {1} / {2}".format(j, self.accuracy(evaluation_data), n_data)
                #print "Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data)
        print "Best Result: " +  str(max(evaluation_accuracy)) + " / " + str(n_data)
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        
    def updateMiniBatch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backPropogate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta*(lmbda/n))*w - (eta / len(mini_batch))*nw
        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch))*nb
        for b, nb in zip(self.biases, nabla_b)]
            
    def backPropogate(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
        
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedForward(x)), y) for (x, y) in data]
        return sum(int(x==y) for (x, y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedForward(x)
            if convert: y = vectorizedResult(y, self.NUM_OUTPUTS)
            cost += self.cost.fn(a, y) / len(data)
            
        cost += 0.5*(lmbda / len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
        
    def testUnkownData(self, data):
        """
        Run the test data through the network and return what values the network outputs.
        """
        results = [np.argmax(self.feedForward(x)) for x in data]
        return results
        
    def save(self, filename):
        """
        Save the network into a json file. Json is used instead of pickle
        ensures that changes to the network's calculations won't cause the
        loading of older networks to fail.
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

        
def vectorizedResult(j, NUM_OUTPUTS):
    e = np.zeros((NUM_OUTPUTS, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
    
def sigmoidPrime(z):
    return sigmoid(z)*(1 - sigmoid(z))
                
def loadNetwork(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    
    return net