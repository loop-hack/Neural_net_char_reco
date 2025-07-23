import json
import random
import sys

import numpy as np

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)
    
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        return (a - y)
    
class Network(object):
    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost


def default_weight_initializer(self):
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y, x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

def large_weight_intializer(self):
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
        return a
    
def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
        evaluation_data = None,
        monitor_evaluation_cost = False,
        monitor_evaluation_accuracy = False,
        monitor_training_cost = False,
        monitor_training_accuracy = False):
    
    if evaluation_data : n_data = len(evaluation_data)
    n = len(training_data)
    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print(f"Epoch {j} training complete")
        if monitor_training_cost:
            cost = self.total_cost(training_data, lmbda)
            training_cost.append(cost)
            print(f"Cost on training data: {cost}")
        if monitor_training_accuracy:
            accuracy = self.accuracy(training_data, convert = True)
            training_accuracy.append(accuracy)
            print(f"Accuracy on training data: { accuracy} / {n}")
        if monitor_evaluation_cost:
            cost = self.total_cost(evaluation_data, lmbda, convert = True)
            evaluation_cost.append(cost)
            print(f"Cost on evaluation data: {cost}")
        if monitor_evaluation_accuracy:
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data}")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


def update_mini_batch(self, mini_batch, eta, lmbda, n):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]