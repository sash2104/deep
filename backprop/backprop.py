#!/usr/bin/env python
# -*- coding:utf-8 -*-

""" minimal implementation of back propagation. you can learn XOR with this code.
-------MODEL INFORMATION-------
activation function:
    input -> hidden: sigmoid
    hidden -> output: identify

layer:
    input: 2
    hidden: 4
    output: 1

loss:
    square error
-------------------------------
"""


import math
import random
import sys

ERROR_THRESHOLD = 1e-4
MAX_EPOCH = 100000

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    # differential function of sigmoid
    return sigmoid(x) * (1 - sigmoid(x))

def dot(v1, v2):
    assert(len(v1) == len(v2))
    return sum(x1 * x2 for x1, x2 in zip(v1, v2))

def squareError(ys, ts):
    # ys is estimated values and ts is correct values
    assert(len(ys) == len(ts))
    error = 0.5 * sum((y-t)*(y-t) for y, t in zip(ys, ts))
    return error

class BackPropagation:
    def __init__(self, num_hidden=4, learning_rate=1.0, bias=False):
        self.lr = learning_rate
        self.num_input = 2
        self.num_hidden = num_hidden
        self.num_output = 1
        self.bias_offset = 1 if bias else 0
        # self.initWeight(self.num_input, self.num_hidden)
        self.initWeight(self.num_hidden, self.num_output)

        # weight from input to hidden
        self.w_hi = self.initWeight(self.num_input, self.num_hidden)
        # weight from hidden to output
        self.w_oh = self.initWeight(self.num_hidden, self.num_output)
        # outputs of hidden layer
        self.y_h = [0.0 for _ in xrange(self.num_hidden)]
        # outputs of output layer
        self.y_o = [0.0 for _ in xrange(self.num_output)]

    def initWeight(self, num_input, num_output):
        w = []
        for o in xrange(num_output):
            w.append([random.uniform(0.0, 1.0)
                      for _ in xrange(num_input + self.bias_offset)])
        return w

    def forward(self, xs, ys, w_yx, output=False):
        assert(len(ys) == len(w_yx))
        xs = [1] * self.bias_offset + xs
        for i in xrange(len(ys)):
            ys[i] = dot(w_yx[i], xs)
            if not output:
                ys[i] = sigmoid(ys[i])

    def backward(self, xs, ts):
        assert(len(self.y_o) == len(ts))
        # weight update of output layer
        for o in xrange(self.num_output):
            for h in xrange(self.num_hidden):
                delta = self.lr * (self.y_o[o] - ts[o]) * self.y_h[h]
                self.w_oh[o][h] -= delta
        # weight update of hidden layer
        for h in xrange(self.num_hidden):
            o_delta = sum((self.y_o[o] - ts[o]) * self.w_oh[o][h]
                           for o in xrange(self.num_output))
            x_h = dot(self.w_hi[h], xs)
            for i in xrange(self.num_input):
                delta = self.lr * o_delta * d_sigmoid(x_h) * xs[i]
                self.w_hi[h][i] -= delta

    def learn(self, xs, ts):
        self.forward(xs, self.y_h, self.w_hi)
        self.forward(self.y_h, self.y_o, self.w_oh)
        self.backward(xs, ts)

    def evaluate(self, epoch, dataset):
        error = 0.0
        print 'Epoch', epoch
        for sample in dataset:
            xs, ts = sample[0], sample[1]
            self.forward(xs, self.y_h, self.w_hi)
            self.forward(self.y_h, self.y_o, self.w_oh)
            error += squareError(self.y_o, ts)
            print ' ', xs, ts, self.y_o
        print ' Error:', error
        return error

def main():
    samples = [[[0, 0], [0]],
               [[0, 1], [1]],
               [[1, 1], [0]],
               [[1, 0], [1]]]
    bp = BackPropagation()
    epoch = 0
    while epoch < MAX_EPOCH:
        xs, ts = samples[epoch % len(samples)]
        bp.learn(xs, ts)
        if epoch % 1000 == 0:
            error = bp.evaluate(epoch, samples)
            if error < ERROR_THRESHOLD:
                print "SUCCEED!"
                break
        epoch += 1
    else:
        print "LEARNING FAILED ... TRY ONCE MORE, PLEASE"
    print
    print "WEIGHT (input -> hidden)"
    print '\n'.join(map(str, bp.w_hi))
    print
    print "WEIGHT (hidden -> output)"
    print '\n'.join(map(str, bp.w_oh))

if __name__ == '__main__':
    main()
