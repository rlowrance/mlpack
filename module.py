'''modules

SUMMARY OF APIs
l2():
    output(theta)   --> number
    gradient(theta) --> vector
linear():
    output(theta, x) --> scores
    gradient(x)      --> vector
linear_n(num_inputs, num_outputs):
    output(theta, x) --> scores
    gradient(x)      --> vector
negative_likelihood():
    output(probs, y)    --> number
    gradient(probs, y)  --> vector
softmax():
    output(input) --> vector
    gradient(output)
squared():
    output(predicted, expected) --> number
    gradient(predicted, expected) --> vector
'''
import module_l2
l2 = module_l2.l2

import module_linear
linear = module_linear.linear

import module_linear_n
linear_n = module_linear_n.linear_n

import module_negative_likelihood
negative_likelihood = module_negative_likelihood.negative_likelihood

import module_softmax
softmax = module_softmax.softmax

import module_squared
squared = module_squared.squared
