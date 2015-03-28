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
from module_linear import linear
from module_linear_n import linear_n
from module_negative_likelihood import negative_likelihood
from module_softmax import softmax


if False:
    # use all imports, so as to avoid an error from pyflakes
    linear()
    linear_n()
    negative_likelihood()
    softmax()
