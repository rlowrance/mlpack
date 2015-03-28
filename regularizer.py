'''regularizers

SUMMARY OF APIs
l1():
    loss(w)                   --> number
    derivative(w, num_biases) --> number
l2():
    loss(w)                   --> number
    derivative(w, num_biases) --> number
'''
from regularizer_l1 import l1
from regularizer_l2 import l2


if False:
    # use all imported functions, so as to avoid an error mesage from pyflakes
    l1()
    l2()
