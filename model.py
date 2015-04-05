'''models

SUMMARY OF APIs
multiclass(num_inputs, num_classes):
    gradient_loss(theta, x, y, probs) --> vector, number
    predict(theta, x) --> probs
linear(l1, l2):
    gradient(theta, x, y [,prediction]) -> vector
    loss(theta, x, y [,prediciton])     -> number
    predict(theta, x)                   -> number
'''
from model_linear import linear
from model_multiclass import multiclass
