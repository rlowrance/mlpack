'''models

SUMMARY OF APIs
multiclass(num_inputs, num_classes):
    gradient_loss(theta, x, y, probs) --> vector, number
    predict(theta, x) --> probs
ols(num_inputs):
    gradient(theta, x, y, [prediction]) -> vector
    loss(theta, x, y, [prediction])     -> number
    predict(theta, x)                   -> number
ridge(num_inputs, regularizer_weights):
    gradient(theta, x, y, [prediction]) -> vector
    loss(theta, x, y, [prediction])     -> number
    predict(theta, x)                   -> number
'''
import model_multiclass
multiclass = model_multiclass.multiclass

import model_ols
ols = model_ols.ols

import model_ridge
ridge = model_ridge.ridge
