'''ridge regression with one output

FUNCTIONS returned by ridge(regularizer_weight)
gradient(thta, x, y, [prediction])) -> vector
loss(thta, x, y, [prediction]))     -> number
predict(theta, x)                   -> number

ARGS
prediction        : np array 1d
regularizer_weight: number >= 0
theta             : np array 1d
x                 : np array 1d
y                 : number
'''
import numpy as np
import unittest
import pdb


import check_gradient
import loss
import module
import random_sample
import regularizer


def ridge(regularizer_weight):
    assert regularizer_weight >= 0
    linear_gradient, linear_output = module.linear()
    squared_derivative, squared_loss = loss.squared()
    l2_derivative, l2_loss = regularizer.l2()

    def split(theta):
        # split theta into bias and weights
        return theta[0], theta[1:]

    def predict(theta, x):
        return linear_output(theta, x)

    def loss_(theta, x, y, prediction=None):
        b, w = split(theta)
        if prediction is None:
            prediction = predict(theta, x)
        return \
            squared_loss(prediction, y) + \
            regularizer_weight * l2_loss(w)

    def gradient(theta, x, y, prediction=None):
        b, w = split(theta)
        num_biases = b.size
        if prediction is None:
            prediction = predict(theta, x)
        part1 = linear_gradient(theta, x) * squared_derivative(prediction, y)
        part2 = regularizer_weight * l2_derivative(w, num_biases)
        return part1 + part2

    return gradient, loss_, predict


class TestRidge(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([-1.0, 2, -3])
        self.x = np.array([4.0, -5])
        self.y = 6
        self.rw = 0.1  # regularizer weight

    def test_predict(self):
        gradient, loss, predict = ridge(self.rw)
        actual = predict(self.theta, self.x)
        self.assertTrue(isinstance(actual, float))
        expected = -1 + 2 * 4 - 3 * -5
        self.assertAlmostEqual(actual, expected)

    def test_loss(self):
        gradient, loss, predict = ridge(self.rw)
        prediction = predict(self.theta, self.x)
        actual = loss(self.theta, self.x, self.y, prediction)
        self.assertTrue(isinstance(actual, float))
        expected_error = 22 - 6
        expected_loss = expected_error * expected_error + 1.3
        self.assertAlmostEqual(actual, expected_loss)

    def test_gradient(self):
        gradient, loss, predict = ridge(self.rw)
        actual = gradient(self.theta, self.x, self.y)
        expected_error = 22 - 6
        gradient_no_reg = np.array([1, 4, -5]) * 2 * expected_error
        gradient_reg = 2.0 * 0.1 * np.array([0, 2, -3])
        expected = gradient_no_reg + gradient_reg
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)

    def test_gradient_via_check_gradient(self):
        def check(gradient, loss):
            def f(theta):
                return loss(theta, self.x, self.y)

            def g(theta):
                return gradient(theta, self.x, self.y)

            for _ in xrange(100):
                theta = next(random_sample.rand(1 + self.x.size))
                stepsize = 1e-4
                tolerance = 1e-3
                ok = check_gradient.roy(f, g, theta, stepsize, tolerance)
                self.assertTrue(ok)

        for rw in (0, .001, .1):
            gradient, loss, predict = ridge(rw)
            check(gradient, loss)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
