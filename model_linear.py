'''linear regression with one output

FUNCTIONS returned by ridge(l1,l2)
gradient(thta, x, y, [prediction])) -> vector
loss(thta, x, y, [prediction]))     -> number
predict(theta, x)                   -> number

ARGS
l1                : number, weight of l1 regularizer
l2                : number, weight of l2 regularizer
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


def linear(l1, l2):
    assert l1 >= 0
    assert l2 >= 0
    linear_gradient, linear_output = module.linear()
    squared_derivative, squared_loss = loss.squared()
    l2_derivative, l2_loss = regularizer.l2()
    l1_derivative, l1_loss = regularizer.l1()

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
            l1 * l1_loss(w) + \
            l2 * l2_loss(w)

    def gradient(theta, x, y, prediction=None):
        b, w = split(theta)
        num_biases = b.size
        if prediction is None:
            prediction = predict(theta, x)
        part1 = linear_gradient(theta, x) * squared_derivative(prediction, y)
        part2 = l1 * l1_derivative(w, num_biases)
        part3 = l2 * l2_derivative(w, num_biases)
        return part1 + part2 + part3

    return gradient, loss_, predict


class TestTest(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([-1.0, 2, -3])
        self.x = np.array([4.0, -5])
        self.y = 6
        self.l1 = 0.2
        self.l2 = 0.1

    def test_predict(self):
        _, _, predict = linear(self.l1, self.l2)
        actual = predict(self.theta, self.x)
        self.assertTrue(isinstance(actual, float))
        expected = -1 + 2 * 4 - 3 * -5
        self.assertAlmostEqual(actual, expected)

    def test_loss(self):
        _, loss, _ = linear(self.l1, self.l2)
        actual = loss(self.theta, self.x, self.y)
        self.assertTrue(isinstance(actual, float))
        expected_error = 22 - 6
        expected_loss = expected_error * expected_error + 1.0 + 1.3
        self.assertAlmostEqual(actual, expected_loss)

    def test_gradient(self):
        gradient, _, _ = linear(self.l1, self.l2)
        actual = gradient(self.theta, self.x, self.y)
        expected_error = 22 - 6
        gradient_no_reg = np.array([1, 4, -5]) * 2 * expected_error
        gradient_l1 = .2 * np.array([0, 1, -1])
        gradient_l2 = .1 * 2.0 * np.array([0, 2, -3])
        expected = gradient_no_reg + gradient_l1 + gradient_l2
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

        for l1 in (0, .001, .1):
            for l2 in (0, .002, .2):
                gradient, loss, _ = linear(l1, l2)
                check(gradient, loss)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
