'''ridge regression with multiple outputs

FUNCTIONS
predict(theta, x)              -> predicted vector
loss(predicted)                -> number
gradient_loss(x, y, predicted) -> vector and number

ARGS
predicted: np array 1d
theta    : np array 1d
x        : np array 1d
y        : number
'''
import numpy as np
import unittest
import pdb
import module


def ridge(num_inputs, regularizer_weight):
    assert regularizer_weight >= 0
    num_outputs = 1
    linear_grad, linear_output = module.linear_n(num_inputs, 1)
    squared_grad, squared_output = module.squared()
    l2_grad, l2_output = module.l2(num_outputs)

    def predict(theta, x):
        return linear_output(theta, x)

    def gradient_loss(theta, x, y, prediction):
        loss_squared = squared_output(prediction, y)
        loss_l2 = l2_output(theta)
        g_linear = linear_grad(x)
        g_squared = squared_grad(prediction, y)
        g_l2 = l2_grad(theta)
        g = g_linear * g_squared + regularizer_weight * g_l2
        return g, loss_squared + regularizer_weight * loss_l2

    return gradient_loss, predict


class TestRidge(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([-1, 2, -3])
        self.x = np.array([4, -5])
        self.y = 6
        self.rw = 0.1  # regularizer weight

    def test_predict(self):
        gradient_loss, predict = ridge(self.x.size, self.rw)
        actual = predict(self.theta, self.x)
        expected = -1 + 2 * 4 - 3 * -5
        self.assertAlmostEqual(actual, expected)

    def test_gradient_loss(self):
        gradient_loss, predict = ridge(self.x.size, self.rw)
        prediction = predict(self.theta, self.x)
        gradient, loss = gradient_loss(self.theta, self.x, self.y, prediction)
        expected_error = 22 - 6
        expected_loss = expected_error * expected_error + 1.3
        self.assertAlmostEqual(loss, expected_loss)
        expected_gradient = np.array([1, 4, -5]) * 2 * expected_error
        eg = expected_gradient + 0.1 * np.array([0, 2 * 2, 2 * -3])
        diff = np.linalg.norm(gradient - eg)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
