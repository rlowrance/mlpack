'''ordinary least squares with one prediction

FUNCTIONS
predict(theta, x)              -> predicted number
loss(predicted, y)             -> number
gradient_loss(x, y, predicted) -> np.array 1d, number

ARGS
predicted: number
x        : np.array 1d
y        : number
'''
import numpy as np
import pdb
import unittest
import module


def ols():
    '''1 output, because ridge will allow multiple outputs'''
    linear_gradient, linear_output = module.linear()
    squared_gradient, squared_output = module.squared()

    def gradient_loss(x, y, predicted):
        g_squared = squared_gradient(predicted, y)
        g_linear = linear_gradient(x)
        g = g_linear * g_squared
        loss = squared_output(predicted, y)
        return g, loss

    def predict(theta, x):
        return linear_output(theta, x)

    return gradient_loss, predict


class Test(unittest.TestCase):
    def test_gradient_loss(self):
        gradient, predict = ols()
        theta = np.array([1, -2, 3])
        x = np.array([-4, 5])
        y = -6
        predicted = predict(theta, x)
        the_gradient, the_loss = gradient(x, y, predicted)
        # check the_gradient
        expected = np.array([60, -240, 300])
        diff = np.linalg.norm(the_gradient - expected)
        self.assertLess(diff, 1e-3)
        # check the_loss
        error = 1 - 2 * -4 + 3 * 5 - -6
        self.assertEqual(the_loss, error * error)

    def test_predict(self):
        _, predict = ols()
        theta = np.array([1, -2, 3])
        x = np.array([-4, 5])
        predicted = predict(theta, x)
        expected = 1 - 2 * -4 + 3 * 5
        self.assertEqual(predicted, expected)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
