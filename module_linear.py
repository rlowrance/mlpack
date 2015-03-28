'''linear with 1 output

FUNCTIONS
output(theta, x)     -> number
derivative(theta, x) -> vector

ARGS
theta: np array 1d
x    : np array 1d
'''
import numpy as np
import unittest
import pdb


def linear():

    def split(theta):
        '''return bias and weights'''
        b = theta[:1]
        w = theta[1:]
        return b, w

    def output(theta, x):
        '''return number'''
        assert x.ndim == 1  # however, code works for x.ndim == 2
        b, w = split(theta)
        result = b + np.dot(x, w.T)  # this is an np.array
        assert result.size == 1
        return result[0]  # return a number

    def gradient(theta, x):
        '''return np.array 1d with |theta| elements'''
        assert theta.size == x.size + 1
        return np.hstack((1, x))

    return gradient, output


class Test(unittest.TestCase):
    def setUp(self):
        self.x = np.array([4.0, -5])
        self.theta = np.array([-1.0, 2, -3])

    def test_output(self):
        _, output = linear()
        actual = output(self.theta, self.x)
        self.assertTrue(isinstance(actual, float))
        expected = -1 + 2 * 4 - 3 * -5
        self.assertEqual(actual, expected)

    def test_gradient(self):
        gradient, _ = linear()
        actual = gradient(self.theta, self.x)
        expected = np.hstack((1, self.x))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
