'''linear with 1 output'''
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
        return b + np.dot(x, w.T)

    def gradient(x):
        '''return np.array 1d with |theta| elements'''
        return np.hstack((1, x))

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = linear()
        theta = np.array([-1, 2, -3])
        x = np.array([4, -5])
        actual = output(theta, x)
        expected = -1 + 2 * 4 - 3 * -5
        self.assertEqual(actual, expected)

    def test_gradient(self):
        gradient, _ = linear()
        x = np.array([4, -5])
        actual = gradient(x)
        expected = np.hstack((1, x))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
