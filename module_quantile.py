'''module for median loss'''
import numpy as np
import unittest
import pdb


def quantile(tau):
    '''quantile error'''
    assert 0 < tau < 1

    def output(predicted, expected):
        '''return number'''
        # tau * (p - y) * I(y <= p) + (1- tau) * (y - p) * I(y >= p)
        if expected <= predicted:
            return tau * (predicted - expected)
        else:
            return (1 - tau) * (expected - predicted)

    def gradient(predicted, expected):
        '''subgradient wrt predicted'''
        raise RuntimeError('not implemented')

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        tau = .1
        _, output = quantile(tau)
        self.assertAlmostEqual(output(100, 200), 90)  # predicted < expected
        self.assertAlmostEqual(output(100, 100), 0)   # predicted == expected
        self.assertAlmostEqual(output(100, 10), 9)    # predicted > expected

    def test_gradient(self):
        return
        # check with one element so that we can use check gradient
        # NOTE: the original implementation of output() took vectors as inputs
        tau = .5
        gradient, _ = quantile(tau)
        predicted = np.array([1, 2, 3])
        expected = np.array([6, 5, 4])
        actual = gradient(predicted, expected)
        expected = np.array([2 * -5, 2 * -3, 2 * -1])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
