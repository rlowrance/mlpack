'''module for logistic loss for binary classification to +1 or -1'''
import numpy as np
import unittest
import pdb
import math


def logistic():
    '''logistic error'''
    def output(predicted, expected):
        '''expected is +1 or -1'''
        return math.log(1 + math.exp(-predicted * expected))

    def gradient(predicted, expected):
        '''gradient wrt predicted'''
        raise NotImplementedError('gradient')

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = logistic()
        predicted = .9
        self.assertAlmostEqual(output(predicted, +1),
                               math.log(1 + math.exp(-predicted * +1)))
        self.assertAlmostEqual(output(predicted, -1),
                               math.log(1 + math.exp(-predicted * -1)))

    def test_gradient(self):
        return
        gradient, _ = logistic()
        predicted = np.array([1, 2, 3])
        y = np.array([6, 5, 4])
        actual = gradient(predicted, y)
        expected = np.array([2 * -5, 2 * -3, 2 * -1])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
