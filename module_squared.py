'''module for squared loss'''
import numpy as np
import unittest
import pdb


def squared():
    '''squared error'''
    def output(predicted, expected):
        error = predicted - expected
        return np.dot(error, error)

    def gradient(predicted, expected):
        '''gradient wrt predicted'''
        error = predicted - expected
        return np.array([2.0 * error])
        pass

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = squared()
        predicted = np.array([1, 2, 3])
        y = np.array([6, 5, 4])
        actual = output(predicted, y)
        expected = 25 + 9 + 1
        self.assertEqual(actual, expected)

    def test_gradient(self):
        gradient, _ = squared()
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
