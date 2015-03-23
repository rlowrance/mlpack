'''module for hinge loss for binary classification to +1 or -1'''
import numpy as np
import unittest
import pdb


def hinge():
    '''hinge loss (use for svm)'''
    def output(predicted, expected):
        '''expected is +1 or -1'''
        return max(0, 1 - predicted * expected)

    def gradient(predicted, expected):
        '''gradient wrt predicted'''
        raise NotImplementedError('gradient')

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = hinge()
        self.assertAlmostEqual(output(-10, -1), 0)
        self.assertAlmostEqual(output(-1.1, -1), 0)
        self.assertAlmostEqual(output(-1.0, -1), 0)
        self.assertAlmostEqual(output(-0.9, -1), 0.1)
        self.assertAlmostEqual(output(0.0, -1), 1)
        self.assertAlmostEqual(output(1.0, -1), 2)
        self.assertAlmostEqual(output(10, -1), 11)

        self.assertAlmostEqual(output(-10, +1), 11)
        self.assertAlmostEqual(output(-1.1, +1), 2.1)
        self.assertAlmostEqual(output(-1.0, +1), 2)
        self.assertAlmostEqual(output(-0.9, +1), 1.9)
        self.assertAlmostEqual(output(0.0, +1), 1)
        self.assertAlmostEqual(output(1.0, +1), 0)
        self.assertAlmostEqual(output(10, +1), 0)

    def test_gradient(self):
        return
        gradient, _ = hinge()
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
