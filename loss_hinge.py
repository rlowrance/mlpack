'''module for hinge loss for binary classification to +1 or -1'''
import numpy as np
import unittest
import pdb


def hinge():
    '''hinge loss (use for svm)'''
    def output(predicted, expected):
        '''expected is +1 or -1'''
        if expected == 1 or expected == -1:
            pass
        else:
            print 'expected is not -1 or +1, as hinge loss expects', expected
            raise RuntimeError('expected not =1 or +1, as required' + expected)
        error = 1 - predicted * expected
        if error > 0:
            return error
        else:
            return 0.0

    def gradient(predicted, expected):
        '''gradient wrt predicted'''
        if predicted * expected >= 1:
            return np.array([0.0])
        else:
            return np.array([-expected])

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
        gradient, _ = hinge()

        def equal(actual, expected1):
            expected = np.array([expected1])
            diff = actual - expected
            self.assertLess(np.linalg.norm(diff), 1e-3)

        equal(gradient(10, 1), 0)
        equal(gradient(1, 1), 0)
        equal(gradient(.5, 1), -1)
        equal(gradient(0, 1), - 1)
        equal(gradient(-10, 1), -1)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
