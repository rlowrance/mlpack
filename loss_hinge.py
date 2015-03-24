'''module for hinge loss for binary classification to +1 or -1'''
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

    def derivative(predicted, expected):
        '''derivative wrt predicted'''
        if predicted * expected >= 1:
            return 0.0
        else:
            return -expected

    return derivative, output


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

    def test_derivative(self):
        derivative, _ = hinge()

        def equal(actual, expected1):
            diff = abs(actual - expected1)
            self.assertLess(diff, 1e-3)

        equal(derivative(10, 1), 0)
        equal(derivative(1, 1), 0)
        equal(derivative(.5, 1), -1)
        equal(derivative(0, 1), - 1)
        equal(derivative(-10, 1), -1)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
