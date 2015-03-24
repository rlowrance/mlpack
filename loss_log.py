'''module for log loss for binary classification to +1 or -1'''
import numpy as np
import unittest
import pdb
import math


def log():
    '''loss = log(1 + exp(-label * predicted)'''
    def output(predicted, expected):
        '''expected is +1 or -1'''
        if expected == 1 or expected == -1:
            pass
        else:
            print 'expected is not -1 or +1, as hinge loss expects', expected
            raise RuntimeError('expected not =1 or +1, as required' + expected)
        return math.log(1 + math.exp(-expected * predicted))

    def gradient(predicted, expected):
        '''gradient wrt predicted'''
        return -expected / (1 + math.exp(expected * predicted))

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = log()

        def close(actual, expected):
            self.assertLess(abs(actual - expected), 1e-3)

        close(output(10, 1), math.log(1 + math.exp(-10)))
        close(output(10, -1), math.log(1 + math.exp(10)))

    def test_gradient(self):
        gradient, _ = log()

        def equal(actual, expected1):
            expected = np.array([expected1])
            diff = actual - expected
            self.assertLess(np.linalg.norm(diff), 1e-3)

        equal(gradient(10, 1), -1/(1 + math.exp(10)))
        equal(gradient(10, -1), 1/(1 + math.exp(-10)))


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
