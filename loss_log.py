'''module for log loss for binary classification to +1 or -1'''
import unittest
import pdb
import math


def log(large=18):
    '''loss = log(1 + exp(-label * predicted))

    constrain the error (z = predicted * expected) to be less than large
    default value for large is from Bottou
    ref: leon.bottout.org/projects/sgd source file loss.h
    '''
    def output(prediction, label):
        '''label is +1 or -1'''
        if label != 1 and label != -1:
            raise RuntimeError('label not =1 or +1, as required: ' + str(label))
        # avoid overflow by using Bottou's trick
        # ref: leon.bottou.org/projects/sgd source file loss.h
        z = prediction * label
        if z > large:
            return math.exp(-z)
        elif z < -large:
            return -z
        else:
            return math.log(1 + math.exp(-z))

    def derivative(prediction, label):
        '''derivative wrt prediction'''
        if label != 1 and label != -1:
            raise RuntimeError('label not =1 or +1, as required: ' + str(label))
        z = prediction * label
        if z > large:
            return label * math.exp(-z)
        elif z < -large:
            return label
        else:
            return -label / (1 + math.exp(z))

    return derivative, output


class Test(unittest.TestCase):
    def test_output(self):
        large = 18
        _, output = log(large)

        def close(actual, expected):
            self.assertLess(abs(actual - expected), 1e-3)

        # z not large
        close(output(10, 1), math.log(1 + math.exp(-10)))
        close(output(10, -1), math.log(1 + math.exp(10)))

        # z large
        close(output(20, 1), math.exp(-20))

        # z small
        close(output(20, -1), 20)

    def test_derivative_known(self):
        large = 18
        derivative, _ = log(large)

        def equal(actual, expected1):
            diff = abs(actual - expected1)
            self.assertLess(diff, 1e-3)

        # small z values
        equal(derivative(10, 1), -1/(1 + math.exp(10)))
        equal(derivative(10, -1), 1/(1 + math.exp(-10)))

        # large z values
        equal(derivative(20, 1), math.exp(-20))

        # small z values
        equal(derivative(20, -1), -1)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
