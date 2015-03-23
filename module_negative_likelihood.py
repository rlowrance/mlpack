'''negative likelihood of probabilities

FUNCTIONS
gradient(probs)  -> vector of zeros
output(probs, y) -> number

ARGS
probs : np.array 1d size s
vector: np.array 1d size s
y     : number of class, y \in {0, 1, ..., num_classes - 1}
'''
import numpy as np
import unittest
import pdb
import check_gradient
import random_sample


def negative_likelihood():
    '''loss = - prob[y]'''
    def output(probs, y):
        return - probs[y]

    def gradient(probs, y):
        result = np.zeros(probs.shape)
        result[y] = -1
        return result

    return gradient, output


class TestNegativeLikelikehood(unittest.TestCase):
    def setUp(self):
        self.probs = np.array([.01, .49, .50])
        self.y = 1

    def test_gradient_check(self):
        # check using check_gradient
        gradient, output = negative_likelihood()

        def f(x):
            return output(x, self.y)

        def g(x):
            return gradient(x, self.y)

        num_tests = 100
        for _ in xrange(num_tests):
            scores = next(random_sample.rand(3))
            probs = scores / scores.sum()
            stepsize = .00001
            tolerance = .001
            ok = check_gradient.roy(f, g, probs, stepsize, tolerance)
            self.assertTrue(ok)

    def test_gradient_known(self):
        # check vs. known result
        gradient, output = negative_likelihood()
        actual = gradient(self.probs, self.y)
        expected = np.array([0, -1, 0])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, .001)

    def test_output(self):
        gradient, output = negative_likelihood()
        actual = output(self.probs, self.y)
        expected = -.49
        diff = abs(actual - expected)
        self.assertLess(diff, .001)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
