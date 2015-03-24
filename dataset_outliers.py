'''generate outliers data set'''
import numpy as np
import pdb
import unittest


import random_sample


def outliers(num_features, num_unique_samples):
    '''return theta, x, y

    ARGS
    num_unique_samples: returns 3 * this number of samples

    theta = 2d randomly
    y values have very large positive outliers
    Ref: ML1, Spring 2015, Hw2
    '''
    def full(value):
        return np.full(num_features, value)

    assert num_features >= 1
    assert num_unique_samples > 0

    num_samples = 3 * num_unique_samples

    # the s-th row of s is drawn uniformly from [0,1]
    x = np.zeros((num_samples, num_features))
    for s in xrange(num_unique_samples):
        t = 3 * s
        x[t] = full(-1.1)
        x[t + 1] = full(-0.9)
        x[t + 2] = full(2.0)

    theta = next(random_sample.rand(num_features + 1))

    # y = x theta + eps
    # eps is num_elements x 1 noise vector drawn from N(0, 0.1)
    eps = next(random_sample.randn(num_samples, 0, 0.1))

    # split theta and produce y
    b, w = theta[0], theta[1:]
    y = b + np.dot(x, w) + eps

    return theta, x, y


class Test(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test(self):
        num_features = 10
        num_samples = 3
        theta, x, y = outliers(num_features, num_samples)
        if self.verbose:
            print theta, x, y
            print theta.shape
        self.assertEqual(theta.ndim, 1)
        self.assertEqual(theta.shape[0], num_features + 1)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[0], 3 * num_samples)
        self.assertEqual(x.shape[1], num_features)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.shape[0], 3 * num_samples)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
