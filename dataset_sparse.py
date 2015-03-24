'''generate sparse data set'''
import numpy as np
import pdb
import unittest


import random_sample


def dataset_sparse(num_features, num_samples):
    '''return theta, x, y

    theta = 10 * {-10,+10} + zeros
    '''
    assert num_features > 10

    # the s-th row of s is drawn uniformly from [0,1]
    x = np.zeros((num_samples, num_features))
    for s in xrange(num_samples):
        x[s] = next(random_sample.rand(num_features))

    # the first 10 components of theta are -10 and +10 randomly
    # the remaining components are zero
    theta = np.zeros(num_features + 1)
    for d in xrange(10):
        if next(random_sample.rand(1)) > .5:
            theta[d] = 10
        else:
            theta[d] = -10

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
        num_features = 20
        num_samples = 100
        theta, x, y = dataset_sparse(num_features, num_samples)
        if self.verbose:
            print theta, x, y
            print theta.shape
        self.assertEqual(theta.ndim, 1)
        self.assertEqual(theta.shape[0], num_features + 1)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[0], num_samples)
        self.assertEqual(x.shape[1], num_features)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.shape[0], num_samples)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
