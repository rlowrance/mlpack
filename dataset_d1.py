'''dataset_d1: generate dataset with 1 feature and 1 label'''
import numpy as np
import pdb
import unittest


import random_sample


def d1(fun, num_samples, x_low, x_high, error_mean, error_variance):
    '''generate label := fun(x) + N(error_mean, error_variance)

    ARGS
    fun: function(number) -> number
    num_samples: number > 0
    x_low, x_high: number, x sampled from uniform [x_low, x_high]
    error_mean, error_variance: number, noise sampled from Normal distribution

    RETURNS
    x: np array 2d, shape num_samples x 1
    y: np array 1d
    '''
    x_dimensions = 1
    y_dimensions = 1
    x = np.zeros((num_samples, x_dimensions))
    y = np.zeros(num_samples)
    for i in xrange(num_samples):
        x_value = next(random_sample.uniform(x_dimensions, x_low, x_high))
        error = \
            next(random_sample.randn(y_dimensions, error_mean, error_variance))
        y_value = fun(x_value) + error
        x[i] = np.array([x_value])
        y[i] = y_value
    return x, y


class Test(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test(self):
        def fun(x):
            return 2.0 * x - 3.0

        num_samples = 10
        x_low = -1
        x_high = 1
        error_mean = 0
        error_variance = 0.1
        x, y = d1(fun, num_samples,
                  x_low, x_high,
                  error_mean, error_variance)
        if self.verbose:
            print x
            print y
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[0], num_samples)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.shape[0], num_samples)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
