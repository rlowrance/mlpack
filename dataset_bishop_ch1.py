'''dataset from bishop-06 chapter 1'''
import math
import numpy as np
import pdb
import unittest


import random_sample


def bishop_ch1(num_samples, error_variance):
    def fun(x):
        return math.sin(2.0 * math.pi * x)

    x_values = np.linspace(start=0.0,
                           stop=1.0,
                           num=num_samples,
                           endpoint=True,
                           retstep=False)

    error_mean = 0
    x = np.zeros((num_samples, 1))
    y = np.zeros(num_samples)
    for i in xrange(num_samples):
        x[i] = x_values[i]
        error_vector = next(random_sample.randn(1, error_mean, error_variance))
        y[i] = fun(x[i][0]) + error_vector[0]
    return x, y


class Test(unittest.TestCase):
    def setUp(self):
        self.num_samples = 10
        self.verbose = False

    def test(self):
        x, y = bishop_ch1(self.num_samples, 0.1)
        if self.verbose:
            for i in xrange(self.num_samples):
                print x[i][0], y[i]
        self.assertEqual(x.shape, (self.num_samples, 1))
        self.assertEqual(y.shape, (self.num_samples,))


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
