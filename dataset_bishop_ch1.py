'''dataset from bishop-06 chapter 1'''
import math
import pdb
import unittest


from dataset_d1 import d1


def bishop_ch1(num_samples):
    def fun(x):
        return math.sin(2.0 * math.pi * x)

    x_low = 0
    x_high = 1
    error_mean = 0
    error_variance = 0.1
    return \
        d1(fun, num_samples, x_low, x_high, error_mean, error_variance)


class Test(unittest.TestCase):
    def setUp(self):
        self.num_samples = 10
        self.verbose = False

    def test(self):
        x, y = bishop_ch1(self.num_samples)
        if self.verbose:
            for i in xrange(self.num_samples):
                print x[i][0], y[i]
        self.assertEqual(x.shape, (self.num_samples, 1))
        self.assertEqual(y.shape, (self.num_samples,))


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
