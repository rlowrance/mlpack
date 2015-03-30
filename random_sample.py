import numpy as np
import unittest
import pdb


def rand(d):
    '''Generate random np.array sample of size d drawn uniformly from [0,1]'''
    while True:
        result = np.random.rand(d)
        yield result


def randn(d, mean, var):
    '''Generate random np.array sample of size d drawn from N(mean, var) '''
    while True:
        result = var * np.random.randn(d) + mean
        yield result


def uniform(d, low, high):
    '''Generate np.array 1d sample of size d drawn uniformly from [low,high]'''
    while True:
        result = np.random.uniform(low, high, size=d)
        yield result


class TestUniform(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test(self):
        for i in xrange(10):
            x = next(uniform(3, -10, 100))
            if self.verbose:
                print i, x


class TestRand(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test_generate(self):
        for i in xrange(10):
            x = next(rand(3))
            if self.verbose:
                print i, x

    def test_iterate(self):
        count = 0
        for value in rand(5):
            if self.verbose:
                print value
            count += 1
            if count > 5:
                break


class TestRandn(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test_generate_0_1(self):
        mean = 0
        var = 1
        n = 100
        means = np.zeros(n)
        for i in xrange(n):
            x = next(randn(3, mean, var))
            means[i] = np.mean(x)
            if self.verbose:
                print i, x
        actual_mean = np.mean(means)
        if self.verbose:
            print 'actual_means', actual_mean
        self.assertLess(abs(mean - actual_mean), .5)

    def test_generate_10_100(self):
        mean = 10
        var = 100
        n = 1000
        means = np.zeros(n)
        for i in xrange(n):
            x = next(randn(3, mean, var))
            means[i] = np.mean(x)
            if self.verbose:
                print i, x
        actual_mean = np.mean(means)
        if self.verbose:
            print 'actual_mean', actual_mean
        self.assertLess(abs(mean - actual_mean), 5)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
