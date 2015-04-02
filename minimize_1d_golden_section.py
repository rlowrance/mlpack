'''golden section search to minimize a function of one variable in [low,high]

NOTE: the function fun is assumed to be unimodal

RETURN
low
high, such that the minimizer is in [low,high]
dict, dictionary with function evalutions dict[x] = f(x)

ARGS
fun(x) -> number
low    -> number
high   -> number
tolerance -> number, required absolute precision of fun(x)

ref: heath-02 p.270 golden section search
'''
import math
import pdb
import unittest


def golden_section(fun, low, high, tolerance, verbose=False):
    assert low < high
    d = {}

    def func(x):
        result = fun(x)
        d[x] = result
        return result

    tau = (math.sqrt(5.0) - 1) / 2.0
    x1 = low + (1 - tau) * (high - low)
    f1 = func(x1)
    x2 = low + tau * (high - low)
    f2 = func(x2)
    while (high - low) > tolerance:
        if verbose:
            print x1, f1, x2, f2
        if f1 > f2:
            low = x1
            x1 = x2
            f1 = f2
            x2 = low + tau * (high - low)
            f2 = func(x2)
        else:
            high = x2
            x2 = x1
            f2 = f1
            x1 = low + (1 - tau) * (high - low)
            f1 = func(x1)
    return low, high, d


class Test(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test(self):
        # from heath-02 p. 272
        def fun(x):
            return 0.5 - x * math.exp(- x * x)

        low_star, high_star, d = \
            golden_section(fun, 0.0, 2.0, .001, verbose=self.verbose)
        if self.verbose:
            print 'low_star', low_star, 'high_star', high_star
        self.assertLess(abs(low_star - .706565), 1e-3)
        self.assertLess(abs(high_star - .707471), 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
