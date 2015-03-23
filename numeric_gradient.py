import numpy as np
import unittest
import pdb


def numeric_gradient(f, x, stepsize):
    '''
    Numeric gradient of function f at x with specified stepsize.

    Args
    f       : function from 1D numpy array to number
    x       : 1D numpy array, evaluation point
    stepsize: number, the gradient is estimated in this radius

    Returns
    gradient: 1D numpy array of same shape as x
    '''

    num_dimensions = x.size
    gradient = np.zeros(num_dimensions)

    for d in xrange(num_dimensions):
        step = np.zeros(num_dimensions)
        step[d] += stepsize
        if False:
            print 'x', x
            print 'step', step
            print 'stepsize', stepsize
            print 'x+step', x + step
            print 'd', d
            print 'f(x+step)', f(x + step)
            value = (f(x + step) - f(x - step)) / (2 * stepsize)
            print 'value', value
        try:
            gradient[d] = (f(x + step) - f(x - step)) / (2 * stepsize)
        except ValueError:
            print 'ValueError numeric_gradient'
            print 'value', value
            pdb.set_trace()

    return gradient


class Test(unittest.TestCase):

    def setUp(self):
        def f(w):
            x = w[0]
            y = w[1]
            return 2 * x * y + 3 * (x ** 2) + 4 * (y ** 3)

            self.f = f
            self.x = np.float64([10, 20])
            self.stepsize = 1e-10

        def test_one(self):
            actual = numeric_gradient(self.f, self.x, self.stepsize)
            expected = np.float64([100, 4820])
            places = 0
            self.assertAlmostEqual(actual[0], expected[0], places)
            self.assertAlmostEqual(actual[1], expected[1], places)

if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
