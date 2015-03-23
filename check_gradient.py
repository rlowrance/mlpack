import pdb
import random_sample
import numpy as np
import unittest
from numeric_gradient import numeric_gradient

if False:
    pdb.set_trace()


def bottou(w, x, y, loss, gradient, delta, tolerance, debug=False):
    '''Use Bottou's recommended approach to check gradient wrt w at x,y.

    Check that loss(w', x, y) ~= loss(w, x, y) + delta * gradient(w, x, y)

    RETURNS
    ok        : True or False, whether the error was less than the tolerance
    w_peturbed: modified w for which condition did not hold, or None

    ARGS
    w        : np.array shape d, weights
    x        : np.array shape d, sample
    y        : number, label for sample
    loss     : function(w, x, y)->number
    gradient : function(w, x, y)-> np.array shape d, gradient wrt w
    delta    : number, how far to step
    tolerance: number, max allowed error

    ref: bottou-12 sgd tricks
    '''

    verbose = False

    def check(delta):
        '''Return ok, w_peturbed, after checking each dimension of w.'''
        if debug:
            pdb.set_trace()
        for j in xrange(w.shape[0]):
            w_prime = w.copy()
            w_prime[j] += delta
            lhs = loss(w_prime, x, y)
            rhs = loss(w, x, y) + delta * gradient(w, x, y)
            if np.linalg.norm(lhs - rhs, 2) >= tolerance:
                if verbose:
                    print 'w', w
                    print 'x', x
                    print 'y', y
                    print 'w_prime', w_prime
                    print 'lhs', lhs
                    print 'rhs', rhs
                    return False, w_prime
        return True, None

    ok_plus, w_plus = check(+delta)
    if not ok_plus:
        return False, w_plus

    ok_minus, w_minus = check(-delta)
    if not ok_minus:
        return False, w_minus

    return True, None


class TestBottou(unittest.TestCase):
    def setUp(self):
        self.verbose = True

        def loss(w, x, y):
            return np.dot(w.T, x) + np.dot(w.T, y) + 3

        def gradient(w, x, y):
            return np.array([x, 2 * y])

        self.loss = loss
        self.gradient = gradient

    def test(self):
        d = 20
        n = 100
        delta = 1e-5
        tolerance = 1e-3
        for i in xrange(n):
            w = next(random_sample.rand(d))
            x = next(random_sample.rand(d))
            y = next(random_sample.rand(d))
            ok, bad_w = bottou(w, x, y,
                               self.loss, self.gradient,
                               delta, tolerance)
            if not ok:
                print 'bad_w', bad_w
            self.assertTrue(ok)


def roy(f, gradient, theta, stepsize, tolerance, verbose=False):
    '''
    Check if gradient is correct.

    Args
    f(theta) -> float              : function
    gradient(theta) -> np.array 1d : function, supposed gradient at theta
    theta                          : point of evalution
    stepsize : number
    tolerance: number
    verbose  : boolean, if True, print details

    Returns
    boolean - True iff numerical derivative within this tolerance in each
              dimensions
    '''
    grad = numeric_gradient(f=f,
                            x=theta,
                            stepsize=stepsize)
    actual_gradient = gradient(theta)
    diff = np.linalg.norm(grad - actual_gradient)
    if verbose:
        print 'numeric gradient', grad
        print 'actual gradient', actual_gradient
        print 'diff', diff
        print 'tolerance', tolerance
    return diff < tolerance


class TestRoy(unittest.TestCase):
    def setUp(self):
        def f(w):
            x = w[0]
            y = w[1]
            return 2 * x * x + 3 * y

        def gradient(w):
            x = w[0]
            # y = w[1]
            return np.float64([4 * x, 3])

        self.f = f
        self.gradient = gradient
        self.x = np.float64([1, 2])
        self.stepsize = 1e-10
        self.tolerance = 1e-5

        def test(self):
            ok = roy(self.f, self.gradient, self.x,
                     self.stepsize, self.tolerance)
            self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
