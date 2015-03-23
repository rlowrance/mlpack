'''Line search methods'''

import numpy as np
import pdb
import unittest


if False:
    pdb.set_trace()


def backtracking(f, grad_f_x, x, p, initial_alpha, tau, c, verbose=False):
    '''Backtracking line search

    RETURNS
    alpha: number, the approximate minimizer is f(x + alpha * p)

    ARGS
    f            : function(x)->number, function to be approximately minimized
    grad_f_x     : np.array 1d, gradient of f at x
    x            : np.array 1d, starting point for the minimization
    p            : np.array 1d, direction for the minimization
    initial_alpha: number, upper bound on alpha, lower values not considered
                   start with a reasonably large value
    tau          : number, how much alpha is shrunk on each trial
    c            : number, fraction of slope considered as good enough
    verbose      : boolean, print if True

    ref: wikipedia at backtracking line search
    '''
    assert initial_alpha > 0
    assert 0 <= tau <= 1
    assert 0 <= c <= 1

    m = np.dot(p.T, grad_f_x)  # slope along search line starting at x
    if m >= 0:
        raise RuntimeError('no decrease possible; m = ' + m)

    t = - c * m
    f_x = f(x)
    alpha = initial_alpha

    # while Armijo-Goldstein is not satisfied, reduce alpha
    while not ((f_x - f(x + alpha * p)) >= alpha * t):
        if verbose:
            print alpha, f_x - f(x + alpha * p), alpha * t
        alpha = tau * alpha

    if verbose:
        print alpha, f_x - f(x + alpha * p), alpha * t
        print 'final alpha', alpha

    return alpha


class Test(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test(self):
        def f(x):
            x0 = x[0]
            return x0 * x0 + 2 * x0 - 3

        def grad(x):
            x0 = x[0]
            return 2 * x0 + 2

        def step(x):
            grad_f_x = grad(x)
            step_size = backtracking(f, grad_f_x, x, -grad_f_x, 10, 0.5, 0.1)
            new_x = x - grad_f_x * step_size
            if self.verbose:
                print x, f(x), step_size, new_x, f(new_x)
            return new_x

        x0 = np.array([1])
        x1 = step(x0)
        x2 = step(x1)
        x3 = step(x2)
        if self.verbose:
            print x0, x1, x2, x3
            print f(x0), f(x1), f(x2), f(x3)
        self.assertLess(f(x1), f(x0))
        self.assertLess(f(x2), f(x1))
        self.assertLess(f(x3), f(x2))
        minimizer = np.array([-1])
        self.assertLess(np.linalg.norm(x3 - minimizer), 0.1)


if __name__ == '__main__':
    unittest.main()
