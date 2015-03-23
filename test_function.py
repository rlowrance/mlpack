'''Test functions

All follow a similar API
f, grad, minimizer, minimum = f(parameter)
'''

import pdb
import numpy as np
import unittest
from numpy_utilities import almostEqual
almost_equal = almostEqual
import check_gradient
check = check_gradient.roy

if False:
    pdb.set_trace()


def sphere(d):
    '''return functions f(x), grad(x) and minimizer

    search domain
    x_i any
    1 <= i <= d

    ARGS
    d: number, number of dimensions

    RETURNS
    f: function(x)->number
    g: function(x)->np.array 1d, gradient of f at x

    WHERE
    x: np.array of shape d

    ref: wikipedia at test functions for optimization
    '''
    def f(x):
        return np.sum(x * x)

    def grad(x):
        result = np.zeros(d)
        for i in xrange(d):
            result[i] = 2.0 * x[i]
        return result

    minimizer = np.zeros(d)  # the origin in d dimensions

    minimum = 0.0

    return f, grad, minimizer, minimum


class TestSphere(unittest.TestCase):
    def test_1d(self):
        f, grad, minimizer, minimum = sphere(1)
        x = np.array([3])
        self.assertAlmostEqual(f(x),
                               9.0)
        self.assertTrue(check(f, grad, x, 1e-5, 1e-2))
        self.assertTrue(almostEqual(grad(x),
                                    np.array([2.0 * 3.0]),
                                    1e-5))
        self.assertTrue(almost_equal(minimizer,
                                     np.array([0]),
                                     1e-5))
        self.assertEqual(minimum, 0)
        self.assertEqual(f(minimizer), minimum)

    def test_2d(self):
        f, grad, minimizer, minimum = sphere(2)
        x = np.array([2, 3])
        self.assertAlmostEqual(f(np.array(x)),
                               13.0)
        self.assertTrue(check(f, grad, x, 1e-5, 1e-2))
        self.assertTrue(almostEqual(grad(np.array(x)),
                                    np.array([4.0, 6.0]),
                                    1e-5))
        self.assertTrue(almost_equal(minimizer,
                                     np.array([0, 0]),
                                     1e-5))
        self.assertEqual(minimum, 0)
        self.assertEqual(f(minimizer), minimum)


def rosenbrock(d):
    '''return f(x), grad(x), and minimizer

    search domain
    x_i any
    1 <= i <= d

    ARGS
    d: number, number of dimensions

    RETURNS
    f: function(x)->number
    g: function(x)->np.array 1d, gradient of f at x

    WHERE
    x: np.array of shape d

    ref: wikipedia at test functions for optimization
    '''
    if d != 2:
        raise NotImplementedError('only d == 2 is implemented')

    def f(x):
        x1, x2 = x[0], x[1]
        t1 = x2 - x1 * x1
        t2 = x1 - 1
        return 100 * t1 * t1 + t2 * t2

    def grad(x):
        x1, x2 = x[0], x[1]
        result = np.zeros(2)
        t = x2 - x1 * x1
        result[0] = 200 * t * (-2.0 * x1) + 2.0 * (x1 - 1)
        result[1] = 200 * t
        return result

    minimizer = np.ones(d)

    minimum = 0.0

    return f, grad, minimizer, minimum


class TestRosenbrock(unittest.TestCase):
    def test_1d(self):
        self.assertRaises(RuntimeError, rosenbrock, 1)

    def test_2d(self):
        d = 2
        f, grad, minimizer, minimum = rosenbrock(d)
        x = np.array([2, 3])
        self.assertAlmostEqual(f(x), 101.0)
        self.assertTrue(check(f, grad, x, 1e-5, 1e-2))
        self.assertTrue(almost_equal(grad(x),
                                     np.array([800 + 2,
                                               -200]),
                                     1e-2))
        self.assertTrue(almost_equal(minimizer,
                                     np.array([1, 1]),
                                     1e-2))
        self.assertAlmostEquals(f(minimizer), minimum)


def beales():
    '''return f(x), grad(x), and minimizer

    search domain: -4.5 <= x,y <= 4.5

    RETURNS
    f: function(x)->number
    g: function(x)->np.array 1d, gradient of f at x

    WHERE
    x: np.array of shape d

    ref: wikipedia at test functions for optimization
    '''
    def in_domain(a):
        return -4.5 <= a <= 4.5

    def check_and_unpack_args(x_array):
        x, y = x_array[0], x_array[1]
        if not in_domain(x):
            raise RuntimeError('first argument not in [-4.5,+4.5]; ' + str(x))
        if not in_domain(y):
            raise RuntimeError('first argument not in [-4.5,+4.5]; ' + str(y))
        return x, y

    def terms(x, y):
        return \
            1.5 - x + x * y, \
            2.25 - x + x * y * y, \
            2.625 - x + x * y * y * y

    def f(x_array):
        x, y = check_and_unpack_args(x_array)
        t1, t2, t3 = terms(x, y)

        return t1 * t1 + t2 * t2 + t3 * t3

    def grad(x_array):
        x, y = check_and_unpack_args(x_array)
        t1, t2, t3 = terms(x, y)

        dx = \
            2 * t1 * (-1 + y) + \
            2 * t2 * (-1 + y * y) + \
            2 * t3 * (-1 + y * y * y)
        dy = \
            2 * t1 * x + \
            2 * t2 * 2 * x * y + \
            2 * t3 * 3 * x * y * y

        if False:
            print 't', t1, t2, t3
            print 'x,y', x, y
            print 'dx', dx
            print 'dy', dy
        return np.array([dx, dy])

    minimizer = np.array([3, 0.5])

    minimum = 0

    return f, grad, minimizer, minimum


class TestBeales(unittest.TestCase):

    def setUp(self):
        self.x = np.array([2, 3])
        self.f, self.grad, self.minimizer, self.minimum = beales()

    def test_f(self):
        self.assertAlmostEqual(self.f(self.x),
                               5.5 * 5.5 + 18.25 * 18.25 + 54.625 * 54.625)
        self.assertRaises(RuntimeError,
                          self.f,
                          np.array([5, 0]))
        self.assertRaises(RuntimeError,
                          self.f,
                          np.array([0, -4.501]))

    def test_grad(self):
        self.assertTrue(check(self.f, self.grad, self.x, 1e-5, 1e-2))

    def test_minimizer(self):
        self.assertTrue(almost_equal(self.minimizer,
                                     np.array([3, 0.5]),
                                     1e-5))

    def test_minimum(self):
        self.assertEqual(self.minimum, 0)
        self.assertAlmostEqual(self.f(self.minimizer), self.minimum)


if __name__ == '__main__':
    unittest.main()
