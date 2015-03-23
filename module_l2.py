'''l2 loss'''
import numpy as np
import unittest
import pdb
import check_gradient
import random_sample


def l2(num_outputs):
    # num_outputs needed to deconstruct theta
    # theta begins with a bias for each output
    assert num_outputs >= 1

    def split_w(theta):
        '''weight portion of theta'''
        # see linear%split
        return theta[num_outputs:]

    def output(theta):
        '''sum of squared weights'''
        w = split_w(theta)
        return np.sum(np.dot(w, w))

    def gradient(theta):
        '''vector'''
        grad_b = np.zeros(num_outputs)
        grad_w = 2.0 * split_w(theta)
        return np.hstack((grad_b, grad_w))

    return gradient, output


class Test(unittest.TestCase):
    def test_1_output(self):
        num_outputs = 1
        _, output = l2(num_outputs)
        theta = np.array([-1, 2, -3])
        actual = output(theta)
        expected = 4 + 9
        self.assertEqual(actual, expected)

    def test_1_gradient(self):
        num_outputs = 1
        gradient, _ = l2(num_outputs)
        theta = np.array([-1, 2, -3])
        actual = gradient(theta)
        expected = np.hstack((0, 4, -6))
        self.assertEqual(actual.shape, (3,))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)

    def test_3_output(self):
        num_outputs = 3
        _, output = l2(num_outputs)
        theta = np.array([-1, 2, -3, 4, -5, 6, -7, 8, -9])
        actual = output(theta)
        expected = 16 + 25 + 36 + 49 + 64 + 81
        self.assertEqual(actual, expected)

    def test_3_gradient(self):
        num_outputs = 3
        gradient, _ = l2(num_outputs)
        theta = np.array([-1, 2, -3, 4, -5, 6, -7, 8, -9])
        actual = gradient(theta)
        expected = np.array([0, 0, 0,
                             8, -10, 12, -14, 16, -18])
        self.assertEqual(actual.shape, (9,))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)

    def test_3_gradient_check(self):
        # check using check_gradient
        num_outputs = 10
        gradient, output = l2(num_outputs)

        def f(x):
            return output(x)

        def g(x):
            return gradient(x)

        num_tests = 100
        for _ in xrange(num_tests):
            x = next(random_sample.rand(10))
            stepsize = 0.00001
            tolerance = 0.001
            ok = check_gradient.roy(f, g, x, stepsize, tolerance)
            self.assertTrue(ok)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
