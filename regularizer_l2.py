'''l2 regularizer

FUNCTIONS
loss(w)           -> number
derivative(theta) -> number

ARGS
theta: np array 1d, the entire set of parameters
w    : np array 1d, just the weights in theta (not the biases)
'''
import numpy as np
import unittest
import pdb


def l2OLD(num_outputs):
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


def l2():
    def loss(w):
        return np.sum(np.dot(w, w))

    def derivative(w, num_biases):
        return np.hstack((np.zeros(num_biases), 2.0 * w))

    return derivative, loss


class Test(unittest.TestCase):
    def setUp(self):
        self.w = np.array([1.0, 2, 3])

    def test_loss(self):
        _, loss = l2()
        self.assertAlmostEqual(loss(self.w), 14.0)

    def test_derivative(self):
        derivative, _ = l2()
        # 2 biases
        actual = derivative(self.w, 2)
        expected = np.array([0.0, 0, 2, 4, 6])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)
        # 0 biases
        actual = derivative(self.w, 0)
        expected = np.array([2.0, 4, 6])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
