'''l1 regularizer

FUNCTIONS
loss(w)         -> number
gradient(theta) -> np array 1d

ARGS
theta: np array 1d, the entire set of parameters
w    : np array 1d, just the weights in theta (not the biases)
'''
import numpy as np
import unittest
import pdb


def l1():
    def gradient(w, num_biases):
        # sign(x) in {-1, 0, +1}
        # we may want values in {-1, +1}
        # but computing sign(x) is probably faster
        return np.hstack((np.zeros(num_biases), np.sign(w)))

    def loss(w):
        return np.sum(np.abs(w))

    return gradient, loss


class Test(unittest.TestCase):
    def setUp(self):
        self.w = np.array([1.0, -2, 0])

    def test_loss(self):
        _, loss = l1()
        self.assertAlmostEqual(loss(self.w), 3)

    def test_gradient(self):
        gradient, _ = l1()
        # 2 biases
        actual = gradient(self.w, 2)
        expected = np.array([0.0, 0, 1, -1, 0])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)
        # 0 biases
        actual = gradient(self.w, 0)
        expected = np.array([1, -1, 0])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
