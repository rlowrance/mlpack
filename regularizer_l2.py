'''l2 regularizer

FUNCTIONS
loss(w)         -> number
gradient(theta) -> numpy array 1d

ARGS
theta: np array 1d, the entire set of parameters
w    : np array 1d, just the weights in theta (not the biases)
'''
import numpy as np
import unittest
import pdb


def l2():
    def gradient(w, num_biases):
        return np.hstack((np.zeros(num_biases), 2.0 * w))

    def loss(w):
        return np.sum(np.dot(w, w))

    return gradient, loss


class Test(unittest.TestCase):
    def setUp(self):
        self.w = np.array([1.0, 2, 3])

    def test_loss(self):
        _, loss = l2()
        self.assertAlmostEqual(loss(self.w), 14.0)

    def test_gradient(self):
        gradient, _ = l2()
        # 2 biases
        actual = gradient(self.w, 2)
        expected = np.array([0.0, 0, 2, 4, 6])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)
        # 0 biases
        actual = gradient(self.w, 0)
        expected = np.array([2.0, 4, 6])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
