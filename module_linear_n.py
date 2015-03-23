'''linear mode with 1 or more outputs'''
import numpy as np
import unittest
import pdb


def linear_n(num_inputs, num_outputs):
    assert num_inputs >= 1
    assert num_outputs >= 1

    def split(theta):
        '''bias vector and weight matrix'''
        b = theta[:num_outputs]
        w = theta[num_outputs:].reshape(num_outputs, num_inputs)
        return b, w

    def output(theta, x):
        '''vector size num_outputs'''
        assert x.ndim == 1  # however, code works for x.ndim == 2
        b, w = split(theta)
        r = (b + np.dot(x, w.T))
        return r

    def gradient(x):
        '''matrix size(num_outputs, num_inputs)'''
        # NOTE: only include arguments actually used
        # NOTE: always return 2d result
        r1 = np.tile(np.hstack((1, x)), num_outputs)
        return r1.reshape(num_outputs, num_inputs + 1)

    return gradient, output


class Test(unittest.TestCase):
    def test_1_output(self):
        num_inputs = 2
        num_outputs = 1
        _, output = linear_n(num_inputs, num_outputs)
        theta = np.array([-1, 2, -3])
        x = np.array([4, -5])
        actual = output(theta, x)
        expected = np.array([-1 + 2 * 4 - 3 * -5])
        self.assertEqual(actual.ndim, 1)
        self.assertEqual(actual.shape, (num_outputs,))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)

    def test_1_gradient(self):
        num_inputs = 2
        num_outputs = 1
        gradient, _ = linear_n(num_inputs, num_outputs)
        x = np.array([4, -5])
        actual = gradient(x)
        expected = np.hstack((1, x))
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape, (num_outputs, num_inputs + 1))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)

    def test_3_output(self):
        num_inputs = 2
        num_outputs = 3
        _, output = linear_n(num_inputs, num_outputs)
        theta = np.array([-1, 2, -3, 4, -5, 6, -7, 8, -9])
        x = np.array([4, -5])
        actual = output(theta, x)
        expected = np.array([-1 + 4 * 4 - 5 * -5,
                             2 + 6 * 4 - 7 * -5,
                             -3 + 8 * 4 - 9 * -5])
        self.assertEqual(actual.ndim, 1)
        self.assertEqual(actual.shape, (num_outputs,))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)

    def test_3_gradient(self):
        num_inputs = 2
        num_outputs = 3
        gradient, _ = linear_n(num_inputs, num_outputs)
        x = np.array([4, -5])
        actual = gradient(x)
        row = np.hstack((1, x))
        expected = np.array([row, row, row])
        self.assertEqual(actual.ndim, 2)
        self.assertEqual(actual.shape, (num_outputs, num_inputs + 1))
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
