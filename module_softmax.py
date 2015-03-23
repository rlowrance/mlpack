'''softmax

FUNCTIONS
output(values)  --> probs
gradient(probs) --> vector

ARGS
probs : np.array 1d size s
values: np.array 1d size s
vector: np.array 1d size s
'''
import numpy as np
import unittest
import pdb


def softmax():
    '''softmax_k = exp(input_k) / sum_k exp(input_k)'''
    def output(input):
        # guard against large input values
        max_value = np.max(input)
        e = np.exp(input - max_value)
        return e / np.sum(e)

    def gradient(output):
        grad = output * output - output
        return grad

    return gradient, output


class Test(unittest.TestCase):
    def test_gradient(self):
        gradient, output = softmax()
        input1 = np.array([-3, 1, 2])
        prob1 = output(input1)
        grad1 = gradient(prob1)
        expected = prob1 * prob1 - prob1
        diff = np.linalg.norm(grad1 - expected)
        self.assertLess(diff, 1e-3)

    def test_output(self):
        gradient, output = softmax()
        input1 = np.array([-3, 1, 2])
        prob1 = output(input1)
        self.assertLess(prob1[0], .01)
        self.assertLess(abs(prob1[1] - .27), .01)
        self.assertLess(abs(prob1[2] - .73), .01)

    def test_output_large(self):
        gradient, output = softmax()
        input0 = np.array([-3, 1, 2])
        large = np.array([1e200, 0, 0])
        input1 = input0 + large
        prob1 = output(input1)
        expected = np.array([1, 0, 0])
        diff = np.linalg.norm(prob1 - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
