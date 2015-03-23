'''module for absolute value'''
import numpy as np
import unittest
import pdb


python_abs = abs


def abs():
    '''absolute error'''
    def output(x):
        return python_abs(x)

    def subgradient(x):
        '''subgradient wrt predicted'''
        if x >= 0:
            return np.array([+1])
        else:
            return np.array([-1])

    return subgradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = abs()
        self.assertEqual(output(+10), 10)
        self.assertEqual(output(0), 0)
        self.assertEqual(output(-10), 10)

    def test_subgradient(self):
        def equal(v1, v2):
            diff = np.linalg.norm(v1 - v2)
            self.assertLess(diff, 1e-3)

        subgradient, _ = abs()
        equal(subgradient(+10), np.array([1]))
        equal(subgradient(0), np.array([1]))
        equal(subgradient(-10), np.array([-1]))


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
