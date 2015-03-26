'''module for squared loss'''
import unittest
import pdb


def squared():
    '''squared error'''
    def output(prediction, label):
        error = prediction - label
        return error * error

    def derivative(prediction, label):
        '''derivative wrt prediction'''
        return 2.0 * (prediction - label)

    return derivative, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = squared()
        self.assertAlmostEqual(output(10, 1), 81)
        self.assertAlmostEqual(output(1, 11), 100)

    def test_derivative(self):
        derivative, _ = squared()
        self.assertAlmostEqual(derivative(10, 1), 2 * 9)
        self.assertAlmostEqual(derivative(1, 11), 2 * -10)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
