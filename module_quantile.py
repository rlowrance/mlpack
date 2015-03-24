'''module for median loss'''
import unittest
import pdb


def quantile(tau):
    '''quantile error'''
    assert 0 < tau < 1

    def output(predicted, expected):
        '''return number'''
        # tau * (p - y) * I(y <= p) + (1- tau) * (y - p) * I(y >= p)
        negative_error = expected - predicted
        if negative_error > 0:
            return tau * negative_error
        else:
            return -(1 - tau) * negative_error

    def derivative(predicted, expected):
        '''derivate wrt predicted'''
        negative_error = expected - predicted
        if negative_error == 0:
            return 0
        elif negative_error > 0:
            return -tau
        else:
            return (1 - tau)

    return derivative, output


class Test(unittest.TestCase):
    def test_output(self):
        tau = .1
        _, output = quantile(tau)
        self.assertAlmostEqual(output(100, 10), 81)
        self.assertAlmostEqual(output(10, 100), 9)

    def test_derivative(self):
        tau = .1
        derivative, _ = quantile(tau)
        self.assertAlmostEqual(derivative(100, +1), .9)
        self.assertAlmostEqual(derivative(100, -1), .9)
        self.assertAlmostEqual(derivative(1, +1), 0)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
