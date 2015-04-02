import math
import pdb
import unittest


def exp_range(low, high, num_steps):
    '''Return iterator for a*b^0 == low, a*b^1, ..., a*b^(num_steps -1) == high

    NOTE: low != 0
    '''
    a = low
    b = math.exp(math.log(high / low) / (num_steps - 1))
    n = 0
    for n in xrange(num_steps):
        yield a * math.pow(b, n)


class TestExprange(unittest.TestCase):
    def almostEqual(self, list_a, list_b):
        self.assertEquals(len(list_a), len(list_b))
        for index in xrange(len(list_a)):
            self.assertAlmostEqual(list_a[index], list_b[index])

    def test(self):
        actual = [x for x in exp_range(.001, 100, 6)]
        expected = [.001, .01, .1, 1, 10, 100]
        self.almostEqual(actual, expected)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
