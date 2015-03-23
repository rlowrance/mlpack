import numpy as np
import unittest
import pdb


def rand(d):
    '''Generate random np.array sample of size d.

    Each element is drawn uniformly from [0,1].
    '''
    while True:
        result = np.random.rand(d)
        yield result


class Test(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def test_generate(self):
        for i in xrange(10):
            x = next(rand(3))
            if self.verbose:
                print i, x

    def test_iterate(self):
        count = 0
        for value in rand(5):
            if self.verbose:
                print value
            count += 1
            if count > 5:
                break


if __name__ == '__main__':
    unittest.main()
