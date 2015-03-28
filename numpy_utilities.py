'''utility functions that take numpy array's as arguments'''
import numpy as np
import unittest


def random_subset(matrix, n):
    '''pick a random subset without replacement of size n form the matrix'''
    selected = np.random.choice(matrix.shape[0], size=n, replace=False)
    return matrix[selected]


class TestRandomSubset(unittest.TestCase):
    def test(self):
        # ref: stackoverflow at numpty-get-random-set-of-rows-from-2d-array
        a = np.random.randint(5, size=(100, 3))
        s = random_subset(a, 10)
        self.assertEqual(s.shape[0], 10)
        self.assertEqual(s.shape[1], 3)


def almost_equal(a, b, tolerance, verbose=False):
    return almostEqual(a, b, tolerance, verbose)


def almostEqual(a, b, tolerance, verbose=False):
    '''Check if |a - b| < tolerance.'''
    diff = np.linalg.norm(a - b, 2)
    ok = diff < tolerance
    if verbose and not ok:
        print 'a', a
        print 'b', b
        print 'diff', diff
        print 'tolerance', tolerance
    return ok


class TestAlmostEqual(unittest.TestCase):
    def test(self):
        a = np.array([1, 1])
        b = np.array([2, 2])
        self.assertTrue(almostEqual(a, a, 1e-5))
        self.assertTrue(almostEqual(a, b, 1.5))
        self.assertFalse(almostEqual(a, b, .99))


def assertAlmostEqual(self, a, b):
    '''
    Assert that two numpy arrays have same shape and elements.

    Args
    self: unittest object
    a   : numpy array
    b   : numpy array
    '''
    self.assertEqual(a.ndim, b.ndim)

    ndim = a.ndim
    for d in range(ndim):
        self.assertEqual(a.shape[d], b.shape[d])

    a_flattened = a.flatten()
    b_flattened = b.flatten()
    for index in xrange(a_flattened.size):
        self.assertAlmostEqual(a_flattened[index], b_flattened[index])


class TestAssertAlmostEqual(unittest.TestCase):
    def setUp(self):
        self.a1d = np.array([1, 2, 3])
        self.b1d = np.array([4, 5, 6])
        self.a2d = np.array([[1, 2, 3],
                             [4, 5, 6]])
        self.b2d = np.array([[1, 2, 3],
                             [4, 5, 99]])

    def test_assertAlmostEqual_equal(self):
        assertAlmostEqual(self, self.a1d, self.a1d)
        assertAlmostEqual(self, self.a2d, self.a2d)


if __name__ == '__main__':
    unittest.main()
