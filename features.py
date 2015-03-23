'''Manipulate feature sets'''

import numpy as np
import unittest
import pdb


def rescale(train, test):
    """
    Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of
                size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    def rescale(m, mins, maxs):
        '''
        Rescale to [0,1].

        Args:
         m   : 2D array
         mins: 1D array of minum values in columns of m
         maxs: 1D array of maximum values in colums of m

        formula: x := (x - min(x))/(max(x) - min(x))
        '''
        numerator = m - mins
        denominator = maxs - mins
        result = numerator / denominator
        return result

    # compute training set statistics for each column
    mins = np.amin(train, axis=0)
    maxs = np.amax(train, axis=0)

    train_rescaled = rescale(train, mins, maxs)
    test_rescaled = rescale(test, mins, maxs)
    return train_rescaled, test_rescaled


class TestRescale(unittest.TestCase):

    def setUp(self):
        self.verbose = False

    def test_one_column(self):
        test = np.float64([[2],
                           [-10],
                           [100]])
        train = test
        train_rescaled, test_rescaled = \
            rescale(train, test)
        if self.verbose:
            print 'train_rescaled'
            print train_rescaled
        self.assertTrue((train_rescaled <= 1.0).all())
        self.assertTrue((train_rescaled >= 0.0).all())

    def test_rand(self):
        test = np.random.rand(2, 3) * 10
        train = np.random.rand(5, 3) * 10
        train_rescaled, test_rescaled = \
            rescale(train, test)
        if self.verbose:
            print 'train_rescaled'
            print train_rescaled
            print 'test_rescaled'
            print test_rescaled
        self.assertTrue((train_rescaled <= 1.0).all())
        self.assertTrue((train_rescaled >= 0.0).all())
        # NOTE: the test vectors satisfying no conditions at all

    def test_known(self):
        '''See lab book for 2/1 for notes.'''
        train = np.float64([[-1],
                            [0],
                            [1]])
        test = np.float64([[-4],
                           [6],
                           [0]])
        train_rescaled, test_rescaled = \
            rescale(train, test)
        if self.verbose:
            print 'train_rescaled'
            print train_rescaled
            print 'test_rescaled'
            print test_rescaled

        def check_equal(v, a, b, c):
            self.assertAlmostEqual(v[0], a)
            self.assertAlmostEqual(v[1], b)
            self.assertAlmostEqual(v[2], c)

        check_equal(train_rescaled, 0, .5, 1)
        check_equal(test_rescaled, -1.5, 3.5, .5)


def delete_constant_features(x):
    '''Retain only columns that have at least two values

    ARGS
    x: np.array 2d

    RETURNS
    non_constant: np.array 2d with possibly fewer columns

    ref: answer to hw1, DS1003, Spring 2015
    '''
    column_max = x.max(axis = 0)
    column_min = x.min(axis = 0)

    at_least_two_values = column_max != column_min
    non_constant = x[:, at_least_two_values]

    return non_constant


class TestDeleteConstantFeatures(unittest.TestCase):

    def test_delete_none(self):
        x = np.array([[1, 1, 2],
                      [10, 20, 30]]).T
        xx = delete_constant_features(x)
        self.assertTrue(np.linalg.norm(x - xx) <= 1e-6)


    def test_delete_one(self):
        x = np.array([[1, 1, 2],
                      [100, 100, 100],
                      [10, 20, 30]]).T
        xx = delete_constant_features(x)
        self.assertEqual(xx.shape[0], 3)
        self.assertEqual(xx.shape[1], 2)


if __name__ == '__main__':
    unittest.main()

if False:
    pdb.set_trace()
