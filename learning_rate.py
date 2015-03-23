'''collection of learning rates for SGD and Gradient Descent'''

import unittest
import pdb
import math


if False:
    pdb.set_trace()


def annealing(initial_learning_rate, learning_rate_decay, verbose=False):
    '''Return an iterator

    ARGS
    initial_learning_rate: number > 0, step size for first iteration
    learning_rate_decay  : number >= 0,
                           in (1/lrd) steps, the learning rate is halved
    verbose              : boolean

    RETURNS
    iterator: calling next(iterator) yields next learning rate

    ex:
        a = annealing(0.001, 100)
        lr1 = next(a)
        lr2 = next(b)

    ref: torch optim package sgd function
    '''
    assert initial_learning_rate > 0
    assert learning_rate_decay >= 0

    num_steps = 0
    while True:
        annealing = 1.0 + num_steps * learning_rate_decay
        learning_rate = initial_learning_rate / annealing
        if verbose:
            print 'num_steps', num_steps
            print 'annealing', annealing
            print 'learning_rate', learning_rate
        yield learning_rate
        num_steps += 1


class TestAnnealing(unittest.TestCase):
    def test_positive_decay(self):
        initial_learning_rate = 0.1
        learning_rate_decay = 1.0 / 3.0
        verbose = False
        learning_rate = annealing(initial_learning_rate,
                                  learning_rate_decay,
                                  verbose)
        result = []
        for i in xrange(4):
            result.append(next(learning_rate))
        self.assertAlmostEqual(result[0], initial_learning_rate)
        self.assertAlmostEqual(result[3], 0.5 * initial_learning_rate)

    def test_zero(self):
        initial_learning_rate = 0.1
        learning_rate_decay = 0
        verbose = False
        learning_rate = annealing(initial_learning_rate,
                                  learning_rate_decay,
                                  verbose)
        result = []
        for i in xrange(4):
            result.append(next(learning_rate))
        self.assertAlmostEqual(result[0], initial_learning_rate)
        self.assertAlmostEqual(result[3], initial_learning_rate)


def one_over_t(initial_learning_rate):
    '''Return an iterator yielding initial_learing_rate/t'''
    t = 0.0
    while True:
        t = t + 1.0
        yield initial_learning_rate / t


class TestOneOverT(unittest.TestCase):
    def test(self):
        lr = one_over_t(10)
        self.assertAlmostEqual(10, next(lr))
        self.assertAlmostEqual(5, next(lr))
        self.assertAlmostEqual(10 / 3.0, next(lr))


def one_over_sqrt_t(initial_learning_rate):
    '''Return an iterator yielding initial_learning_rate / sqrt(t)'''
    t = 0.0
    while True:
        t = t + 1.0
        yield initial_learning_rate / math.sqrt(t)


class TestOneOverSqrtT(unittest.TestCase):
    def test(self):
        lr = one_over_sqrt_t(10)
        self.assertAlmostEqual(10, next(lr))
        self.assertAlmostEqual(10 / math.sqrt(2), next(lr))
        self.assertAlmostEqual(10 / math.sqrt(3.0), next(lr))


if __name__ == '__main__':
    unittest.main()
