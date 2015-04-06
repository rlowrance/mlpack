'''regression test for linear regression using sgd_batch_auto for fitting

APPROACH: Replicate bishop-06 chapter 1 example of polynomial regression
'''
import math
import numpy as np
import pdb
import unittest


import dataset
import model
from sgd_batch_auto import sgd_batch_auto


def loss_all(theta, loss, xs, ys):
    pdb.set_trace()
    loss_total = 0.0
    num_samples = xs.shape[0]
    for i in xrange(num_samples):
        loss_total += loss(theta, xs[i], ys[i])
    return loss_total / num_samples


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

        # generate data
        def data():
            num_samples = 1000
            error_variance = .1
            return dataset.bishop_ch1(num_samples, error_variance)

        self.tx, self.train_y = data()
        self.vx, self.validate_y = data()

    def mean_loss(self, theta, loss):
        '''return mean loss on validation set'''
        total_loss = 0.0
        num_samples = self.validate_x.shape[0]
        for i in xrange(num_samples):
            total_loss += loss(theta, self.validate_x[i], self.validate_y[i])
        return total_loss / num_samples

    def phi(self, xs, order):
        '''return features, a polynomial of the given order'''
        verbose = False
        assert order >= 1
        num_samples = xs.shape[0]
        result = np.zeros((num_samples, order))
        for i in xrange(num_samples):
            for o in xrange(order):
                result[i][o] = math.pow(xs[i], o + 1)
            if verbose:
                print xs[i]
                print result[i]
        return result

    def fit(self, l1, l2, gradient, loss):
        theta0 = np.zeros(1 + self.train_x.shape[1])
        subset_size = self.train_x.shape[0]  # very few datapoints
        theta_star = sgd_batch_auto(regularizer_weight=l1 + l2,
                                    gradient=gradient,
                                    loss=loss,
                                    theta0=theta0,
                                    train_x=self.train_x,
                                    train_y=self.train_y,
                                    validate_x=self.validate_x,
                                    validate_y=self.validate_y,
                                    alpha0_subset_size=subset_size,
                                    verbose=True)
        return theta_star

    def find_thetas(self, l1, l2):
        '''return dict of optimal theta values for increasing num features'''
        verbose = True
        gradient, loss, predict = model.linear(l1, l2)

        result = {}
        num_features_low = 1
        num_features_low = 9
        num_features_high = 9
        for num_features in xrange(num_features_low, num_features_high + 1):
            # NOTE: not sure that model_linear works if no x values
            # transform generated random x values into features
            self.train_x = self.phi(self.tx, num_features)
            self.validate_x = self.phi(self.vx, num_features)

            theta_star = self.fit(l1, l2, gradient, loss)
            error = self.mean_loss(theta_star, loss)
            if verbose:
                print num_features, theta_star, error
            result[num_features] = (theta_star, error)

        print 'num_features, theta_star'
        for num_features in sorted(result):
            theta_star, error = result[num_features]
            print num_features, theta_star

        print 'num_features, error'
        for num_features in sorted(result):
            theta_star, error = result[num_features]
            print num_features, error
        pdb.set_trace()
        return result

    def test_no_regularizer(self):
        thetas = self.find_thetas(l1=0, l2=0)
        print 'test thetas', thetas

    @unittest.skip('')
    def test_l1(self):
        thetas = self.find_thetas(l1=.1, l2=0)
        print 'test thetas', thetas

    @unittest.skip('')
    def test_l2(self):
        thetas = self.find_thetas(l1=0, l2=.2)
        print 'test thetas', thetas

    @unittest.skip('')
    def test_l1_l2(self):
        thetas = self.find_thetas(l1=.1, l2=.2)
        print 'test thetas', thetas


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
