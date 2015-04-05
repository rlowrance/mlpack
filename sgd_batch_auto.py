'''auto-tuning batch sgd

ARGS
regularizer_weight: number
theta0            : vector
train_x           : matrix
train_y           : vector
validate_x        : matrix
validate_y        : vector
alpha0_subset_set : number, num samples from train_x used to select initial
                    learning rate

RETURN
theta_star        : vector, optimal
'''
import math
import numpy as np
import unittest
import pdb
import warnings

import check_gradient
import dataset
import minimize_1d
import model
import python_utilities as pu
import random_sample


def argmin(iterable):
    min_value = float('inf')
    min_index = None
    for i in xrange(len(iterable)):
        value = iterable[i]
        if value < min_value:
            min_value = value
            min_index = i
    return min_index


def learning_rate(initial_learning_rate, regularizer_weight, t):
    return \
        initial_learning_rate / \
        (1.0 + regularizer_weight * initial_learning_rate * t)


def always_decreasing(lst):
    # return True iff only decreasing in last portion of list
    num_to_consider = 10
    t = len(lst) - (num_to_consider + 1)
    for i in xrange(num_to_consider):
        if lst[t + i] > lst[t + i + 1]:
            pass
        else:
            return False
    return True


def epoch(alpha0, theta0, xs, ys, gradient, t,
          next_alpha=None, check_gradient=None, loss=None):
    '''return final theta and updated t at end of epoch'''
    verbose = False
    num_samples = xs.shape[0]
    theta = theta0
    for i in xrange(num_samples):
        alpha = alpha0 if next_alpha is None else next_alpha(alpha0, t)
        if verbose:
            print 'old theta', theta
            print 'alpha', alpha
            print 'g', gradient(theta, xs[i], ys[i])
        theta = theta - alpha * gradient(theta, xs[i], ys[i])
        if verbose:
            print 'new theta', theta
        t += 1
        if check_gradient is not None:
            check_gradient(theta, xs[i], ys[i],
                           loss, gradient,
                           1e-5, 1e03)
    return theta, t


def loss_mean(theta, xs, ys, loss):
    '''return mean loss across the sample from using theta'''
    loss_total = 0.0
    num_samples = xs.shape[0]
    for i in xrange(num_samples):
        loss_total += loss(theta, xs[i], ys[i])
    return loss_total / num_samples


def set_initial_learning_rate(search_method,
                              gradient, loss, regularizer_weight,
                              theta0,
                              xs, ys):
    '''return scalar alpha based on specified training samples'''
    verbose = False
    check_gradient_flag = False
    num_samples = xs.shape[0]

    def epoch(alpha, theta0):
        theta = theta0
        for i in xrange(num_samples):
            if check_gradient_flag:
                delta = 1e-5
                tolerance = 1e-3
                ok, theta_bad = check_gradient.bottou(theta, xs[i], ys[i],
                                                      loss, gradient,
                                                      delta, tolerance)
                if not ok:
                    print 'theta_bad', theta_bad
                    assert ok
            theta = theta - alpha * gradient(theta, xs[i], ys[i])
        return theta

    def loss_with_alpha(alpha):
        '''Return mean loss using alpha for one epoch'''
        theta = epoch(alpha, theta0)
        return loss_mean(theta, xs, ys, loss)

    def safe_loss_with_alpha(alpha):
        # handle warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                result = loss_with_alpha(alpha)
            except Warning:
                result = float('inf')
        return result

    if search_method == 'golden section':
        low_alpha, high_alpha, d = \
            minimize_1d.golden_section(safe_loss_with_alpha,
                                       0.0,
                                       100,
                                       0.0001,
                                       verbose=False)
        if verbose:
            print 'low_alpha', low_alpha
            print 'high_alpha', high_alpha
            print 'number function evaluations', len(d)
            print 'function evaluations'
            for k in sorted(d):
                print k, d[k]
        best_alpha = (low_alpha + high_alpha) / 2.0
    elif search_method == 'exponential range':
        scores = []
        for alpha in pu.exp_range(.01, 1, 101):
            score = safe_loss_with_alpha(alpha)
            print 'alpha', alpha, 'score', score
            scores.append(score)
            # print 'scores', scores
    elif search_method == 'range':
        scores = []
        for i in xrange(1, 100):
            alpha = i / 100.0
            score = safe_loss_with_alpha(alpha)
            print 'alpha', alpha, 'score', score
            scores.append(score)
            # print 'scores', scores
    else:
        raise RuntimeError('bad search_method: ' + str(search_method))

    # follow bottou-12 by slightly reducing the best initial learning rate
    # "for robustness"
    used_alpha = .9 * best_alpha
    if verbose:
        print 'initial learning rate', used_alpha
        theta = epoch(used_alpha, theta0)
        print 'final theta', theta

    return used_alpha


def no_change_in_long_time(vl_list, tolerance):
    '''Return True iff no change in the validation loss for a long time'''
    # print 'validation lost list', vl_list
    l = len(vl_list)
    if l < 10:
        return False
    example = vl_list[l - 9]
    diff = abs(example - vl_list[-1])
    # print 'diff', diff
    return diff < tolerance


def sgd_batch_auto(regularizer_weight, theta0,
                   train_x, train_y, validate_x, validate_y,
                   alpha0_subset_size,
                   verbose=False):
    '''autofit sgd for ridge regression

    IDEA 1
    Set the initial learning rate to be optimal in a sample of the training
    data.

    IDEA 2
    Repeat:
        Perform ASGD on training data.
        Determine current loss on training and validation data.
        Stop when the loss on the validation data has not decreased for a
        long time.
        Decide that something is wrong if the loss on the training set is
        not decreasing.

    IDEA 3
    Use bottou-12's idea for decreasing the learning rate. These ideas
    are specific to ridge regression, where the learning rate is
    alpha_t = alpha_0 * (1 + alpha_0 * learning_rate * t)^-1

    NOTE: We don't implement AGSD.
    NOTE: The features in train_x, validate_x should be normlized, as we
          don't rescale them.
    '''
    gradient, loss, predict = model.ridge(regularizer_weight)
    num_samples = train_x.shape[0]

    # select random sample of train data on which to set initial learning rate
    subset_indices = np.random.choice(alpha0_subset_size,
                                      size=alpha0_subset_size,
                                      replace=False)

    # find the best initial alpha (learning rate), using the subset
    alpha0_search_method = 'golden section'
    alpha0 = \
        set_initial_learning_rate(alpha0_search_method,
                                  gradient, loss, regularizer_weight,
                                  theta0,
                                  train_x[subset_indices],
                                  train_y[subset_indices])
    if verbose:
        print 'alpha0', alpha0

    # train on epochs
    def next_alpha(alpha0, t):
        # NOTE: Bottou suggests a different annealing schedule if ASGD
        result = alpha0 / (1 + alpha0 * regularizer_weight * t)
        # print 'next_alpha', alpha0, t, regularizer_weight, result
        return result

    def epoch(alph0, theta, t):
        for i in xrange(num_samples):
            t += 1
            alpha = next_alpha(alpha0, t)
            theta = theta - alpha * gradient(theta, train_x[i], train_y[i])
        return theta, t

    theta = theta0
    all_training_losses = []
    all_validation_losses = []
    tolerance = .01
    t = 0
    epoch_num = 0
    if verbose:
        print 'epoch,loss train,loss validate, theta'
    while True:
        epoch_num += 1
        theta, t = epoch(alpha0, theta, t)
        tl = loss_mean(theta, train_x, train_y, loss)
        vl = loss_mean(theta, validate_x, validate_y, loss)
        if verbose:
            print epoch_num, tl, vl, theta
        all_training_losses.append(tl)
        all_validation_losses.append(vl)
        # assert is_decreasing(all_training_losses)
        if no_change_in_long_time(all_validation_losses, tolerance):
            break
    if verbose:
        print 'final theta', theta
    return theta


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.debugging = True

    @unittest.skip('')
    def test_bishop_ch1(self):
        # test using dataset from Bishop-06 Chapter 1
        num_train_samples = 10000
        num_validate_samples = 100
        train_x, train_y = dataset.bishop(num_train_samples)
        validate_x, validate_y = dataset.bishop(num_validate_samples)

    def generate_data(self,
                      num_train, num_validate, num_dimensions, error_variance):
        def make_xsys(num_samples, num_dimensions, theta, predict):
            xs = np.zeros((num_samples, num_dimensions))
            ys = np.zeros(num_samples)
            for s in xrange(num_samples):
                # populate with drawns from uniform [-1,1]
                sample = next(random_sample.uniform(num_dimensions, -1, 1))
                xs[s] = sample
                error = next(random_sample.randn(1, 0, error_variance))[0]
                # print 'index', s, 'error', error
                ys[s] = predict(theta, sample) + error
            return xs, ys

        _, _, predict = model.ridge(regularizer_weight=0)
        theta = np.zeros(num_dimensions + 1)
        abs_value = 1
        sign = 1
        for d in xrange(num_dimensions + 1):
            theta[d] = abs_value * sign
            abs_value += 1
            sign *= -1

        train_x, train_y = make_xsys(num_train, num_dimensions,
                                     theta, predict)
        validate_x, validate_y = make_xsys(num_validate, num_dimensions,
                                           theta, predict)

        return theta, train_x, train_y, validate_x, validate_y, predict

    def bayes_rmse(self, theta, xs, ys, predict):
        # lowest achievable error (which is from using the true theta
        # and the training data xs, ys)
        num_samples = xs.shape[0]
        total_squared_error = 0.0
        for i in xrange(num_samples):
            error = predict(theta, xs[i]) - ys[i]
            total_squared_error += error * error
        return math.sqrt(total_squared_error / num_samples)

    def test_1(self):
        # for now test only running to completion
        verbose = False
        num_train = 2000
        num_validate = 100
        num_dimensions = 3
        error_variance = 1e-2
        theta_actual, train_x, train_y, validate_x, validate_y, predict = \
            self.generate_data(num_train,
                               num_validate,
                               num_dimensions,
                               error_variance)

        if verbose:
            print 'theta actual', theta_actual
            print 'error_variance', error_variance
            print 'bayes RMSE', self.bayes_rmse(theta_actual,
                                                train_x, train_y,
                                                predict)
        regularizer_weight = 0.001
        # regularizer_weight = 0.0

        theta0 = np.zeros(num_dimensions + 1)
        theta_star = sgd_batch_auto(regularizer_weight, theta0,
                                    train_x, train_y, validate_x, validate_y,
                                    int(num_train * .1))
        if verbose:
            print 'theta_star', theta_star
        diff = np.linalg.norm(theta_actual - theta_star)
        self.assertLess(diff, 2e-2)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
