'''regression test of model_ols'''
import model
import pdb
import sys
import numpy as np


import Logger
import Record
import random_sample


class Control(Record):
    def __init__(self, command_line_arguments):
        pdb.set_trace()
        Record.__init__(self, 'control')

        if len(command_line_arguments) > 1:
            raise RuntimeError('command line arguments ignored')

        me = 'model_ols_regression_test'

        self.path_out_log = '../data/working/' + me + '.txt'
        self.num_features = 1
        self.num_train_samples = 1000
        self.num_test_samples = 100
        self.true_theta = np.array([10, 1, -2])


def true_function(x, theta):
    return theta[0] + np.dot(x, theta[1:])


def generate_data(num_samples, control):
    '''return train_samples, train_labels'''
    train_samples = np.array(num_samples, control.num_features)
    train_labels = np.array(num_samples)
    for i in xrange(num_samples):
        x = next(random_sample.rand(control.num_features))
        y = true_function(x, control.true_theta)
        train_samples[i] = x
        train_labels[i] = y
    return train_samples, train_labels


def main():
    pdb.set_trace()
    control = Control(sys.argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control

    # train
    train_samples, train_labels = generate_data(control.num_train_samples,
                                                control)
    theta_star = fit_model(train_samples, train_labels)

    # test
    test_samples, test_labels = generate_data(control.num_test_samples,
                                              control)
    _, predict = model.ols()
    total_loss = 0.0
    for s in xrange(control.num_test_samples):
        prediction = predict(theta_star, test_samples[s])
        error = prediction - test_labels[s]
        total_loss += error * error
    mean_loss = total_loss / control.num_test_samples
    print 'mean_loss', mean_loss


if False:
    pdb.set_trace()
main()
