'''module for squared loss, with adjustments for min and max expected values'''
import numpy as np
import unittest
import pdb


def squared_vw():
    '''return squared error squashed for expected values, as a scalar'''
    def output(predicted, expected, min_expected, max_expected):
        '''ARGS all scalars numbers'''
        # follow vw loss_functions squaredLoss
        if predicted <= max_expected and predicted >= min_expected:
            # predicted inside of expected range
            # use classic squared loss
            error = predicted - expected
            return error * error
        elif predicted < min_expected:
            if expected == min_expected:
                # pretend predicted and expected are the minimum expected
                return 0.0
            else:
                offset = expected - min_expected
                error = min_expected - predicted
                return 2.0 * offset * offset * offset * error
        else:
            if expected == max_expected:
                # pretend predicted and expected are the maximum expected
                return 0.0
            else:
                # predicted
                offset = max_expected - expected
                error = predicted - max_expected
                return 2.0 * offset * offset * offset * error
        return np.dot(error, error)

    def gradient(predicted, expected, min_expected, max_expected):
        '''gradient wrt predicted

        squash predicted into expected range
        '''
        if predicted < min_expected:
            # pretend predicted == min
            error = min_expected - expected
        elif predicted > max_expected:
            # pretend predicted == max
            error = max_expected - expected
        else:
            error = predicted - expected
        return np.array([2.0 * error])

    return gradient, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = squared_vw()
        # predicted in range
        self.assertAlmostEqual(output(12, 15, 10, 20), 9)  # within range
        # predicted < range
        self.assertAlmostEqual(output(9, 10, 10, 20), 0)  # expected at min
        #   offset = 2, error = 1
        self.assertAlmostEqual(output(9, 12, 10, 20), 2 * 2 * 2 * 2 * 1)
        # predicted > range
        self.assertAlmostEqual(output(23, 20, 20, 20), 0)  # expected at max
        #    offset = 5, error = 3
        self.assertAlmostEqual(output(23, 15, 20, 20), 2 * 5 * 5 * 5 * 3)

    def test_gradient(self):
        gradient, _ = squared_vw()

        def same(actual, expected1):
            expected = np.array([expected1])
            diff = np.linalg.norm(actual - expected)
            self.assertLess(diff, 1e-3)

        same(gradient(12, 15, 10, 20), 2.0 * -3)
        same(gradient(9, 15, 10, 20), 2.0 * -5)
        same(gradient(30, 15, 10, 20), 2.0 * 5)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
