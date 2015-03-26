'''module for squared loss, with adjustments for min and max label values'''
import unittest
import pdb


def squared_vw():
    '''return squared error squashed for label values, as a scalar'''
    def output(prediction, label, min_label, max_label):
        '''ARGS all scalars numbers'''
        # follow vw loss_functions squaredLoss
        if prediction <= max_label and prediction >= min_label:
            # prediction inside of label range
            # use classic squared loss
            error = prediction - label
            return error * error
        elif prediction < min_label:
            if label == min_label:
                # pretend prediction and label are the minimum label
                return 0.0
            else:
                offset = label - min_label
                error = min_label - prediction
                return 2.0 * offset * offset * offset * error
        else:
            if label == max_label:
                # pretend prediction and label are the maximum label
                return 0.0
            else:
                # prediction
                offset = max_label - label
                error = prediction - max_label
                return 2.0 * offset * offset * offset * error
        return error * error

    def derivative(prediction, label, min_label, max_label):
        '''derivative wrt prediction

        squash prediction into label range
        '''
        if prediction < min_label:
            # pretend prediction == min
            error = min_label - label
        elif prediction > max_label:
            # pretend prediction == max
            error = max_label - label
        else:
            error = prediction - label
        return 2.0 * error

    return derivative, output


class Test(unittest.TestCase):
    def test_output(self):
        _, output = squared_vw()
        # prediction in range
        self.assertAlmostEqual(output(12, 15, 10, 20), 9)  # within range
        # prediction < range
        self.assertAlmostEqual(output(9, 10, 10, 20), 0)  # label at min
        #   offset = 2, error = 1
        self.assertAlmostEqual(output(9, 12, 10, 20), 2 * 2 * 2 * 2 * 1)
        # prediction > range
        self.assertAlmostEqual(output(23, 20, 20, 20), 0)  # label at max
        #    offset = 5, error = 3
        self.assertAlmostEqual(output(23, 15, 20, 20), 2 * 5 * 5 * 5 * 3)

    def test_derivative(self):
        derivative, _ = squared_vw()
        self.assertAlmostEqual(derivative(12, 15, 10, 20), 2.0 * -3)
        self.assertAlmostEqual(derivative(9, 15, 10, 20), 2.0 * -5)
        self.assertAlmostEqual(derivative(30, 15, 10, 20), 2.0 * 5)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
