'''ordinary least squares with one prediction

FUNCTIONS returned by ols()
gradient(thta, x, y, [prediction])) -> vector
loss(thta, x, y, [prediction]))     -> number
predict(theta, x)                   -> number

ARGS
prediction: np array 1d
theta     : np array 1d
x         : np array 1d
y         : number
'''
import numpy as np
import pdb
import unittest
import model


def ols():
    '''ols is ridge regression with a zero weight for the regularizer'''
    return model.ridge(0)


class Test(unittest.TestCase):
    # tests are copied from ridge
    # with the modification that the regression weight is set to 0
    def setUp(self):
        self.theta = np.array([-1.0, 2, -3])
        self.x = np.array([4.0, -5])
        self.y = 6
        self.rw = 0.1  # regularizer weight

    def test_predict(self):
        gradient, loss, predict = ols()
        actual = predict(self.theta, self.x)
        self.assertTrue(isinstance(actual, float))
        expected = -1 + 2 * 4 - 3 * -5
        self.assertAlmostEqual(actual, expected)

    def test_loss(self):
        gradient, loss, predict = ols()
        prediction = predict(self.theta, self.x)
        actual = loss(self.theta, self.x, self.y, prediction)
        self.assertTrue(isinstance(actual, float))
        expected_error = 22 - 6
        expected_loss = expected_error * expected_error
        self.assertAlmostEqual(actual, expected_loss)

    def test_gradient(self):
        gradient, loss, predict = ols()
        actual = gradient(self.theta, self.x, self.y)
        expected_error = 22 - 6
        gradient_no_reg = np.array([1, 4, -5]) * 2 * expected_error
        expected = gradient_no_reg
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
