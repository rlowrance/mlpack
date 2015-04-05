'''multiclass classification model

TODO: revise to split gradient_loss into gradient() and loss()

FUNCTIONS
gradient_loss()
predict()

ARGS
num_inputs : number
num_classes: number
'''
import numpy as np
import unittest
import pdb
import module


def multiclass(num_inputs, num_classes):
    '''multinomial regression without a regularizer'''
    # MAYBE: rework using module_linear instead of module_linear_n
    # Idea: avoid need to introduce Jacobian derivative
    assert num_classes > 0
    linear_grad, linear_output = module.linear_n(num_inputs, num_classes)
    softmax_grad, softmax_output = module.softmax()
    nl_grad, nl_output = module.negative_likelihood()

    def predict(theta, x):
        scores = linear_output(theta, x)
        probs = softmax_output(scores)
        return probs

    def gradient_loss(theta, x, y, probs):
        raise RuntimeError('unittest not run')
        verbose = False
        loss = nl_output(probs, y)
        g_linear = linear_grad(x)
        g_softmax = softmax_grad(probs)
        g_nl = nl_grad(probs, y)
        if verbose:
            print 'g_linear', g_linear
            print 'g_softmax', g_softmax
            print 'g_nl', g_nl
        g = g_linear * g_softmax * g_nl
        return g, loss

    return gradient_loss, predict


class Test(unittest.TestCase):
    def test_predict(self):
        num_inputs = 2
        num_classes = 3
        gradient_loss, predict = multiclass(num_inputs, num_classes)
        theta = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = np.array([-10, 11])
        actual = predict(theta, x)
        expected = np.array([.002, .047, .950])
        diff = np.linalg.norm(actual - expected)
        self.assertLess(diff, 1e-2)

    def test_gradient_loss(self):
        return
        num_inputs = 2
        num_classes = 3
        gradient_loss, predict = multiclass(num_inputs, num_classes)
        theta = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = np.array([-10, 11])
        y = 1  # class label
        probs = predict(theta, x)
        gradient, loss = gradient_loss(theta, x, y, probs)
        if False:
            print 'gradient', gradient
            print 'loss', loss
            print 'write tests'


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
