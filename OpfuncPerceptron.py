from Opfunc import Opfunc
import numpy as np
import pdb


class OpfuncPerceptron(Opfunc):
    '''Perceptron classifier.'''

    def __init__(self, phi):
        '''
        ARGS
        phi: function(x) -> transformed x
        '''
        Opfunc.__init__(self)
        self.phi = phi

    def opfunc(self, w, x, y):
        '''Return f(w), df_dw(x, y).

        ARGS
        w: np.array 1D of weights
        x: np.array 1D (for now, later, allow 2D)
        y: number (for now, later, allow 1D)

        NOTE: for classification, code y as +1 or -1
        '''
        phi_x = self.phi(x)
        y_times_phi_x = y * self._score(w, x)
        loss = max(0, - y_times_phi_x)
        grad = phi_x * y if y_times_phi_x <= 0 else np.zeros_like(phi_x)
        return loss, -grad

    def predict(self, w, x):
        '''Return +1 or -1'''
        return 1 if self._score(w, x) > 0 else -1

    def _score(self, w, x):
        return np.dot(w.T, self.phi(x))


if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):
        def setUp(self):
            self.verbose = False

        def test_opfunc(self):
            def phi(x):
                x0 = x[0]
                x1 = x[1]
                return np.array([x0, x1, x0 * x1])

            opfunc = OpfuncPerceptron(phi)
            w = np.array([1, 2, 3])
            x = np.array([4, 5])
            y = 1  # must be -1 or +1
            loss, grad = opfunc.opfunc(w, x, y)
            if self.verbose:
                print 'loss', loss
                print 'grad', grad
            loss_expected = 0
            grad_expected = np.array([0, 0, 0])
            self.assertAlmostEqual(loss, loss_expected)
            self.assertTrue(np.linalg.norm(grad - grad_expected) < .001)

            y = -1  # must be -1 or +1
            loss, grad = opfunc.opfunc(w, x, y)
            if self.verbose:
                print 'loss', loss
                print 'grad', grad
            loss_expected = 74
            grad_expected = - np.array([-4, -5, -20])
            self.assertAlmostEqual(loss, loss_expected)
            self.assertTrue(np.linalg.norm(grad - grad_expected) < .001)

        def test_predict(self):
            def phi(x):
                x0 = x[0]
                x1 = x[1]
                return np.array([x0, x1, x0 * x1])

            adaline = OpfuncPerceptron(phi)
            w = np.array([1, 2, 3])
            x = np.array([4, 5])

            y = adaline.predict(w, x)
            self.assertEqual(y, 1)

            w2 = np.array([0, 0, 0])
            y2 = adaline.predict(w2, x)
            self.assertEqual(y2, -1)

    unittest.main()
