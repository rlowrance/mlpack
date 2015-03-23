'''Loss functions

protocol
gradient_loss = f()

where
gradient_loss(theta, design, target, predict)->gradient and loss

where
gradient: np.array 1d
loss    : number
'''

import numpy as np
import pdb
import unittest
import numpy_utilities as npu
import random_sample
import check_gradient
import line_search
import predict

def squared(predict):
    '''loss = sum_i error_i^2'''
    def gradient_loss(theta, design, target):
        # optimize around use cases, as speed is important
        if design.ndim == 1:
            error = predict(theta, design) - target
            loss = error * error
            b, w = split(theta)
            grad_b = 2.0 * error
            grad_w = 2.0 * np.dot(error, error)
            grad = np.hstack((grad_b, grad_w))
            return grad, loss
        elif design.ndim == 2:
            num_samples = design.shape[0]
            error = predict(theta, design) - target
            loss = np.dot(error, error) / num_samples
            grad_w = 2.0 * np.sum(error)
            grad_b = 2.0 * np.dot(error, design) / num_samples
            grad = np.hstack((grad_b, grad_w))
            return grad, loss
        else:
            assert design.ndim ==1 or design.ndim == 2

    return gradient_loss


class TestSquared1D(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([-1, 1, -3])
        self.x = np.array([4, 2])
        self.y = 6

    def test_gradient_loss(self):
        gradient_loss = squared(predict.linear)
        self.assertTrue('write me')  # note: no regularizer




def _score_bw(b, w, x, transform):
    '''Return b and w^T phi(x) and phi(x)'''
    phi_x = transform(x)
    return b + np.dot(w.T, phi_x), phi_x


def _score(w, x, transform):
    '''Return w^T phi(x) and phi(x)'''
    phi_x = transform(x)
    return np.dot(w.T, phi_x), phi_x


def l1_norm(transform):
    '''l1_norm loss = | error |

    ARG
    transform: function(x)->np.array 1d

    RETURN 4 functions
    gradient(b, w, x, y)              -> np.array shape d, subgraident
    loss(b, w, x, y)                  -> number
    predict(b, w, x)                  -> number
    update(b, w, x, y, learning_rate) -> w*

    WHERE
    w: 1d np.array
    x: 1d np.array
    y: scalar, either +1 or -1

    Note: An alternative to the subgradient below is the Huber loss,
    which is continuous.
    '''
    def subgradient(b, w, x, y):
        '''Return a subgradient of |error|'''
        error = predict(b, w, x) - y
        return -1 if error < 0 else 1

    def loss(b, w, x, y):
        '''Return loss for sample (x,y) given weights w.'''
        return abs(predict(b, w, x) - y)

    def predict(b, w, x):
        '''Return class label y.'''
        score, _ = _score_bw(b, w, x, transform)
        return score

    def update(b, w, x, y, learning_rate):
        '''Return updated w.'''
        raise NotImplementedError

    return subgradient, loss, predict, update


class TestL1Norm(unittest.TestCase):
    def setUp(self):
        self.b = 1
        self.w = np.array([-2, 3, -4])
        self.x = np.array([5, -6])
        self.y = 7

        def transform(x):
            x1 = x[0]
            x2 = x[1]
            return np.array([x1, x2, x1 * x2])

        self.transform = transform

    def test_gradient_negative(self):
        def Identity(x):
            return x

        b = 0
        w = np.array([1])
        x = np.array([1])
        y = 3
        gradient, loss, predict, update = l1_norm(Identity)
        grad = gradient(b, w, x, y)
        self.assertTrue(npu.almost_equal(grad,
                                         np.array([-1]),
                                         1e-7))

    def test_gradient_positive(self):
        def Identity(x):
            return x

        b = 0
        w = np.array([1])
        x = np.array([1])
        y = -3
        gradient, loss, predict, update = l1_norm(Identity)
        grad = gradient(b, w, x, y)
        self.assertTrue(npu.almost_equal(grad,
                                         np.array([1]),
                                         1e-7))

    def test_gradient_zero(self):
        def Identity(x):
            return x

        b = 0
        w = np.array([1])
        x = np.array([1])
        y = 1
        gradient, loss, predict, update = l1_norm(Identity)
        grad = gradient(b, w, x, y)
        self.assertTrue(npu.almost_equal(grad,
                                         np.array([1]),
                                         1e-7))

    def test_loss(self):
        gradient, loss, predict, update = l1_norm(self.transform)
        self.assertAlmostEqual(loss(self.b, self.w, self.x, self.y),
                               86)
        self.assertAlmostEqual(loss(self.b, -self.w, self.x, self.y),
                               98)

    def test_predict(self):
        gradient, loss, predict, update = l1_norm(self.transform)
        self.assertAlmostEqual(predict(self.b, self.w, self.x),
                               93)
        self.assertAlmostEqual(predict(self.b, -self.w, self.x),
                               -91)


def adaline(transform):
    '''Return functions loss, predict, update.

    loss(w, x, y) -> number
    predict(w, x) -> +1 or -1
    update(w, x, y, learning_rate) -> w*

    ARGS
    w: 1d np.array
    x: 1d np.array
    y: scalar, either +1 or -1

    ref: bottou-12 sgd tricks
    '''
    def loss(w, x, y):
        '''Return loss for sample (x,y) given weights w.'''
        score, _ = _score(w, x, transform)
        error = y - score
        return 0.5 * error * error

    def predict(w, x):
        '''Return class label y.'''
        score, _ = _score(w, x, transform)
        return 1 if score > 0 else -1

    def update(w, x, y, learning_rate):
        '''Return updated w.'''
        score, phi_x = _score(w, x, transform)
        return w + learning_rate * (y - score) * phi_x

    return loss, predict, update


def kmeans(num_means):
    '''Return fuctions loss, predict, update.

    loss(w, z) -> number
    predict(w, x) -> k*, number of the cluster, |w| = k
    update(w, x) -> w*

    WHERE
    w: 2d np.array of shape num_means x d, centroids
    z: 2d np.array of shape t x d, data values
    x: 1d np.array of shape d
    y: integer?

    ref: bottou-12 sgd tricks
    '''
    n = np.zeros(num_means)  # counts are initially zero

    def loss(w, z):
        '''Return number, 1/2 the squared distance to nearest centroids.
        '''
        sum_loss = 0
        for row_index in xrange(z.shape[0]):
            x = z[row_index]
            k_star, loss = predict(w, x)
            sum_loss += loss
        return sum_loss

    def predict(w, x):
        '''Return k* and the loss for k*

        RETURNS
        k_star: integer, index of nearest centroid in w to x
        loss  : float, loss in using w[k] to estimate x
                euclidean distance
        '''
        k_star = None
        min_loss = float('inf')
        for k in xrange(num_means):
            distance = np.linalg.norm(x - w[k])  # 2-norm
            loss = 0.5 * distance * distance
            if loss < min_loss:
                min_loss = loss
                k_star = k
        return k_star, min_loss

    def update(w, x):
        k_star, _ = predict(w, x)
        n[k_star] += 1
        new_w = np.copy(w)
        new_w[k_star] += (x - w[k_star]) / n[k_star]
        return new_w

    return loss, predict, update


def lasso(regularizer_weight, transform):
    '''Return fuctions loss, predict, update
    ARGS
    regulizer_weight: number, importance of L2 regularizer
    transform(x) --> 1d np.array

    RETURNS
    loss(u, v, x, y) -> number
    predict(w, x) -> k*, number of the cluster, |w| = k
    update(w, x, y) -> w*

    WHERE
    u: 1d np. array of shape d, non-negative weights
    v: 1d np. array of shape d, non-negative weights
    x: 1d np.array of shape d
    y: integer, +1 or -1

    ref: bottou-12 sgd tricks
    '''
    def loss(u, v, x, y):
        w = u - v
        score, _ = _score(w, x, transform)
        error = y - score
        # NOTE: 1-norm is sum of absolute values
        return regularizer_weight * np.linalg.norm(w, 1) + 0.5 * error * error

    def predict(u, v, x):
        w = u - v
        score, _ = _score(w, x, transform)
        return 1 if score > 0 else 0  # ?

    def update(u, v, x, y, learning_rate):
        w = u - v
        score, phi_x = _score(w, x, transform)
        error = y - score
        u_new = plus(u - learning_rate * (regularizer_weight - error) * phi_x)
        v_new = plus(v - learning_rate * (regularizer_weight + error) * phi_x)
        return u_new, v_new

    def plus(v):
        '''replace negative values with zero'''
        return np.where(v < 0, 0, v)

    return loss, predict, update


def perceptron(transform):
    '''Return functions loss, predict, update.

    loss(w, x, y) -> number
    predict(w, x) -> +1 or -1
    update(w, x, y, learning_rate) -> w*

    ARGS
    w: 1d np.array
    x: 1d np.array
    y: scalar, either +1 or -1

    ref: bottou-12 sgd tricks
    '''
    def loss(w, x, y):
        '''Return loss for sample (x,y) given weights w.'''
        score, _ = _score(w, x, transform)
        return max(0, -y * score)

    def predict(w, x):
        '''Return class label y.'''
        score, _ = _score(w, x, transform)
        return 1 if score > 0 else -1

    def update(w, x, y, learning_rate):
        '''Return updated w.'''
        score, phi_x = _score(w, x, transform)
        if y * score <= 0:
            return w + learning_rate * y * phi_x
        else:
            return w

    return loss, predict, update


def ridge(regularizer_weight, verbose=False):
    '''Return functions gradient, loss, predict, theta_split, update

    theta = [b, w]
    loss = (1/n) (yhat - y) ^2 + lambda * w^T w
    '''
    def split(theta):
        '''Return b and w'''
        return theta[0], theta[1:]

    def loss_error(theta, design, target):
        '''Return avg. loss and error '''
        yhat = predict(theta, design)
        error = yhat - target
        b, w = split(theta)
        # assume num dimensions is 1 or 2
        num_samples = float(1 if design.ndim == 1 else design.shape[0])
        # use the fact that np.dot() for scalars is defined as multiplicaiton
        loss = \
            (np.dot(error, error) / num_samples) + \
            regularizer_weight * np.dot(w, w)
        return loss, error

    def loss(theta, design, target):
        loss, _ = loss_error(theta, design, target)
        return loss

    def gradient_loss(theta, design, target):
        '''Return avg. gradient and loss at theta'''
        if design.ndim == 1:
            # if design is 1d, then the target must be a
            # plain python number
            assert not isinstance(target, np.ndarray)
        loss, error = loss_error(theta, design, target)
        num_instances = float(1 if design.ndim == 1 else design.shape[0])
        b, w = split(theta)
        grad_b = 2.0 * np.sum(error)
        grad_unreg = 2.0 * np.dot(error, design) / num_instances
        reg = 2.0 * regularizer_weight * w  # NOTE: not averaged
        grad_reg = grad_unreg + reg
        grad = np.hstack((grad_b, grad_reg))
        return grad, loss

    def gradient(theta, design, target):
        gradient, _ = gradient_loss(theta, design, target)
        return gradient

    def predict(theta, design):
        b, w = split(theta)
        return b + np.dot(design, w)

    def update(theta, x, y, learning_rate):
        '''Return new theta and list of loss before the step'''
        grad, loss = gradient_loss(theta, x, y)
        if learning_rate == 'backtracking':
            pdb.set_trace()

            def my_loss(theta):
                return loss(theta, x, y)

            lr = line_search.backtracking(my_loss,
                                          grad,
                                          theta,
                                          -grad,
                                          10,
                                          .5,
                                          .5)
        else:
            lr = learning_rate

        theta_new = theta - lr * grad
        return theta_new, [loss]

    return gradient, gradient_loss, loss, predict, update


class TestRidgeMany(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([-1, 1, -3])
        self.z = np.array([[1, 2],
                           [3, 4]])
        self.t = np.array([5, 6])
        self.regularizer_weight = .1
        gradient, gradient_loss, loss, predict, update = \
            ridge(self.regularizer_weight)
        self.gradient = gradient
        self.loss = loss
        self.predict = predict
        self.update = update

    def test_predict(self):
        self.assertAlmostEqual(self.predict(self.theta,
                                            self.z[0]),
                               -6)
        self.assertAlmostEqual(self.predict(self.theta,
                                            self.z[1]),
                               -10)

    def test_loss(self):
        _, _, loss, _, _ = ridge(self.regularizer_weight)
        self.assertAlmostEqual(loss(self.theta, self.z, self.t),
                               (121 + 256) / 2.0 + 1.0)

    def test_gradient_completes(self):
        # just check for completion
        pdb.set_trace()
        self.gradient(self.theta,
                      self.z,
                      self.t)

    def test_gradient_steps(self):
        def loss(theta):
            return self.loss(theta, self.z, self.t)

        def p(theta):
            if False:
                print 'theta, loss', theta, loss(theta)

        theta0 = np.zeros(3)
        p(theta0)

        lr = .010  # learning rate

        theta1 = theta0 - lr * self.gradient(theta0, self.z, self.t)
        p(theta1)

        theta2 = theta1 - lr * self.gradient(theta1, self.z, self.t)
        p(theta2)

        self.assertLess(loss(theta2), loss(theta1))

    def test_check_gradient(self):
        gradient, _, loss, _, _ = ridge(0)  # regularizer 0
        d = self.z.shape[1]
        n = 100
        delta = 1e-5
        tolerance = 1e-2
        for _ in xrange(n):
            theta = next(random_sample.rand(d + 1))
            x1 = next(random_sample.rand(d))
            x2 = next(random_sample.rand(d))
            z = np.array([x1, x2])
            y1 = next(random_sample.rand(1))[0]
            y2 = next(random_sample.rand(1))[0]
            t = np.array([y1, y2])

            def myloss(theta, x, y):
                return loss(theta, z, t)

            def mygradient(theta, x, y):
                return gradient(theta, z, t)

            ok, peturbed_theta = check_gradient.bottou(theta, z, t,
                                                       loss, gradient,
                                                       delta, tolerance)
            if not ok:
                print 'bad theta', peturbed_theta
            self.assertTrue(ok)


class TestRidge1(unittest.TestCase):
    '''provide 1 training sample'''
    def setUp(self):
        self.theta = np.array([-1, 1, -3])
        self.x = np.array([4, 2])
        self.y = 6
        self.regularizer_weight = .1
        self.learning_rate = .5
        gradient, gradient_loss, loss, predict, update = \
            ridge(self.regularizer_weight)
        self.loss = loss
        self.predict = predict
        self.update = update
        self.gradient = gradient

    def test_loss(self):
        _, _, loss, _, _ = ridge(self.regularizer_weight)
        self.assertAlmostEqual(loss(self.theta, self.x, self.y),
                               81 + .1 * 10)

    def test_gradient(self):
        a = self.gradient(self.theta, self.x, self.y),
        prediction = -1 + 1 * 4 - 3 * 2
        error = prediction - 6
        grad_b = 2 * error
        grad_w0 = 2 * error * self.x[0] + 2 * self.regularizer_weight * 1
        grad_w1 = 2 * error * self.x[1] + 2 * self.regularizer_weight * -3
        b = np.array([grad_b, grad_w0, grad_w1])
        self.assertTrue(npu.almostEqual(a, b, 1e-3, False))

    def test_predict(self):
        self.assertAlmostEqual(self.predict(self.theta,
                                            self.x),
                               -3.0)

    def test_update(self):
        # for now, just check for completion
        theta_new, prev_loss = self.update(self.theta,
                                           self.x,
                                           self.y,
                                           self.learning_rate)
        if False:
            print 'test_update theta_new', theta_new

    def test_with_check_gradient(self):
        cases = 100
        d = 20
        for case in xrange(cases):
            theta = next(random_sample.rand(d + 1))  # allow for bias
            x = next(random_sample.rand(d))
            y = next(random_sample.rand(1))[0]  # need python number
            print self.loss(theta, x, y)
            print self.gradient(theta, x, y)
            delta = 1e-5
            tolerance = 1e-2

            def loss(theta, x, y):
                return self.loss(theta, x, y)

            def gradient(theta, x, y):
                return self.gradient(theta, x, y)

            check_gradient.bottou(theta, x, y,
                                  loss, gradient,
                                  delta, tolerance)
        self.assertTrue(True)


class RegressionTestSmallKmeans(unittest.TestCase):
    def setUp(self):
        self.z = np.array([[1, 1],
                           [1, 1.5],
                           [2, 0.5]])
        self.targets = np.array([-1, -1, 1])
        self.w = np.array([self.z[0], self.z[2]])
        self.num_means = 2
        loss, predict, update = kmeans(self.num_means)
        self.loss = loss
        self.predict = predict
        self.update = update

    def test_predict(self):
        k_star, loss = self.predict(self.w, self.z[0])
        self.assertEqual(k_star, 0)
        min_distance = 0
        self.assertAlmostEqual(loss, .5 * min_distance * min_distance)

        k_star, loss = self.predict(self.w, self.z[1])
        self.assertEqual(k_star, 0)
        min_distance = .5
        self.assertAlmostEqual(loss, .5 * min_distance * min_distance)

        k_star, loss = self.predict(self.w, self.z[2])
        self.assertEqual(k_star, 1)
        min_distance = 0
        self.assertAlmostEqual(loss, .5 * min_distance * min_distance)

    def test_loss(self):
        loss = self.loss(self.w, self.z)
        self.assertAlmostEqual(loss, 0.125)

    def test_update(self):
        def use(x):

            pass

        def step(old_w, index, tolerance):
            new_w = self.update(old_w, self.z[index])
            delta_w_norm = np.linalg.norm(new_w - old_w)
            if False:
                print 'new_w', new_w
                print 'delta_w_norm', delta_w_norm
            self.assertTrue(delta_w_norm < tolerance)
            return new_w

        w1 = step(self.w, 0, 1)
        w2 = step(w1, 1, 1)
        w3 = step(w2, 2, 1)
        use(w3)


def svm(regularizer_weight, transform):
    '''Return fuctions loss, predict, update.
    ARGS
    regulizer_weight: number, importance of L2 regularizer
    transform(x) --> 1d np.array

    RETURNS
    loss(w, x, y) -> number
    predict(w, x) -> k*, number of the cluster, |w| = k
    update(w, x, y) -> w*

    WHERE
    w: 1d np.array of shape d, weights
    x: 1d np.array of shape d
    y: integer, +1 or -1

    ref: bottou-12 sgd tricks
    '''
    def loss(w, x, y):
        score, _ = _score(w, x, transform)
        return regularizer_weight * np.dot(w, w) + max(0, 1 - y * score)

    def predict(w, x):
        score, _ = _score(w, x, transform)
        return 1 if score > 0 else 0  # ?

    def update(w, x, y, learning_rate):
        score, phi_x = _score(w, x, transform)
        new_w = w - learning_rate * regularizer_weight * w
        if y * score > 1:
            return new_w
        else:
            return new_w + learning_rate * y * phi_x

    return loss, predict, update


class UnitTestBottou(unittest.TestCase):
    '''Unit test of functions from Bottou

    These are
    adaline
    perceptron
    kmeans
    svm
    lasso
    '''
    def test_lasso(self):
        regularizer_weight = 0.1
        loss, predict, update = lasso(regularizer_weight=regularizer_weight,
                                      transform=self.transform)
        w, x, learning_rate = self.wxlr()
        d = self.transform(x).size
        u = np.zeros(d)
        v = np.zeros(d)

        y = +1
        yhat = predict(u, v, x)
        self.assertEqual(yhat, 0)
        self.assertAlmostEqual(loss(u, v, x, y), .5)
        u1, v1 = update(u, v, x, y, learning_rate)
        if False:
            print 'u1', u1
            print 'v1', v1
        self.np_almost_equal(u1, np.array([.36, .45, 1.8]))
        self.np_almost_equal(v1, np.array([0, 0, 0]))

        y = -1
        yhat = predict(u1, v1, w)
        u2, v2 = update(u1, v1, x, y, learning_rate)
        if False:
            print 'yhat', yhat
            print 'u2', u2
            print 'v2', v2
            print 'write more'

    def test_svm(self):
        regularizer_weight = 0.1
        loss, predict, update = svm(regularizer_weight=regularizer_weight,
                                    transform=self.transform)
        w, x, learning_rate = self.wxlr()

        y = +1
        yhat = predict(w, x)
        self.assertEqual(yhat, +1)
        expected = 1.4
        self.assertAlmostEqual(loss(w, x, y), expected)
        w1 = update(w, x, y, learning_rate)
        expected = w - learning_rate * regularizer_weight * w
        self.np_almost_equal(w1, expected)

        y = -1
        yhat = predict(w, x)
        self.assertEqual(yhat, +1)
        expected = 76.4
        self.assertAlmostEqual(loss(w, x, y), expected)
        w2 = update(w1, x, y, learning_rate)
        expected = np.array([0.5801, 1.4602, 0.9403])
        self.np_almost_equal(w2, expected)

    def np_almost_equal(self, a, b):
        norm_diff = np.linalg.norm(a - b)
        if False:
            print 'np_almost_equal'
            print 'a', a
            print 'b', b
            print 'norm diff', norm_diff
        self.assertTrue(np.abs(norm_diff) < 0.1)

    def setUp(self):
        def transform(x):
            x0 = x[0]
            x1 = x[1]
            return np.array([x0, x1, x0 * x1])

        def wxlr():
            w = np.array([1, 2, 3])
            x = np.array([4, 5])
            learning_rate = 0.1
            return w, x, learning_rate

        self.transform = transform
        self.wxlr = wxlr

    def test_adaline(self):
        loss, predict, update = adaline(self.transform)
        w, x, learning_rate = self.wxlr()

        y = +1
        error = -73
        self.assertAlmostEqual(loss(w, x, y), 0.5 * error * error)
        self.assertEqual(predict(w, x), +1)
        expected_update = w + learning_rate * error * np.array([4, 5, 20])
        self.np_almost_equal(update(w, x, y, learning_rate),
                             expected_update)

        y = -1
        error = -75
        self.assertAlmostEqual(loss(w, x, y), 0.5 * error * error)
        self.assertEqual(predict(w, x), +1)
        expected_update = w + learning_rate * error * np.array([4, 5, 20])
        self.np_almost_equal(update(w, x, y, learning_rate),
                             expected_update)

    def test_perceptron(self):
        loss, predict, update = perceptron(self.transform)
        w, x, learning_rate = self.wxlr()

        y = +1
        self.assertAlmostEqual(loss(w, x, y), 0)
        self.assertEqual(predict(w, x), +1)
        self.np_almost_equal(update(w, x, y, learning_rate),
                             w)

        y = -1
        self.assertAlmostEqual(loss(w, x, y), 74)
        self.assertEqual(predict(w, x), +1)
        expected_update = w + learning_rate * y * np.array([4, 5, 20])
        self.np_almost_equal(update(w, x, y, learning_rate),
                             expected_update)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
