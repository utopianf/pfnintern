import numpy as np


def np_log(x):
    """
    For overflow
    :param x:
    :return: log of x
    """
    return np.log(np.clip(a=x, a_min=1e-7, a_max=x))


def numerical_gradient(f, x, e=1e-6):
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        fx = f(x)

        x[idx] = np.float(tmp_val) + e
        fxh = f(x)

        grad[idx] = (fxh - fx) / e

        x[idx] = tmp_val
        it.iternext()

    return grad


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


def output_vector(X):
    """

    :param X: vertex list
    :type X: np.array
    :param X_A: Adjacency matrix
    :type X_A: np.array
    :return: output graph
    """
    return X.sum(axis=0)


class GNN(object):
    def __init__(self, D=8, W=None, A=None, b=None, T=2):
        """

        :type D: int
        :param D: Input size.

        """
        self.D = D
        self.T = T
        self.params = {}
        if W is None:
            self.params['W'] = 0.4 * np.random.randn(D, D)
        else:
            self.params['W'] = W

        if A is None:
            self.params['A'] = 0.4 * np.random.randn(D)
        else:
            self.params['A'] = A

        if b is None:
            self.params['b'] = np.zeros(1)
        else:
            self.params['b'] = b

        self.layer1 = Relu()
        self.layer2 = Sigmoid()

    def update_graph(self, X, X_A):
        for i in range(self.T):
            A = np.tensordot(X_A, X, axes=1)
            X = self.layer1.forward(self.params['W'] @ A.T).T

        return X

    def predict(self, X, X_A):
        """

        :param X: Vertex group
        :type X: np.ndarray
        :param X_A: Edge group
        :type X_A: np.ndarray
        :return: prediction
        """
        X = self.update_graph(X, X_A)

        h_G = output_vector(X)
        s = self.params['A'] @ h_G.T + self.params['b']
        p = self.layer2.forward(s)

        return p

    def loss(self, X, X_A, t):
        """

        :param X: Vertex group
        :type X: np.ndarray
        :param X_A: Edge group
        :type X_A: np.ndarray
        :param t: Label which is included in {0, 1}
        :type t: int
        :return: Loss
        """
        p = self.predict(X, X_A)
        # y = 1 if p > 0.5 else 0

        return -1.0 * t * np_log(p) - (1.0 - t) * np_log(1.0 - p)

    def gradient(self, X, X_A, t, e=1e-3):
        loss_W = lambda W: self.loss(X, X_A, t)

        grads = {'W': numerical_gradient(loss_W, self.params['W'], e=e),
                 'b': numerical_gradient(loss_W, self.params['b'], e=e),
                 'A': numerical_gradient(loss_W, self.params['A'], e=e)}

        return grads

    def update(self, X, X_A, t, e=1e-3, a=1e-4):
        grads = self.gradient(X, X_A, t, e=e)

        self.params['W'] = self.params['W'] - a * grads['W']
        self.params['b'] = self.params['b'] - a * grads['b']
        self.params['A'] = self.params['A'] - a * grads['A']
