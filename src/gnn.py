import numpy as np


def np_log(x):
    """
    For overflow
    :param x:
    :return: log of x
    """
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))


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


class Graph:
    def __init__(self, A=None, X=None, D=None):
        """

        :param A: Adjacency matrix for the Graph.
        :type A: np.array
        :param X: A list of vertex of the Graph.
        :type X: np.array
        :param D:
        :type D: int
        """
        self.A = A
        self.X = X
        self.D = D

    def from_file(self, graph_file):
        self.A = np.loadtxt(graph_file, skiprows=1, dtype=int)
        self.X = np.zeros((len(self.A), self.D))
        self.X[:, 0:1] = 1


class GNN:
    def __init__(self, D, W_init=None, A=None, b=1.0, T=3):
        """

        :type D: int
        :param D: Input size.

        """
        self.D = D
        self.T = T
        self.params = {}
        if W_init is None:
            self.params['W'] = np.ones((D, D))
        else:
            self.params['W'] = W_init

        if A is None:
            self.params['A'] = np.ones(D)
        else:
            self.params['A'] = A

        self.params['b'] = b

        self.layer1 = Relu()
        self.layer2 = Sigmoid()

    def output_vector(self, X, X_A):
        """

        :param X: vertex list
        :type X: np.array
        :param X_A: Adjacency matrix
        :type X_A: np.array
        :return: output graph
        """
        for i in range(self.T):
            A = np.tensordot(X_A, X, axes=1)
            X = self.layer1.forward(self.W @ A.T).T

        return X.sum(axis=0)

    def loss(self, X, X_A, y):
        h_G = self.output_vector(X, X_A, self.T)
        s = self.A @ h_G.T + self.b
        p = self.layer2.forward(s)
        # y = 1 if p > 0.5 else 0

        return -1.0 * y * np_log(p) - (1.0 - y) * np_log(1.0 - p)

    def gradient(self, X, X_A, y, e=1e-6):
        loss_W = lambda W: self.loss(X, X_A, y)

        grads = {'W': numerical_gradient(loss_W, self.params['W'], e=e),
                 'b': numerical_gradient(loss_W, self.params['b'], e=e),
                 'A': numerical_gradient(loss_W, self.params['A'], e=e)}

        return grads

    def update(self, X, X_A, y, e=1e-6, a=1e-3):
        grads = self.gradient(X, X_A, y, e=e)

        self.params['W'] = self.params['W'] - a * grads['W']
        self.params['b'] = self.params['b'] - a * grads['b']
        self.params['A'] = self.params['A'] - a * grads['A']
