import unittest

import numpy as np

from src.gnn import GNN, output_vector


class Test(unittest.TestCase):
    def test_p1_fixed_W(self):
        W = np.array([[2, 3, 1], [1, 3, 2], [0, 4, 5]])
        A = np.ones(3)
        b = 1.0
        T = 3
        g = GNN(D=3, W=W, A=A, b=b, T=T)

        X = np.array([[1, 0, 0],
                      [1, 0, 0]])
        X_A = np.array([[0, 1],
                        [1, 0]])

        h = output_vector(g.update_graph(X, X_A))
        self.assertEqual(h, np.array([66, 60, 80]))

    def test_p2_decreasing_loss(self):
        X_A = np.array([[0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0]])

        X = np.zeros((10, 8))
        X[:, 0] = 1

        g = GNN()
        t = 0
        loss = g.loss(X, X_A, t)
        while abs(loss) > 1e-2:
            g.update(X, X_A, t)
            loss = g.loss(X, X_A, t)
        self.assertLessEqual(loss, 1e-2)
