import unittest

import numpy as np

from src.gnn import GNN


class Test(unittest.TestCase):
    def test_p1_fixed_W(self):
        W = np.array([[2, 3, 1], [1, 3, 2], [0, 4, 5]])
        g = GNN(3, W_init=W)

        X = np.array([[1, 0, 0],
                      [1, 0, 0]])
        X_A = np.array([[0, 1],
                        [1, 0]])

        h = g.output_vector(X, X_A, T=3)
        self.assertEqual(h, np.array([66, 60, 80]))
