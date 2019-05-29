from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np

from src.gnn import GNN


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
    print(loss[0])
print("Final Loss: {}".format(loss[0]))
