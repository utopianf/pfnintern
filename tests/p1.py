from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np

from src.gnn import GNN, output_vector


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
# [66 60 80]
print(h)
