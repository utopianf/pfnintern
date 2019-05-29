from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

import copy

import numpy as np
from src.gnn import GNN, SGD, Momentum, Adam

B = 20
D = 8
a = 1e-4
m = 0.9

gnn1 = GNN()
gnn2 = copy.deepcopy(gnn1)
gnn3 = copy.deepcopy(gnn1)
opt1 = SGD()
opt2 = Momentum()
opt3 = Adam()

ps = list(range(2000))
X_As = []
ts = []

# datasets/train

for i in range(2000):
    X_As.append(np.loadtxt('datasets/train/{}_graph.txt'.format(i), skiprows=1))
    ts.append(int(np.loadtxt('datasets/train/{}_label.txt'.format(i))))

for e in range(20):
    np.random.shuffle(ps)
    test = ps[1800:]
    pre = ps[:1800]
    loss1 = 0
    loss2 = 0
    loss3 = 0
    for i in range(1800//B):
        g1 = {'W': np.zeros((D, D)),
              'A': np.zeros(D),
              'b': 0.0}
        g2 = {'W': np.zeros((D, D)),
              'A': np.zeros(D),
              'b': 0.0}
        g3 = {'W': np.zeros((D, D)),
              'A': np.zeros(D),
              'b': 0.0}
        w = {'W': np.zeros((D, D)),
             'A': np.zeros(D),
             'b': 0.0}
        for j in range(i*B, (i+1)*B):
            p = pre[j]
            X_A = X_As[p]
            X = np.zeros((X_A.shape[0], D))
            X[:, 0] = 1
            t = ts[p]
            grads1 = gnn1.gradient(X, X_A, t)
            grads2 = gnn2.gradient(X, X_A, t)
            grads3 = gnn3.gradient(X, X_A, t)
            l1 = gnn1.loss(X, X_A, t)
            l2 = gnn2.loss(X, X_A, t)
            l3 = gnn3.loss(X, X_A, t)

            for key in ('W', 'A', 'b'):
                g1[key] += grads1[key]
                g2[key] += grads2[key]
                g3[key] += grads3[key]
            loss1 += l1
            loss2 += l2
            loss3 += l3

        for key in ('W', 'A', 'b'):
            g1[key] /= B
            g2[key] /= B
            g3[key] /= B
            opt1.update(gnn1.params, g1)
            opt2.update(gnn2.params, g2)
            opt3.update(gnn3.params, g3)

    Y1, Y2, Y3 = 0, 0, 0
    for t in test:
        X_A = X_As[t]
        X = np.zeros((X_A.shape[0], D))
        X[:, 0] = 1
        t = ts[t]
        y1 = 1 if gnn1.predict(X, X_A) > 0.5 else 0
        y2 = 1 if gnn2.predict(X, X_A) > 0.5 else 0
        y3 = 1 if gnn3.predict(X, X_A) > 0.5 else 0
        Y1 += int(y1 == t)
        Y2 += int(y2 == t)
        Y3 += int(y3 == t)

    print("Loss: {0} {1} {2}\nAccuracy: {3} {4} {5}".format(loss1[0]/1800, loss2[0]/1800, loss3[0]/1800,
                                                            Y1/200.0, Y2/200.0, Y3/200.0))

# datasets/test
X_As = []
with open('prediction.txt', 'a') as f:
    for X_A in X_As:
        X = np.zeros((X_A.shape[0], D))
        X[:, 0] = 1
        y = 1 if gnn3.predict(X, X_A) > 0.5 else 0
        f.write("{}\n".format(str(y)))