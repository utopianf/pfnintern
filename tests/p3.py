import numpy as np
from src.gnn import GNN

B = 20
D = 8
a = 1e-4
m = 0.9

gnn1 = GNN()
gnn2 = GNN(W=gnn1.params['W'], A=gnn1.params['A'], b=gnn1.params['b'])

pre = list(range(2000))

for e in range(20):
    np.random.shuffle(pre)
    loss = 0
    for i in range(2000//B):
        g1 = {'W': np.zeros((D, D)),
              'A': np.zeros(D),
              'b': np.zeros(1)}
        g2 = {'W': np.zeros((D, D)),
              'A': np.zeros(D),
              'b': np.zeros(1)}
        w = {'W': np.zeros((D, D)),
             'A': np.zeros(D),
             'b': np.zeros(1)}
        for j in range(i*B, (i+1)*B):
            p = pre[j]
            X_A = np.loadtxt('datasets/train/{}_graph.txt'.format(p), skiprows=1)
            X = np.zeros((X_A.shape[0], D))
            X[:, 0] = 1
            t = np.loadtxt('datasets/train/{}_label.txt'.format(p))
            grads1 = gnn1.gradient(X, X_A, t[0])
            grads2 = gnn2.gradient(X, X_A, t[0])
            l = gnn1.loss(X, X_A, t[0])

            for key in ('W', 'A', 'b'):
                g1[key] = g1[key] + grads1[key]
                g2[key] = g2[key] + grads2[key]
            loss = loss + l

        for key in ('W', 'A', 'b'):
            g1[key] = g1[key] / B
            g2[key] = g2[key] / B
            gnn1.params[key] = gnn1.params[key] - a * g1[key]
            gnn2.params[key] = gnn2.params[key] - a * g2[key] + m * w[key]
            w[key] = m * w[key] - a * g2[key]
    print(loss/2000)
