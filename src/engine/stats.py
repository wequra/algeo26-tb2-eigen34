import numpy as np

def mean_face(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X harus berukuran d x N, tapi berbentuk {X.shape}")
    d, N = X.shape
    mu = np.zeros(d)
    for i in range(N):
        for j in range(d):
            mu[j] += X[j][i]
    
    return mu / float(N)
 