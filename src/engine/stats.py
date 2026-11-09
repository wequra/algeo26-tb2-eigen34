import math

def mean_face(X):
    d = len(X)
    if d == 0:
        raise ValueError("X is empty")
    N = len(X[0])
    if N == 0:
        raise ValueError("X has zero columns")
    for j in range(d):
        if len(X[j]) != N:
            raise ValueError(f"X is ragged at row {j}: {len(X[j])} != {N}")

    mu = [0.0] * d
    for i in range(N):
        for j in range(d):
            mu[j] += X[j][i]
    for j in range(d):
        mu[j] /= N
    return mu


def center(X, mu):
    d = len(X)
    N = len(X[0])
    if len(mu) != d:
        raise ValueError(f"dim mismatch: X has {d} rows, mu has {len(mu)}")
    A = []
    for j in range(d):
        m = mu[j]
        row = X[j]
        A.append([row[i] - m for i in range(N)])
    return A


def euclid(c_q, Z):
    d = len(c_q)
    N = len(Z[0]) if d else 0
    if len(Z) != d:
        raise ValueError(f"dim mismatch: c_q is {d}-D, Z has {len(Z)} rows")
    dists = [0.0] * N
    for i in range(N):
        s = 0.0
        for j in range(d):
            diff = Z[j][i] - c_q[j]
            s += diff * diff
        dists[i] = math.sqrt(s)
    return dists


def nearest(c_q, Z, labels=None):
    dists = euclid(c_q, Z)
    idx = 0
    best = dists[0]
    for i in range(1, len(dists)):
        if dists[i] < best:          
            best = dists[i]
            idx = i
    return idx if labels is None else labels[idx]