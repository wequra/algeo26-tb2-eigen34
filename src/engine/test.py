import numpy as np
from stats import *

if __name__ == "__main__":
    X = [[10.0, 30.0],    
        [20.0, 40.0]]      

    mu = mean_face(X)
    assert mu == [20.0, 30.0], mu
    print("mean_face:", mu)            

    A = center(X, mu)
    print(A)
    assert A == [[-10.0, 10.0], [-10.0, 10.0]], A
    print("center:", A)

    d = euclid([13.0, 24.0], X)
    assert abs(d[0] - 5.0) < 1e-12
    assert abs(d[1] - math.sqrt(545)) < 1e-12
    print("euclid: ", d)

    Z_tie = [[1.0, -1.0],
            [0.0,  0.0]]
    assert nearest([0.0, 0.0], Z_tie) == 0
    assert nearest([0.0, 0.0], Z_tie, labels=["a", "b"]) == "a"
    print("tie-break : 0 ")

    mu_A = mean_face(A)
    assert max(abs(v) for v in mu_A) < 1e-12
    print("mean centered:", mu_A)     
