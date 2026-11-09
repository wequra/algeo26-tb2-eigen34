import numpy as np
from stats import mean_face

def test():
    X = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float) 
    mu = mean_face(X)
    print(mu)

if __name__ == "__main__":
    test()