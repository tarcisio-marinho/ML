import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

class KMeans:
    def __init__(self, k=2, tol=0.001, max_itr=300):
        self.k = k
        self.tol = tol
        self.max_itr = max_iter
    
    def fit(self, data):
        pass

    def predict(self, data):
        pass


if __name__ == "__main__":
    style.use("ggplot")

    x = np.array([[1, 2],
                [1.5, 1.8],
                [8, 8],
                [1, 0.6],
                [9, 11]])

    plt.scatter(x[:,0], x[:,-1], s=150)
    plt.show()

