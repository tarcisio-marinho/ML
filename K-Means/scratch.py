import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

class KMeans:
    def __init__(self, k=2, tol=0.001, max_itr=300):
        self.k = k
        self.tol = tol
        self.max_itr = max_itr
    
    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_itr):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
            
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if(np.sum((current_centroid - original_centroid) / original_centroid*100.0) > self.tol):
                    optimized = False
            
            if(optimized):
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


if __name__ == "__main__":
    
    x = np.array([[1, 2],
                [1.5, 1.8],
                [8, 8],
                [1, 0.6],
                [9, 11],
                [5, 8]])

    colors = ["g", "r", "c", "b", "k"]

    clf = KMeans()
    clf.fit(x)
    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o"
                    , color="k", linewidths=5)

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    plt.show()
