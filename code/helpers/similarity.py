import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csc_matrix
from method.settings import Settings

class Similarity:
    def __init__(self, setup: Settings, data: np.ndarray):
        self.k = setup.k
        self.distance_metric = setup.metric.value
        self.data = data
    
    def construct_W(self):
        """
        Construct the affinity matrix W through different ways

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        Output
        ------
        W: {sparse matrix}, shape (n_samples, n_samples)
            output affinity matrix W
        """
        n_samples, _ = np.shape(self.data)

        # compute pairwise euclidean distances
        D = pairwise_distances(self.data, metric=self.distance_metric)
        D **= 2
        # sort the distance matrix D in ascending order
        idx = np.argsort(D, axis=1)
        # choose the k-nearest neighbors for each instance
        idx_new = idx[:, 0:self.k+1]
        G = np.zeros((n_samples*(self.k+1), 3))
        G[:, 0] = np.tile(np.arange(n_samples), (self.k+1, 1)).reshape(-1)
        G[:, 1] = np.ravel(idx_new, order='F')
        G[:, 2] = 1
        # build the sparse affinity matrix W
        W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
        bigger = np.transpose(W) > W
        W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
        return W
                
