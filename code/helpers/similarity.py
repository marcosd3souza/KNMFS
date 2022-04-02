import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csc_matrix
from method.settings import Settings
from numpy import linalg as LA
import numpy.matlib

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
        # np.fill_diagonal(D, 1)
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
    
    @staticmethod
    def construct_spectral_info(S):
        n_samples = S.shape[0]
        # build the degree matrix
        X_sum = np.array(S.sum(axis=1))
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            D[i, i] = X_sum[i]

        # build the laplacian matrix
        L = D - S
        d1 = np.power(np.array(S.sum(axis=1)), -0.5)
        d1[np.isinf(d1)] = 0
        d2 = np.power(np.array(S.sum(axis=1)), 0.5)
        v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
        v = v/LA.norm(v)

        # build the normalized laplacian matrix
        L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)

        # calculate and construct spectral information
        vals, vecs = np.linalg.eig(L_hat)
        vals = np.flipud(vals)
        vecs = np.fliplr(vecs)
        
        return vals, vecs
                
