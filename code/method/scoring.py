from scipy.stats import rankdata
from typing import List, Any
import numpy as np
from enum import Enum
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
from scipy.sparse import *
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from method.settings import Settings, DistanceMetric, ScoringMethod
from helpers.similarity import Similarity

import numpy.matlib
from scipy.sparse import *

class Scoring:
    def __init__(self, communities: List[Any], data: np.ndarray, settings: Settings):
        self.communities = communities
        self.data = data
        self.settings = settings

    def get_variance_scores(self, nodes):
        objs = self.data.shape[0]

        off = [f for f in range(objs) if f not in nodes]
        com = self.data[nodes]
        off = self.data[off]

        d_com = np.var(com, axis=0)
        d_off = np.var(off, axis=0)

        d_all = np.var(self.data, axis=0)
        
        scores = d_all / ((d_off + 0.0000001) / (d_com + 9999999))

        return scores
    
    def get_ordinal_scores(self, community: np.ndarray):
        means = np.mean(community, axis=0)
        stds = np.std(community, axis=0)
        
        # means = [1 if abs(m) == 0 else abs(m) for m in means]
        # stds = [1 if abs(s) == 0 else abs(s) for s in stds]
        
        # S = [1 / (s / m) for s,m in zip(stds , means)]
        
        # return rankdata(S)

        return 1 / (stds / (means ** 2))
    
    def get_norm2_all_fea(self, distance_metric: str, W=None):
        # construct new dataset based on norm and the communities
        objs, n_features = self.data.shape
        I = np.ones(n_features)
        
        new_X = []
        scores = []
        for _, c in enumerate(self.communities):
            nodes = list(c)
            data_in = self.data[nodes]

            off = [f for f in range(objs) if f not in nodes]
            data_off = self.data[off]

            # prototype_index = np.argmax(W[nodes].sum(axis=1))
            # prototype = data_in[prototype_index]
            prototype = np.median(data_in, axis=0)
            dist_in = np.linalg.norm(data_in - prototype, axis=0) + 0.000001
        
            deviation = np.var(data_in, axis=0) + 0.00001
            dist_out = np.linalg.norm(data_off - prototype, axis=0) 

            c_scores = dist_out / dist_in # - deviation
            
            scores.append(c_scores)

        return np.mean(scores, axis=0)
    
    def calc_score(self, com_data: np.array, I: np.array):
        merged = np.concatenate([com_data, I], axis=1)
        # dist = distance.cdist(com_data, metric=self.settings.scoring_metric.value)
        dist = pairwise_distances(merged, metric=self.settings.scoring_metric.value)
        dist **= 2
        # dist = dist + 0.00001
        # dist = np.log(dist)
        # return 1/(np.linalg.norm(dist) + 0.00001)
        return 1/(np.sum(dist) + 0.00001)

    def get_norm2_scores(self, nodes: List, distance_metric: str):
        # objs = self.data.shape[0]

        # off = [f for f in range(objs) if f not in nodes]
        # off_community = self.data[off]
        # scaled_off_data = StandardScaler().fit_transform(off_community)
        # data_off_community = np.transpose(scaled_off_data)
        
        com_data = self.data[nodes]
        n_samples, n_features = com_data.shape
        I = np.ones(n_samples)[:, None]

        scores_1 = [self.calc_score(com_data[:, f][:, None], I) for f in range(n_features)]
        # scaled_data_community = StandardScaler().fit_transform(com_data) 
        data_community = np.transpose(com_data)

        # scaled_data = Normalizer(norm='l2').fit_transform(data_community)

        # D_off = pairwise_distances(data_off_community, metric=distance_metric)
        D_com = pairwise_distances(data_community, metric=distance_metric)
        D_com **= 2
        # D_com_scaled = MinMaxScaler().fit_transform(D_com)

        D_com = 1 / (D_com + 0.00001)

        scores_2 = np.linalg.norm(D_com, axis=0)
        
        # devs = (np.std(data_community, axis=1) + 0.00001)
        # means = (np.mean(data_community, axis=1) + 0.00001)
        # dev **= 2
        # scores = dist_norm / (devs/means)
        # D_off_dist = np.linalg.norm(D_off, axis=0)

        # scores = np.multiply(D_off_dist, (1/D_com_dist))
        return np.multiply(scores_1, 0.5) + np.multiply(scores_2, 0.5)

    def get_spec_scores(self, X, W):
        """
        This function implements the SPEC feature selection

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        kwargs: {dictionary}
            style: {int}
                style == -1, the first feature ranking function, use all eigenvalues
                style == 0, the second feature ranking function, use all except the 1st eigenvalue
                style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
            W: {sparse matrix}, shape (n_samples, n_samples}
                input affinity matrix

        Output
        ------
        w_fea: {numpy array}, shape (n_features,)
            SPEC feature score for each feature

        Reference
        ---------
        Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
        """

        # if 'style' not in kwargs:
        #     kwargs['style'] = 0
        # if 'W' not in kwargs:
        #     kwargs['W'] = rbf_kernel(X, gamma=1)

        style = -1 #kwargs['style']
        # W = kwargs['W']
        # if type(W) is numpy.ndarray:
        #     W = csc_matrix(W)

        n_samples, n_features = X.shape

        # build the degree matrix
        X_sum = np.array(W.sum(axis=1))
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            D[i, i] = X_sum[i]

        # build the laplacian matrix
        L = D - W
        d1 = np.power(np.array(W.sum(axis=1)), -0.5)
        d1[np.isinf(d1)] = 0
        d2 = np.power(np.array(W.sum(axis=1)), 0.5)
        v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
        v = v/np.linalg.norm(v)

        # build the normalized laplacian matrix
        L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)

        # calculate and construct spectral information
        s, U = np.linalg.eigh(L_hat)
        s = np.flipud(s)
        U = np.fliplr(U)

        # begin to select features
        w_fea = np.ones(n_features)*1000

        for i in range(n_features):
            f = X[:, i]
            F_hat = np.dot(np.diag(d2[:, 0]), f)
            l = np.linalg.norm(F_hat)
            if l < 100*np.spacing(1):
                w_fea[i] = 1000
                continue
            else:
                F_hat = F_hat/l
            a = np.array(np.dot(np.transpose(F_hat), U))
            a = np.multiply(a, a)
            a = np.transpose(a)

            # use f'Lf formulation
            if style == -1:
                w_fea[i] = np.sum(a * s)
            # using all eigenvalues except the 1st
            elif style == 0:
                a1 = a[0:n_samples-1]
                w_fea[i] = np.sum(a1 * s[0:n_samples-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
            # use first k except the 1st
            else:
                a1 = a[n_samples-style:n_samples-1]
                w_fea[i] = np.sum(a1 * (2-s[n_samples-style: n_samples-1]))

        if style != -1 and style != 0:
            w_fea[w_fea == 1000] = -1000

        return w_fea

    def get_laplace_scores(self, nodes: List, W):
        """
        This function implements the laplacian score feature selection, steps are as follows:
        1. Construct the affinity matrix W if it is not specified
        2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
        3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
        4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        kwargs: {dictionary}
            W: {sparse matrix}, shape (n_samples, n_samples)
                input affinity matrix

        Output
        ------
        score: {numpy array}, shape (n_features,)
            laplacian score for each feature

        Reference
        ---------
        He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
        """
        # W = Similarity(self.settings, community).construct_W()

        n_samples, _ = np.shape(self.data)
        n_nodes = len(nodes)

        # compute pairwise euclidean distances
        # D = pairwise_distances(self.data, metric=self.settings.metric.value)
        # D **= 2
        # sort the distance matrix D in ascending order
        # idx = np.argsort(D, axis=1)
        # choose the k-nearest neighbors for each instance
        # -------------------without W>
        # idx_new = np.repeat(nodes, n_nodes)
        # G = np.zeros((n_nodes*n_nodes, 3))
        # G[:, 0] = np.tile(nodes, (n_nodes, 1)).reshape(-1)
        # G[:, 1] = np.ravel(idx_new, order='F')
        # G[:, 2] = 1

        # # build the sparse affinity matrix W
        # W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
        # bigger = np.transpose(W) > W
        # W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
        # ------------------- without W>
        
        # build the diagonal D matrix from affinity matrix W
        D = np.array(np.sum(W, axis=1))
        L = W
        tmp = np.dot(np.transpose(D), self.data)
        D = diags(np.transpose(D), [0])
        Xt = np.transpose(self.data)
        t1 = np.transpose(np.dot(Xt, D.todense()))
        t2 = np.transpose(np.dot(Xt, L.todense()))
        # compute the numerator of Lr
        D_prime = np.sum(np.multiply(t1, self.data), 0) - np.multiply(tmp, tmp)/D.sum()
        # compute the denominator of Lr
        L_prime = np.sum(np.multiply(t2, self.data), 0) - np.multiply(tmp, tmp)/D.sum()
        # avoid the denominator of Lr to be 0
        D_prime[D_prime < 1e-12] = 10000

        # compute laplacian score for all features
        score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
        return np.transpose(score)
    
    def get_scores(self, W=None):

        # scores = self.get_spec_scores(self.data, W)

        scores = np.zeros(self.data.shape[1])

        if self.settings.scoring_method == ScoringMethod.EUCLIDEAN_NORM:
            scores = self.get_norm2_all_fea(self.settings.scoring_metric.value, W)
        else:
            for _, c in enumerate(self.communities):
                nodes = list(c)
                community_data = self.data[nodes]

                if self.settings.scoring_method == ScoringMethod.LAPLACE:
                    scores += self.get_laplace_scores(nodes, W)
                elif self.settings.scoring_method == ScoringMethod.ORDINAL:
                    scores += self.get_ordinal_scores(community_data)
                elif self.settings.scoring_method == ScoringMethod.EUCLIDEAN_NORM:
                    scores += self.get_norm2_scores(nodes, self.settings.scoring_metric.value)
                elif self.settings.scoring_method == ScoringMethod.VARIANCE:
                    scores += self.get_variance_scores(nodes)
        
        # scores = scores / len(self.communities)

        return scores