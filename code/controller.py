from method.settings import Settings
from method.scoring import Scoring
from method.graph import Graph
from helpers.similarity import Similarity
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import KernelPCA

from sklearn import metrics

class Controller:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.scaled_data = StandardScaler().fit_transform(data)

    def update_settings(self, setup: Settings):
        self.setup = setup
    
    def get_ranking(self):

        # data_cov = np.cov(self.data)
        # X_transformed = KernelPCA(n_components=50, kernel='linear', n_jobs=-1).fit_transform(self.data)
        # scaled_data = StandardScaler().fit_transform(self.data)
        
        # W = kneighbors_graph(scaled_data, self.setup.k, metric=self.setup.metric.value, n_jobs=-1)

        # feat0 = np.min(self.scaled_data, axis=0)
        # feat1 = np.max(self.scaled_data, axis=0)
        # feat2 = np.mean(self.scaled_data, axis=0)
        # # feat3 = np.var(self.scaled_data, axis=0)
        # feat4 = np.median(self.scaled_data, axis=0)
        # # feat5 = np.quantile(self.scaled_data, 0.25, axis=0)
        # # feat6 = np.quantile(self.scaled_data, 0.75, axis=0)
        # # feat7 = np.linalg.norm(self.data, axis=0)

        # X_feat = np.transpose([feat0, feat1, feat2, feat4])
        # n_hyperfeat = 5

        # kmeans = KMeans(
        #     n_clusters=n_hyperfeat, 
        #     init='k-means++', 
        #     verbose=0, 
        #     random_state=0, 
        #     copy_x=True 
        # )
        # model = kmeans.fit(X_feat)

        # # silhouette = metrics.silhouette_score(X_feat, model.labels_, metric='euclidean')

        # X_new = []
        # for i in range(0, n_hyperfeat):
        #     hyperfeat_norm = np.linalg.norm(self.scaled_data[:, model.labels_ == i], axis=1)
        #     X_new.append(hyperfeat_norm)
        
        # X_new = np.transpose(X_new)
        
        W = Similarity(self.setup, self.scaled_data).construct_W()
        # D = np.diag(W.sum(axis=0))

        # L = D - W

        # vals, vecs = np.linalg.eig(L)
        # idx = np.argsort(vals)
        # reduced = vecs[:, idx[0:10]].real

        # A = kneighbors_graph(self.data, self.setup.k, metric=self.setup.metric.value, mode='connectivity', n_jobs=-1)
                
        graph = Graph(W)

        partition = graph.get_communities(self.setup.community_method)
        modularity = graph.get_modularity(partition)

        # print('modularity: ', modularity)

        # modularity = 0

        # kmeans = KMeans(
        #     n_clusters=10, 
        #     init='k-means++', 
        #     n_init=50, 
        #     verbose=0, 
        #     copy_x=True, 
        #     n_jobs=-1,
        #     random_state=0)
        # kmeans.fit(reduced)

        # partition = kmeans.labels_
        # p = {}
        # for i, v in enumerate(partition):
        #     p.setdefault(v, set()).add(i)
        
        # partition = [v for v in p.values()]

        n_cluster = len(partition)

        scores = Scoring(partition, self.scaled_data, self.setup).get_scores(W)
        feat_idx = np.argsort(scores, 0)
        feat_idx = feat_idx[::-1]
        
        return feat_idx, n_cluster, modularity