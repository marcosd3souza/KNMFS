from method.settings import Settings
from method.scoring import Scoring
from method.graph import Graph
from helpers.similarity import Similarity
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from scipy.ndimage import gaussian_filter
from scipy import sparse

class Controller:
    def __init__(self, data: np.ndarray):
        self.data = data

    def update_settings(self, setup: Settings):
        self.setup = setup

    def get_ranking(self):
        # S = Similarity(self.setup, self.data).construct_W()
        S = self._get_nmf_S(self.data)
        # X_new, idx = self._reduce_data(S)
        # S = Similarity(self.setup, X_new).construct_W()
            
        graph = Graph(S)

        partition = graph.get_communities(self.setup.community_method)
        modularity = graph.get_modularity(partition)

        scores = Scoring(partition, self.data).get_scores()

        feat_idx = np.argsort(scores, 0)[::-1]

        print('modularity: ', modularity)
        
        return feat_idx, len(partition), modularity
    
    def _reduce_data(self, S: np.ndarray) -> np.ndarray:
        M = sparse.csr_matrix(S)
        vals, vecs = Similarity.construct_spectral_info(M)
        idx = np.argsort(vals.real)
        X_new = vecs[:, idx[0:100]].real

        return X_new, idx

    def _get_nmf_S(self, X: np.ndarray) -> np.ndarray:
        
        nmf = NMF(
            n_components=20,
            init='random',
            random_state=0,
            max_iter=5000
        )

        S = Similarity(self.setup, X).construct_W()

        # initial loss
        nmf.fit_transform(S)
        initial_loss = nmf.reconstruction_err_
        print('Initial loss: ', initial_loss)

        # Initialization
        T = 30
        best_loss = 999999999999999
        old_loss = 999999999999999
        best_S = S
        tol = 0.0001
        g_filter = 1.2

        for t in range(T):
            # a decomposição comprime os dados de entrada
            # logo as características mais relevantes (vizinhança forte)
            # tende a persistir nos dados comprimidos
            # e os detalhes menos relevantes da matriz de similaridade
            # tendem a desaparecer
            
            W = nmf.fit_transform(S)
            H = nmf.components_
            loss = nmf.reconstruction_err_

            # loss_rate.append(loss)
            if (loss > old_loss) or abs(old_loss - loss) <= tol:
                break

            if loss < best_loss:
                best_loss = loss
                best_S = Similarity(self.setup, W.dot(H)).construct_W()

            old_loss = loss
            
            S = gaussian_filter(W.dot(H), sigma=g_filter)

        print('best loss: ', best_loss)
        print('n iter: ', t)
        
        return best_S
    