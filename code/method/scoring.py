from typing import List, Any
import numpy as np
from scipy.sparse import *
from sklearn.ensemble import RandomForestClassifier

class Scoring:
    def __init__(self, communities: List[Any], data: np.ndarray):
        self.communities = communities
        self.data = data

    def get_scores(self):
        scores = []
        objs, n_features = self.data.shape
        
        clf = RandomForestClassifier(random_state=0)
        clustering = np.zeros(objs)

        for _, values in enumerate(self.communities):
            for idx in values:
                clustering[idx] = 1
            
            clf.fit(self.data, clustering)
            scores.append(clf.feature_importances_)
            # scores.append(clf.coef_[0])

            clustering = np.zeros(objs)
        
        # return np.mean(scores, axis=0)
        return np.linalg.norm(scores, axis=0)