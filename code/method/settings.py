from enum import Enum

class DistanceMetric(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'
    MANHATTAN = 'manhattan'

class CommunityMethod(Enum):
    LOUVAIN = 'LOUVAIN'
    GREEDY = 'GREEDY'
    GIRVAN = 'GIRVAN'
    LABEL_PROPAGATION = 'LABEL_PROPAGATION'

class Settings:
    def __init__(self, k:int, metric:DistanceMetric, community_method: CommunityMethod, scoring_metric: DistanceMetric=DistanceMetric.EUCLIDEAN):
        self.k = k
        self.metric = metric
        self.community_method = community_method        
        self.scoring_metric = scoring_metric