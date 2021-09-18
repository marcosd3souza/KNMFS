from enum import Enum

class ScoringMethod(Enum):
    ORDINAL = 'ORDINAL'
    EUCLIDEAN_NORM = 'EUCLIDEAN_NORM'
    LAPLACE = 'LAPLACE'
    VARIANCE = 'VARIANCE'

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
    def __init__(self, k:int, metric:DistanceMetric, community_method: CommunityMethod, scoring_method: ScoringMethod, scoring_metric: DistanceMetric=DistanceMetric.EUCLIDEAN):
        self.k = k
        self.metric = metric
        self.community_method = community_method
        self.scoring_method = scoring_method
        self.scoring_metric = scoring_metric