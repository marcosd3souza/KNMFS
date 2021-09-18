import networkx as nx
from itertools import islice
from networkx.algorithms import community
from networkx.algorithms.community.quality import modularity
import community as community_louvain
from enum import Enum
from typing import List, Any
from method.settings import CommunityMethod

class Graph:
    def __init__(self, w):
        self.G = nx.from_scipy_sparse_matrix(w)
    
    def get_modularity(self, communities):
        return modularity(self.G, communities)

    def get_louvain_communities(self):
        partition = community_louvain.best_partition(self.G, resolution=1.0, randomize=False)
        v = {}

        for key, value in partition.items():
            v.setdefault(value, set()).add(key)

        return [i for i in v.values()]

    def get_label_propagation_communities(self):
        return list(community.label_propagation_communities(self.G))

    def get_greedy_communities(self):
        return list(community.greedy_modularity_communities(self.G))

    def get_girvan_communities(self):
        gn_generator = community.girvan_newman(self.G)
        return next(islice(gn_generator, 0, None))
    
    def get_communities(self, method: CommunityMethod) -> List[Any]:
        communities = []

        if method == CommunityMethod.LOUVAIN:
            communities = self.get_louvain_communities()
        elif method == CommunityMethod.GREEDY:
            communities = self.get_greedy_communities()
        elif method == CommunityMethod.GIRVAN:
            communities = self.get_girvan_communities()
        elif method == CommunityMethod.LABEL_PROPAGATION:
            communities = self.get_label_propagation_communities()

        return communities            

        
        
