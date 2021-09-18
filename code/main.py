from method.settings import Settings, DistanceMetric, ScoringMethod, CommunityMethod
from controller import Controller
from helpers.data_reader import DataReader
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_model(x_selected, n_clusters):
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init='k-means++', 
        n_init=200, 
        verbose=0, 
        random_state=0, 
        copy_x=True 
    )
    return kmeans.fit(x_selected)


def get_indices(data, y_true, n_clusters):
    model = get_model(data, n_clusters)
    
    y_predict = model.labels_

    for i in range(0, len(y_predict)):
        y_predict[i] = y_predict[i] + 1

    # calculate NMI
    nmi = -1
    if y_true is not None:
        nmi = v_measure_score(y_true, y_predict)
    
#     avg_sil = metrics.silhouette_score(data, model.labels_, metric='euclidean')
    avg_sil = metrics.silhouette_score(data, model.labels_, metric='euclidean')

    corrected_rand = adjusted_rand_score(y_true, y_predict)

    acc = accuracy_score(y_true, y_predict)
           
    return avg_sil, nmi, corrected_rand, acc


def get_results(x_selected, y_true, n_cluster):
    
    x_selected_scaled = StandardScaler().fit_transform(x_selected)  

    return get_indices(x_selected_scaled, y_true.values, n_cluster)

datasets = [   
     ('image', 'warpAR10P'),
     ('image', 'warpPIE10P'),
     ('image', 'pixraw10P'),
     ('image', 'orlraws10P'),
     ('biological', 'nci9'),
     ('biological', 'ProstateGE'), 
     ('biological', 'Carcinom'), 
     ('biological', 'TOX171'), 
     ('biological', 'ALLAML'),
     ('biological', 'SMK_CAN'), 
     ('text', 'PCMAC'),
     ('text', 'RELATHE'),
     ('text', 'BASEHOCK'),
# # # #     ('others', 'ELD_2011_2014'),
     ('others', 'isolet')
]

silhouettes = []

nmi_means = []
nmi_stds = []
nmi_max = []

cr_means = []
cr_stds = []
cr_max = []

acc_means = []
acc_stds = []
acc_max = []

nmis_select = []

for domain, name in datasets:
    print('--------------------------------------------------------------------------')
    print('---------- dataset results ----------')
    
    print('domain: ', domain)
    print('dataset: ', name)
    X, y = DataReader(name).get_dataset()
    data = X.values

    all_opt_to_save = pd.DataFrame()

    dt_1 = datetime.now()
    
    best_sil = 0
    sil_selected = 0
    best_nmi = 0
    nmi_select = 0
    nmi_select_by_sil = 0
    best_data = pd.DataFrame()
    f_selected = 0
    best_n_cluster = 0
    n_cluster_select = 0
    
    best_modularity = 0

    best_fitness = 0
    
    nmis = []
    crs = []   
    accs = [] 
    
#     perc = [int(p * n_features) for p in np.arange(0.01, 0.1, 0.01)]
#     perc = int(0.2 * n_features)
    magic = [m for m in range(10, 100, 10)]
    
    k_candi = [5] #3, 5, 7]
    metric_candi = [DistanceMetric.COSINE]#, DistanceMetric.EUCLIDEAN]
    
    community_method_candi = [CommunityMethod.LOUVAIN]#, CommunityMethod.GREEDY]

    scoring_metric_candi = [DistanceMetric.COSINE]#, DistanceMetric.MANHATTAN, DistanceMetric.EUCLIDEAN]
    scoring_candi = [ScoringMethod.EUCLIDEAN_NORM]# , ScoringMethod.ORDINAL, ScoringMethod.EUCLIDEAN_NORM]

    controller = Controller(data)
    # X = StandardScaler().fit_transform(data)

    for k in k_candi:
        for metric in metric_candi:
            for scoring_method in scoring_candi:
                for scoring_metric in scoring_metric_candi:
                    for community_method in community_method_candi:
                
                        setup = Settings(k, metric, community_method, scoring_method, scoring_metric)
                        controller.update_settings(setup)
                        rank, n_clusters, modularity = controller.get_ranking()                        
        #                 rank, _, k, partition = get_features_rank_by_centrality(X, config)
                        
                        for cut in magic:        
                            selection = rank[0:cut]
                            data_reduced = X.iloc[:, selection]

                    #         for k in k4:
                            sil, nmi, cr, acc = get_results(data_reduced, y, n_clusters) 
                            # print(f'sil: {sil} nmi: {nmi}')

                            nmis.append(nmi)
                            crs.append(cr)
                            accs.append(acc)

                            fitness = sil

                            partial_result_to_save = pd.DataFrame(
                                {
                                    'cut_point': [cut],
                                    'avg_sil': [sil], 
                                    'nmi': [nmi],
                                    'corrected_rand': [cr]
                                }
                            )

                            all_opt_to_save = pd.concat([all_opt_to_save, partial_result_to_save])

                            if fitness > best_fitness:
                                best_fitness = fitness

                                nmi_select_by_sil = nmi
                                best_sil = sil
                                best_modularity = modularity
                                best_data = data_reduced
                                n_cluster_select = n_clusters
                                f_selected = cut

                            if nmi > best_nmi:
                                best_nmi = nmi 

    all_opt_to_save.to_csv("./result_all_opt_MSUFS_in_" + name + ".csv", sep=";")

    best_data.to_csv("./" + name + "_after_FS_MSUFS.csv", sep=";")

    dt_2 = datetime.now()
    time_diff = dt_2 - dt_1
    print('time: ', time_diff)    

    nmis_select.append(nmi_select_by_sil)
    silhouettes.append(best_sil)

    nmi_max.append(best_nmi)
    nmi_means.append(np.mean(nmis))
    nmi_stds.append(np.std(nmis))

    cr_max.append(max(crs))
    cr_means.append(np.mean(crs))
    cr_stds.append(np.std(crs))

    acc_max.append(max(accs))
    acc_means.append(np.mean(accs))
    acc_stds.append(np.std(accs))

    print('--------------- results -----------------')
    print('NMI select: ', nmi_select_by_sil)
    print('Silhouette: ', best_sil)
    print('NMI best: ', best_nmi)

    print('--------------- cluster -----------------')
    print('n feat: ', f_selected)
    print('n cluster: ', n_cluster_select)

#     print('--------------- after -----------------')
#     _, _ = show_graph(best_data, partition)
#     print('best n cluster by sil: ', best_n_cluster)
    print('--------------------------------------------------------------------------')
    
print('Silhouette Avg: ')
print(np.mean(silhouettes))
print('nmis max Avg: ')
print(np.mean(nmi_max))
print('nmis mean Avg: ')
print(np.mean(nmi_means))
print('crs max Avg: ')
print(np.mean(cr_max))
print('crs means Avg: ')
print(np.mean(cr_means))

print('Silhouettes:')
for i in range(len(datasets)):
    print(silhouettes[i])

print('--------------------------------------------------------------------------')

print('nmis max:')
for i in range(len(datasets)):
    print(nmi_max[i])

print('--------------------------------------------------------------------------')

print('nmis means:')
for i in range(len(datasets)):
    print(nmi_means[i])

print('nmis stds:')
for i in range(len(datasets)):
    print(nmi_stds[i])

print('--------------------------------------------------------------------------')

print('crs max:')
for i in range(len(datasets)):
    print(cr_max[i])

print('crs means:')
for i in range(len(datasets)):
    print(cr_means[i])

print('crs stds:')
for i in range(len(datasets)):
    print(cr_stds[i])

print('--------------------------------------------------------------------------')

print('accs max:')
for i in range(len(datasets)):
    print(acc_max[i])

print('accs means:')
for i in range(len(datasets)):
    print(acc_means[i])

print('accs stds:')
for i in range(len(datasets)):
    print(acc_stds[i])