![KNMFS](https://github.com/marcosd3souza/KNMFS/blob/main/KNMFS.png)

# KNMFS
Unsupervised Feature Selection Method Based on Iterative Similarity Graph Factorization and Clustering by Modularity

# Abstract

Feature selection is an important research area aimed at eliminating unwanted features from high-dimensional datasets. Feature selection methods are categorized according to the availability of labels. Supervised methods usually select features that have high correlations with the data la-
bels, however, this cannot be done by unsupervised methods, since the labels are not available, making the development of them even more challenging. Some unsupervised methods proposed in the literature get around this problem by generating “pseudo-labels”, using clustering techniques, and then making feature selection as in the supervised scenario. There are also methods in which the selection is not guided by a clustering criterion, generally performing some kind of reconstruction of the original dataset in low dimensionality. However, in both approaches a local structure is generated, usually starting from a similarity matrix built by using the entire set of features. This may compromise the results of the methods when there are many irrelevant and noisy features, which makes it difficult to reveal patterns or an initial representation of the data. Another drawback of some current methods is the large amount of hyper-parameters, or the computational time required for a single execution, which can render such methods unfeasible for some large datasets. In this work we propose a new unsupervised feature selection method, called KNMFS, capable of performing feature selection with low computational time consumption, that is able to eliminate noisy features by interactively learning the similarity matrix. The results of the experiments carried out in this work demonstrated that the KNMFS method, in comparison with state-of-the-art methods, obtained good results according to the ARI and NMI metrics. Friedman’s statistical tests were also performed to give stronger evidence to the reported results.

# Pre-Conditions
It's necessary install some packages/softwares before execution:

- Octave GNU Linux
- Scikit-learn (package to execute K-Means)
- oct2py (package to execute octave code)
- Anaconda 1.6+

# Execution
For KNMFS execution run main.py in the code/ folder

# Paper
For more details access https://www.sciencedirect.com/science/article/pii/S095741742201288X
