# MSUFS
Unsupervised Feature Selection Method Based on Graph Modularity and Data Clustering

# Abstract

Feature selection is an important area of research aimed at eliminating unwanted features from high-dimensional datasets. In unsupervised learning, the selection becomes even more challenging because data labels are not available. Many unsupervised methods are suggested in the literature, but the computational complexity, whether due to the excessive number of hyper-parameters or the execution time, makes some methods unfeasible for a real application. Another highlight of the methods is that the selection, in many cases, is not guided by a clustering criterion, making the basic idea of these methods a kind of reconstruction of the original set in a low dimensionality. In this work we propose a new unsupervised feature selection method, called MSUFS, capable of performing the feature selection in a low computational consumption and the most important features are listed according to the power to maximize a previously identified partition. The results of the experiments carried out in this work demonstrated that the MSUFS method, in comparison with the state-of-the-art methods, obtained good performances when the most popularly used metrics for the clustering activity were observed. Friedman's statistical test was also used to give more robustness to the obtained results.

# Pre-Conditions
It's necessary install some packages/softwares before execution:

- Octave GNU Linux
- Scikit-learn (package to execute K-Means)
- oct2py (package to execute octave code)
- Anaconda 1.6+

# Execution
For MSUFS execution run main.py in the code/ folder
