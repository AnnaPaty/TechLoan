import numpy as np
import pandas as pd

# Decomposition
import umap
import sklearn.decomposition as skld
import sklearn.manifold as sklm

# Clustering
import sklearn.cluster as sklc
import scipy.cluster.hierarchy as sch
from sklearn.metrics import calinski_harabasz_score  # A.K.A Pseudo-F Statistic

# Visualization
from matplotlib import pyplot as plt

# Save model
import pickle

# Find the data
import os
comitter_data_csv = os.getcwd().split("TechLoan")[0]+"TechLoan/data/processed/committer-level_dataframe.csv"
df = pd.read_csv(comitter_data_csv)

# Data used by the clustering methods should not include the names of the committers
data = df.loc[:, df.columns != 'COMMITTER']


def plot_data(data, transform=None, clusters=None, title=None, fig_dim=(5, 5)):
    """
    Displays scatterplot of the data with an optional transformation and clusters.

    > data (no default): The numerical data to be represented in the scatterplot. If
    no transform is indicated, it should be 2D.
    > transform (default None): The dimensionality reduction transform to obtain 
    bidimensional data to represent. Available transforms are: 'umap', 'pca' and 'tnse', 
    although 'pca' is not recommended. If None, no transform is applied to the data
    so it is expected that it already is bidimensional.
    > clusers (default None): array of numeric labels for each data point (same length
    as the number of rows in 'data') to paint each label with a distinct color.
    If None, all data points have the same color.
    > title (default None): adds title to the plot. If None, no title is added.
    > fig_dim (default (5, 5)): tuple of two elements (width,height) representing the size of the dendogram (ignored
    if the dendogram is not visualized).
    """
    if transform is None:
        x = data
    elif transform == "umap":
        x = umap.UMAP().fit_transform(data)
    elif transform == "pca":
        print("Warning: PCA is not recommended for TDD.")
        x = skld.PCA().fit_transform(data)
    elif transform == "tnse":
        TNSEembedder = sklm.TSNE(perplexity=8.0, early_exaggeration=40.0, learning_rate=100,
                             n_iter=1000, min_grad_norm=1e-07, metric='euclidean',
                             random_state=1234, method='exact', n_jobs=-1)
        x = TNSEembedder.fit_transform(data.values)
    else:
        print("transform does not exist")
        return

    fig = plt.figure(figsize=fig_dim)
    if clusters is not None:
        plt.scatter(x[:,0], x[:,1], c = clusters)
    else:
        plt.scatter(x[:,0], x[:,1])
    if title is not None:
        plt.title(title)
    fig.show()


def hieragglo(data, transform=None, linkage="ward", criteria=None, parameter=None, dendro=False, fig_dim=(20, 8)):
    """
    This function can be used to explore the data visualizing the dendograms without the need to
    define a criteria for the clustering (criteria=None), returning the corresponding linkage
    matrix and, if a criteria is defined, it will return a vector integers representing the
    assigned cluster for each entry.

    > data (no default): dataframe of entries and their features.
    > transform (default None): transformation to be applied to the data.
    > linkage (default ward): string defining the type of linkage used for the linkage matrix ("single",
    "average", "complete", "ward").
    > criteria (default None): If criteria is "n_clusters", function argument 'parameter' will represent
    the number of clusters that will be returned. If criteria is "height", function argument
    'parameter' will represent the height where the dendogram must be cut. Else,
    if criteria is None, no cluster will be returned; use this option to explore
    the data ('parameter' will be ignored).
    > dendro (default None): boolean representing whether the dendogram should be visualized or not.
    > fig_dim (default (20,8)): tuple of two elements (width,height) representing the size of the dendogram (ignored
    if the dendogram is not visualized).
    """
    if type(linkage) is not str or linkage not in {"single", "average", "complete", "ward"}:
        print("ERROR: argument 'linkage' can only have values 'single', 'average', 'complete', 'ward'")
        return

    do_clustering = criteria is not None and parameter is not None

    if transform is not None:
        if transform == "umap":
            x = umap.UMAP().fit_transform(data)
        else:
            print("ERROR: argument 'transform' can only have values 'umap'")
            return None
    else:
        x = data

    linkage_mat = sch.linkage(x, method=linkage)

    if dendro:         # plotting the dendogram
        fig = plt.figure(figsize=fig_dim)
        _ = sch.dendrogram(linkage_mat)
        plt.title(f"Dendrogram (linkage = {linkage})")
        if do_clustering and criteria == "height":
            plt.axhline(y=parameter, color='r', linestyle='--')
        plt.show()

    if do_clustering:  # computing the clusters
        if criteria == "n_clusters":
            clusters = sch.cut_tree(linkage_mat, n_clusters=parameter)
        elif criteria == "height":
            clusters = sch.cut_tree(linkage_mat, height=parameter)
        return clusters
    return None

if __name__=="__main__":
    embedded = umap.UMAP().fit_transform(data)
    ## Best clusters for k=3 and k=4 ## 
    c3 = hieragglo(embedded, transform=None, linkage="complete", criteria="n_clusters", parameter=3, dendro=False)
    c4 = sklc.KMeans(n_clusters=3).fit_predict(embedded)
    
    # Save models
    pickle.dump(c3, open('./../../models/hieragglo.sav'))
    pickle.dump(c3, open('./../../models/kmeans.sav'))

