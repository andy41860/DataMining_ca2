## Data Mining and Visualisation: Assignment 2
# Name: En-Jui Chang  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

np.random.seed(60)

## Data preprocessing

df = pd.read_csv("dataset", header=None, names=["features"])
# get label
label_str = df["features"].str.split().apply(lambda x: " ".join(x[0:1]))
# remove the words in the begining of the strings
df["features"] = df["features"].str.split().apply(lambda x: " ".join(x[1:]))
# convert the string type data to the float type
df["features"] = df["features"].str.split().apply(lambda x: [float(i) for i in x])
# convert 300 features from one column to 300 columns in pandas dataframe
df = pd.DataFrame([dict(zip(range(len(row)), row)) for row in df["features"]])
# convert data to numpy ndarray
x_pre = df.values

pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_pre)
x = normalized_data = normalize(pca_result)


## Create a class to implement KMeans, KMeans++, Bisecting KMeans algorithms

class KMeans:

    def __init__(self, k, max_iters=100):
        """
        Initialize the clusters(k) and the maximum number of iterations(max_iters) in the class
        """
        self.k = k
        self.max_iters = max_iters

    def _KmeansPlusPlus_init(self, x):
        """
        Initialize the centroids for the k-means++ algorithm
        """
        n_samples, n_features = x.shape
        centroids = np.empty((self.k, n_features))
        # choose first centroid at random
        centroids[0] = x[np.random.choice(n_samples)]
        for i in range(1, self.k):
            # calculate distances to previous centroids
            distances = np.sqrt(((x - centroids[:i, np.newaxis])**2).sum(axis=2))
            # calculate the probability for choosing new centroid.
            probs = (distances**2).min(axis=0) / ((distances**2).min(axis=0)).sum()
            # choose new centroid
            centroids[i] = x[np.random.choice(n_samples, p=probs)]
        return np.array(centroids)

    def fit_Kmeans(self, x):
        """
        Fit the KMeans algorithm on the input data (x)
        """
        # randomly initialize the centroids
        self.centroids = x[np.random.choice(x.shape[0], size=self.k, replace=False)]
        # changing the centroids in `max_iters` times if not converge
        for _ in range(self.max_iters):
            # calculate distances from centroids to each point
            distances = np.sqrt(((x - self.centroids[:, np.newaxis, :])**2).sum(axis=2))
            # assign each point to its closest centroid: labels.shape (n_sample(327),)
            clusters_labels = np.argmin(distances, axis=0)
            # calculate the next centroids as mean of points assigned to them
            l = list()
            for i in range(self.k):
                l.append(x[clusters_labels == i].mean(axis=0))
            next_centroids = np.array(l)
            # if new centroids = original centroids: converge and break the loop
            if np.array_equal(self.centroids, next_centroids):
                break
            # if not converge: update the centroids to new centroids
            self.centroids = next_centroids
        return self.centroids

    def fit_KmeansPlusPlus(self, x):
        """
        Fit the KMeans++ algorithm on the input data (x)
        """
        # Kmeans++ centroids initialization
        self.centroids = self._KmeansPlusPlus_init(x)
        for _ in range(self.max_iters):
            # calculate distances from centroids to each point
            distances = np.sqrt(((x - self.centroids[:, np.newaxis, :])**2).sum(axis=2))
            # assign each point to its closest centroid
            clusters_labels = np.argmin(distances, axis=0)
            # calculate the next centroids as mean of points assigned to them
            l = list()
            for i in range(self.k):
                l.append(x[clusters_labels == i].mean(axis=0))
            next_centroids = np.array(l)
            # if new centroids = original centroids: converge and break the loop
            if np.array_equal(self.centroids, next_centroids):
                break
            # if not converge: update the centroids to new centroids
            self.centroids = next_centroids
        return self.centroids

    def fit_BisectingKMeans(self, x):
        """
        Fit the Bisecting KMeans algorithm on the input data (x)
        """
        # initialize a list to keep track of all the clusters and its data
        clusters = [x]
        # repeat the process until k clusters are formed (tree structure)
        while len(clusters) < self.k:
            # choose the cluster with the largest sum of square distance
            max_dist = -1
            for c in clusters:
                dist = ((c - c.mean(axis=0))**2).sum()
                if dist > max_dist:
                    max_dist = dist
                    max_c = c
            # apply KMeans algorithm on the chosen cluster to split it into two new clusters
            bisecting_km = KMeans(k=2, max_iters=self.max_iters)
            bisecting_km.fit_Kmeans(max_c)
            # remove the original cluster
            clusters = [i for i in clusters if not np.array_equal(i, max_c)]
            # add the two new clusters
            clusters.append(max_c[bisecting_km.predict(max_c) == 0])
            clusters.append(max_c[bisecting_km.predict(max_c) == 1])
        # calculate the final centroids
        l = list()
        for c in clusters:
            l.append(c.mean(axis=0))
        self.centroids = np.array(l)
        return self.centroids

    def predict(self, x):
        """
        Predict the input data (x) belongs to which cluster (labels)
        """
        # calculate distances from centroids to each point
        distances = np.sqrt(((x - self.centroids[:, np.newaxis, :])**2).sum(axis=2))
        # retrun the clusters labels for each points
        return np.argmin(distances, axis=0) 


## Create a function to calculate the Silhouette coefficients

def silhouette_coefficient(x, labels):
    """
    Calculates the Silhouette coefficients.
    """
    # calculate the pairwise distances between all points
    distances = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            distances[i][j] = np.sqrt(np.sum((x[i] - x[j])**2))
            distances[j][i] = distances[i][j]
    # initialize a numpy arrays to save the Silhouette coefficients for each point
    s = np.zeros(x.shape[0])
    # loop through every point
    for i in range(x.shape[0]):
        # calculate the mean distance between the point and all other points in its cluster (ai)
        ai = np.mean(distances[i, labels == labels[i]])
        # initialize the minimum bi to infinity
        min_bi = np.inf
        # loop through every other cluster
        for j in range(labels.max() + 1):
            # skip the computation for the same cluster
            if j == labels[i]:
                continue
            # calculate the mean distance between the point and all points in the other cluster (bi)
            bi = np.mean(distances[i, labels == j])
            # if bi < min_bi, update the minimum bi
            if bi < min_bi:
                min_bi = bi
        # calculate the silhouette coefficients for this point
        s[i] = (min_bi - ai) / max(ai, min_bi)
    # calculate the mean silhouette coefficient of all points
    silhouette_coef = np.mean(s)        
    return silhouette_coef


## Create a function for visualizing Silhouette coefficients

def plot_silhouette_coefficients(km_coeffs, kmp_coeffs, bkm_coeffs, figsize=(8, 6)):
    """
    Visualize the Silhouette coefficients for every k in Kmeans algorithm.
    """    
    k_values = range(2, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_values, km_coeffs, label="K-Means")
    ax.plot(k_values, kmp_coeffs, label="K-Means++")
    ax.plot(k_values, bkm_coeffs, label="Bisecting K-Means")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette coefficients")
    ax.legend()
    ax.set_title("Silhouette coefficients for 3 different K-Means algorithms")
    plt.show()


## Create a function to generate a scatter plot with clusters

def plot_clusters_scatter(data, classes, type):
    """
    Visualize the data points with assigned clusters.
    """    
    colors = ["r", "g", "b"]
    for i in range(len(colors)):
        plt.scatter(data[classes == i, 0], data[classes == i, 1], c=colors[i], label=f"Cluster {i}")   
    plt.title("Scatter Plot with Clusters ({})".format(type))
    plt.xlabel("Feature 1 (PCA)")
    plt.ylabel("Feature 2 (PCA)")
    plt.legend()
    plt.show()

## Instantiate the KMeans objects and compute Silhouette coefficients.

# Standaed Kmeans algorithm
# When k=1, Silhouette coefficient doesn't exist.
km2 = KMeans(k=2)
km2.fit_Kmeans(x)
labels_km2 = km2.predict(x)
km2_coef = silhouette_coefficient(x, labels_km2)
km3 = KMeans(k=3)
km3.fit_Kmeans(x)
labels_km3 = km3.predict(x)
km3_coef = silhouette_coefficient(x, labels_km3)
km4 = KMeans(k=4)
km4.fit_Kmeans(x)
labels_km4 = km4.predict(x)
km4_coef = silhouette_coefficient(x, labels_km4)
km5 = KMeans(k=5)
km5.fit_Kmeans(x)
labels_km5 = km5.predict(x)
km5_coef = silhouette_coefficient(x, labels_km5)
km6 = KMeans(k=6)
km6.fit_Kmeans(x)
labels_km6 = km6.predict(x)
km6_coef = silhouette_coefficient(x, labels_km6)
km7 = KMeans(k=7)
km7.fit_Kmeans(x)
labels_km7 = km7.predict(x)
km7_coef = silhouette_coefficient(x, labels_km7)
km8 = KMeans(k=8)
km8.fit_Kmeans(x)
labels_km8 = km8.predict(x)
km8_coef = silhouette_coefficient(x, labels_km8)
km9 = KMeans(k=9)
km9.fit_Kmeans(x)
labels_km9 = km9.predict(x)
km9_coef = silhouette_coefficient(x, labels_km9)
# A list to save Silhouette coefficients when k=2~9 using Kmeans (for visualization)
km_coef = [km2_coef, km3_coef, km4_coef, km5_coef, km6_coef, km7_coef, km8_coef, km9_coef]

# KMeans++ algorithm
# When k=1, Silhouette coefficient doesn't exist.
kmp2 = KMeans(k=2)
kmp2.fit_KmeansPlusPlus(x)
labels_kmp2 = kmp2.predict(x)
kmp2_coef = silhouette_coefficient(x, labels_kmp2)
kmp3 = KMeans(k=3)
kmp3.fit_KmeansPlusPlus(x)
labels_kmp3 = kmp3.predict(x)
kmp3_coef = silhouette_coefficient(x, labels_kmp3)
kmp4 = KMeans(k=4)
kmp4.fit_KmeansPlusPlus(x)
labels_kmp4 = kmp4.predict(x)
kmp4_coef = silhouette_coefficient(x, labels_kmp4)
kmp5 = KMeans(k=5)
kmp5.fit_KmeansPlusPlus(x)
labels_kmp5 = kmp5.predict(x)
kmp5_coef = silhouette_coefficient(x, labels_kmp5)
kmp6 = KMeans(k=6)
kmp6.fit_KmeansPlusPlus(x)
labels_kmp6 = kmp6.predict(x)
kmp6_coef = silhouette_coefficient(x, labels_kmp6)
kmp7 = KMeans(k=7)
kmp7.fit_KmeansPlusPlus(x)
labels_kmp7 = kmp7.predict(x)
kmp7_coef = silhouette_coefficient(x, labels_kmp7)
kmp8 = KMeans(k=8)
kmp8.fit_KmeansPlusPlus(x)
labels_kmp8 = kmp8.predict(x)
kmp8_coef = silhouette_coefficient(x, labels_kmp8)
kmp9 = KMeans(k=9)
kmp9.fit_KmeansPlusPlus(x)
labels_kmp9 = kmp9.predict(x)
kmp9_coef = silhouette_coefficient(x, labels_kmp9)
# A list to save Silhouette coefficients when k=2~9 using KMeans++ (for visualization)
kmp_coef = [kmp2_coef, kmp3_coef, kmp4_coef, kmp5_coef, kmp6_coef, kmp7_coef, kmp8_coef, kmp9_coef]

# Bisecting KMeans algorithm
# When k=1, Silhouette coefficient doesn't exist.
bkm2 = KMeans(k=2)
bkm2.fit_BisectingKMeans(x)
labels_bkm2 = bkm2.predict(x)
bkm2_coef = silhouette_coefficient(x, labels_bkm2)
bkm3 = KMeans(k=3)
bkm3.fit_BisectingKMeans(x)
labels_bkm3 = bkm3.predict(x)
bkm3_coef = silhouette_coefficient(x, labels_bkm3)
bkm4 = KMeans(k=4)
bkm4.fit_BisectingKMeans(x)
labels_bkm4 = bkm4.predict(x)
bkm4_coef = silhouette_coefficient(x, labels_bkm4)
bkm5 = KMeans(k=5)
bkm5.fit_BisectingKMeans(x)
labels_bkm5 = bkm5.predict(x)
bkm5_coef = silhouette_coefficient(x, labels_bkm5)
bkm6 = KMeans(k=6)
bkm6.fit_BisectingKMeans(x)
labels_bkm6 = bkm6.predict(x)
bkm6_coef = silhouette_coefficient(x, labels_bkm6)
bkm7 = KMeans(k=7)
bkm7.fit_BisectingKMeans(x)
labels_bkm7 = bkm7.predict(x)
bkm7_coef = silhouette_coefficient(x, labels_bkm7)
bkm8 = KMeans(k=8)
bkm8.fit_BisectingKMeans(x)
labels_bkm8 = bkm8.predict(x)
bkm8_coef = silhouette_coefficient(x, labels_bkm8)
bkm9 = KMeans(k=9)
bkm9.fit_KmeansPlusPlus(x)
labels_bkm9 = bkm9.predict(x)
bkm9_coef = silhouette_coefficient(x, labels_bkm9)
# A list to save Silhouette coefficients when k=2~9 using Bisecting KMeans (for visualization)
bkm_coef = [bkm2_coef, bkm3_coef, bkm4_coef, bkm5_coef, bkm6_coef, bkm7_coef, bkm8_coef, bkm9_coef]

## Visualization: Silhouette coefficients when k=2~9 using 3 different K-Means algorithms
plot_silhouette_coefficients(km_coef, kmp_coef, bkm_coef)


## create a pandas dataframe to know the cluster of each words
df_km3 = pd.DataFrame({"word": label_str, "cluster": labels_km3}, columns=["word", "cluster"])
df_kmp3 = pd.DataFrame({"word": label_str, "cluster": labels_kmp3}, columns=["word", "cluster"])
df_bkm3 = pd.DataFrame({"word": label_str, "cluster": labels_bkm3}, columns=["word", "cluster"])
df_km3.to_csv("km3_result.csv", index=False)
df_kmp3.to_csv("kmp3_result.csv", index=False)
df_bkm3.to_csv("bkm3_result.csv", index=False)


## Visualization: a scatter plot with clusters using 3 K-means algorithms
plot_clusters_scatter(x, labels_km3, "K-means")
plot_clusters_scatter(x, labels_kmp3, "K-means++")
plot_clusters_scatter(x, labels_bkm3, "Bisecting K-means")


