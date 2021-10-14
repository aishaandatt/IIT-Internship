import numpy as np

from collections import defaultdict
from sklearn.cluster import KMeans

# this is your array with the values
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])


# This function creates the classifier
# n_clusters is the number of clusters you want to use to classify your data
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# you can see the labels with:
print(kmeans.labels_)

# the output will be something like:
#array([0, 0, 0, 1, 1, 1], dtype=int32)
# the values (0,1) tell you to what cluster does every of your data points correspond to

# You can predict new points with
kmeans.predict([[0, 0], [4, 4]])

#array([0, 1], dtype=int32)

# or see were the centres of your clusters are
kmeans.cluster_centers_
# array([[ 1.,  2.],
#     [ 4.,  2.]])
