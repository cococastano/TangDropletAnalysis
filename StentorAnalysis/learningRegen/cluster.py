import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans

# DESCRIPTION:
# Performs a cluster analysis. Expects a timepoint and picture index for each sample.
# Plots histogram of each cluster with selected sample images to associate with time point
# Plots samples by first two principal components to verify no overfitting

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

########################################################################################################################
N_PCACMP = 4
N_CLUSTER = 4

########################################################################################################################
# Read data
df = pd.read_csv("20180802_cluster_data_filtered.csv", index_col=None)

full_data = df.values
#full_data = full_data[full_data[:,1] < 274, :]

print(full_data.shape)

# Column 1 - Time point
# Column 2 - Image index
# Column 3+ - Data
t = full_data[:, 1]
ind = full_data[:, 2]
data = full_data[:, 3:]

# Normalize and center data
u = np.mean(data, axis=0)
sd = np.std(data, axis=0)

data = (data - u) / sd

# Filter outliers (Possibly poor contouring)
mask = np.abs(data) >= 5
linemask = ~np.any(mask, axis=1)
linemask = np.reshape(linemask, (len(linemask), 1))

print(linemask.shape)
data = data[linemask[:, 0], :]
t = t[linemask[:, 0]]
ind = ind[linemask[:, 0]]

# Unique times
# OLD - used to color by time point
time, time_counts = np.unique(t, return_counts=True)
max_t = float(np.max(time))

########################################################################################################################
# PCA
pca = PCA(n_components=N_PCACMP)
pca.fit(data)

PC = np.dot(data, pca.components_.T)

print(PC.shape)

# HOW MUCH IS FROM TEXTURE:
print(pca.components_.T[:, 0])

## TEMP GET RID OF OUTLIERS
# mask1 = np.abs(PC[:, 0]) < 10
# mask2 = np.abs(PC[:, 1]) < 10
# mask3 = np.abs(PC[:, 2]) < 10
#
# mask = mask1 & mask2 & mask3
#
# data = data[mask, :]
# t = t[mask]
# ind = ind[mask]


pca.fit(data)
PC = np.dot(data, pca.components_.T)

# CLUSTER
specCluster = SpectralClustering(n_clusters=N_CLUSTER, random_state=1, n_init=10, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans')
specLabels = specCluster.fit_predict(PC)

kMeanCluster = KMeans(n_clusters=N_CLUSTER, random_state=1, n_init=10)
kMeanLabels = kMeanCluster.fit_predict(PC)

u_labels = np.unique(specLabels)
print(u_labels)

########################################################################################################################
# PLOT
# TODO Write as a separate function

means_sort = np.zeros((N_CLUSTER,))

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

for c in np.arange(N_CLUSTER):
    rows = [d == u_labels[c] for d in specLabels]
    ax.scatter(PC[rows, 0], PC[rows, 1], PC[rows, 2], cmap='Accent')

    means_sort[c] = np.mean(t[rows])

    imgPool = ind[rows]
    n_draw = 5
    if len(imgPool) < 5:
        n_draw = len(imgPool)

    select = np.random.choice(imgPool, n_draw, replace=False)
    stacked = []
    for pic in np.arange(len(select)):
        filename = '20180802_drawn_pics/' + str(int(select[pic])) + '.png'
        print(filename)
        imgToAdd = cv.imread(filename, cv.IMREAD_COLOR)
        imgToAdd = np.uint8(imgToAdd)
        if pic == 0:
            stacked = np.copy(imgToAdd)
        else:
            stacked = np.hstack((stacked, imgToAdd))

    savename = 'specCluster_orig_' + str(int(u_labels[c])) + '.png'
    cv.imwrite(savename, stacked)

plt.figure(1)
plt.legend(('C-0', 'C-1', 'C-2', 'C-3', 'C-4', 'C-5'), fontsize=18)
ax.set_xlabel('PC 1', fontsize=18)
ax.set_ylabel('PC 2', fontsize=18)
ax.set_zlabel('PC 3', fontsize=18)

plt.title('SpecCluster', fontsize=22)

order = np.argsort(means_sort)

plt.figure(2)
for c in np.arange(N_CLUSTER):
    pos = order[c]
    plt.subplot(N_CLUSTER, 1, c + 1)
    rows = [d == u_labels[pos] for d in specLabels]
    plt.hist(t[rows], bins=[0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    median = np.median(t[rows])
    plt.plot([median, median], [0, 30], c='r')
    plt.ylabel('C-%d' % pos, fontsize=14)
    plt.ylim((0, 30))
    plt.suptitle('Histogram of Time post-wound (hrs)', fontsize=20)

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

for c in np.arange(N_CLUSTER):
    rows = [d == u_labels[c] for d in kMeanLabels]
    ax.scatter(PC[rows, 0], PC[rows, 1], PC[rows, 2], cmap='Accent')

    means_sort[c] = np.mean(t[rows])

    imgPool = ind[rows]
    n_draw = 5
    if len(imgPool) < 5:
        n_draw = len(imgPool)

    select = np.random.choice(imgPool, n_draw, replace=False)
    stacked = []
    for pic in np.arange(len(select)):
        filename = '20180802_drawn_pics/' + str(int(select[pic])) + '.png'
        imgToAdd = cv.imread(filename, cv.IMREAD_COLOR)
        imgToAdd = np.uint8(imgToAdd)
        if pic == 0:
            stacked = np.copy(imgToAdd)
        else:
            stacked = np.hstack((stacked, imgToAdd))

    savename = 'kMeanCluster_orig_' + str(int(u_labels[c])) + '.png'
    cv.imwrite(savename, stacked)

plt.figure(3)
plt.legend(('C-0', 'C-1', 'C-2', 'C-3', 'C-4', 'C-5'), fontsize=18)
ax.set_xlabel('PC 1', fontsize=18)
ax.set_ylabel('PC 2', fontsize=18)
ax.set_zlabel('PC 3', fontsize=18)
plt.title('KMeanCluster', fontsize=22)

order = np.argsort(means_sort)

plt.figure(4)
for c in np.arange(N_CLUSTER):
    pos = order[c]
    plt.subplot(N_CLUSTER, 1, c + 1)
    rows = [d == u_labels[pos] for d in kMeanLabels]
    plt.hist(t[rows], bins=[0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    median = np.median(t[rows])
    plt.plot([median, median], [0, 30], c='r')
    plt.ylabel('C-%d' % pos, fontsize=14)
    plt.ylim((0, 30))


plt.show()