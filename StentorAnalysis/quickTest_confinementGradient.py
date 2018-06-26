import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans

# Check OpenCV loaded
print("Open CV version " + cv.__version__)

N_PCACMP = 3

df_cg = pd.read_csv("GC_contour_data_updated.csv", index_col=None)

cg_full_data = df_cg.values
#full_data = full_data[full_data[:,1] < 274, :]

# Column 0 - Time point
# Column 1 - Image index
# Column 2+ - Data
t_cg = cg_full_data[:, 0]
ind_cg = cg_full_data[:, 1]
data_cg = cg_full_data[:, 2:]

# Normalize and center data
u = np.mean(data_cg, axis=0)
sd = np.std(data_cg, axis=0)

data_cg = (data_cg - u) / sd

pca = PCA(n_components=N_PCACMP)
pca.fit(data_cg)

cg_PC = np.dot(data_cg, pca.components_.T)

plt.figure(1)
case1_idx = [0, 6, 7, 8, 9, 10]    # Unwounded
case2_idx = [11, 12, 13, 1]         # Wounded but healed
case3_idx = [2, 3, 4, 5]            # Wounded but unhealed

case1 = cg_PC[case1_idx, :]
case2 = cg_PC[case2_idx, :]
case3 = cg_PC[case3_idx, :]

plt.scatter(case1[:, 0], case1[:, 1], c='b', marker='*', s=200)
plt.scatter(case2[:, 0], case2[:, 1], c='k', marker='*', s=200)
plt.scatter(case3[:, 0], case3[:, 1], c='g', marker='*', s=200)
plt.legend(('Unwounded', 'Wounded + Healed', 'Wounded + No Heal'))
plt.title('Quick look at CG cells', fontsize=18)
plt.xlabel('PC1', fontsize=16)
plt.ylabel('PC2', fontsize=16)


plt.show()