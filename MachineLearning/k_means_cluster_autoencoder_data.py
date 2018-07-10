import numpy as np
import os.path
import data_utils
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import cv2
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, KMeans
from autoencoder_mod import autoencoder
"""
Nicolas Castano 
Last revision: 6/17/18

Run k-means cluster off of autoencoded data. Indicate model_name and adjust
path in model_file

"""
# will we be using GPUs?
USE_GPU = False
if USE_GPU and torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')
# float values used
dtype = torch.float32

def encode_data(loader, model,ret_n_params=3):
    # set model to evaluation mode
    model.eval()
    dims = np.empty((0,ret_n_params), float)    
    classes = np.empty((0,1), int)  
    index = 0
    with torch.no_grad():
        for X, y in loader:
            # move to device, e.g. GPU or CPU
            X = X.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=torch.long)
            # cv2.imshow('decoded im',model.get_decoded_im(X))
            # cv2.waitKey(1000)
            encoded = model.encode_to_n_dims(X,n=ret_n_params)[0]
            dims = np.append(dims, [np.array(encoded)],axis=0)
            classes = np.append(classes, [[np.array(y)[0][0]]], axis=0)
            index += 1
            if index % 20 == 0: print('encoding dataset', index)
    return (dims,classes)

model_name = 'auto_encode_v4'
model_file = 'C:/Users/nicas/Documents/CS231N-ConvNNImageRecognition/' + \
             'Project/' + model_name + '.pt'

# to avoid memory problems partition data calls and loop over frame ranges
def chunks(l, n):
    """yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('loading existing model...')
my_model = torch.load(model_file)
encoded_vals = []

n_params = 50

data = np.empty((0,n_params), float)
data_3d = np.empty((0,3), float)
classes = np.empty((0,1), int)

class_names = ['no break', 'break']  # 0 is no break and 1 is break
frame_range = list(range(0,56,4))
num_classes = len(class_names)
part_frame_range = list(chunks(frame_range,2))
for i, sub_range in enumerate(part_frame_range):
    print()
    print('PULLING PARTITION %d OF %d' % (i,len(part_frame_range)-1))
    num_train = 34*len(sub_range) # 3400
    num_val = 2*len(sub_range) # 200
    num_test = 10*len(sub_range) # 188
    (X_train, y_train,
     X_val, y_val, X_test, y_test) = \
         data_utils.get_data(frame_range=sub_range,
                             num_train=num_train,
                             num_validation=num_val,
                             num_test=num_test,
                             feature_list=None,
                             reshape_frames=False,
                             crop_at_constr=True,
                             blur_im=True)
                 
    # create tesor objects, normalize and zero center and pass into data 
    #loaders
    # hardcoded means and standard deviation of pixel values
    #data
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    
    X_all = torch.cat((X_train,X_val,X_test),0)
    y_all = torch.cat((y_train,y_val,y_test),0)
    all_data = torch.utils.data.TensorDataset(X_all, y_all)
    print('data shape: ', X_all.shape)
    print()
    loader_all_data = torch.utils.data.DataLoader(all_data, shuffle=False)
    # pull out encoded dims
    print('ENCODING TO %d PARAMS' % (n_params))
    params,out_classes = encode_data(loader_all_data,my_model,
                                 ret_n_params=n_params)
    print('ENCODING TO 3 PARAMS')
    params_3d,_ = encode_data(loader_all_data,my_model,
                                    ret_n_params=3)
    data = np.append(data,params,axis=0)
    classes = np.append(classes,out_classes)
#    print(len(classes),np.count_nonzero(classes),len(classes)-np.count_nonzero(classes))
    data_3d = np.append(data_3d,params_3d,axis=0)
    
    
# zero center and normalize
means = np.mean(data, axis=0)
sd = np.std(data, axis=0)
data = (data - means) / sd

means_3d = np.mean(data_3d, axis=0)
sd_3d = np.std(data_3d, axis=0)
data_3d = (data_3d - means_3d) / sd_3d

    
# CLUSTER
N_CLUSTER = 2
specCluster = SpectralClustering(n_clusters=N_CLUSTER, random_state=1, 
                                 n_init=30)
specLabels = specCluster.fit_predict(data)

kMeansCluster = KMeans(n_clusters=N_CLUSTER, random_state=1, n_init=30)
kMeansLabels = kMeansCluster.fit_predict(data)

u_labels_spec = np.unique(specLabels)
u_labels_kMeans = np.unique(kMeansLabels)
print(u_labels_spec)
print(u_labels_kMeans)

plt.close('all')

# break and no break color maps
cmap_break = colormap.get_cmap('winter')
cmap_nobreak = colormap.get_cmap('autumn')
break_colors = []
nobreak_colors = []
for c in np.arange(1,N_CLUSTER+1):
    pull_color = float(c/N_CLUSTER)
    break_colors.append(cmap_break(pull_color))
    nobreak_colors.append(cmap_nobreak(pull_color))
    

fig2D_spec = plt.figure('2D_spec')
fig2D_kMeans = plt.figure('2D_kMeans')
fig3D_spec = plt.figure('3D_spec')
ax_spec = fig3D_spec.add_subplot(111, projection='3d')
fig3D_kMeans = plt.figure('3D_kMeans')
ax_kMeans = fig3D_kMeans.add_subplot(111, projection='3d')

my_clusters = []

for cluster in np.arange(N_CLUSTER):
    my_clusters.append('cluster ' + str(cluster))
    # plot spectral clusters with o markers for no break and * for break
    rows_spec = [d == u_labels_spec[cluster] for d in specLabels]
    x = data_3d[rows_spec, 0]
    y = data_3d[rows_spec, 1]
    z = data_3d[rows_spec, 2]
    c = classes[rows_spec]
    x_nobreak, y_nobreak, z_nobreak = [], [], []
    x_break, y_break, z_break = [], [], []
    for (i,x,y,z) in zip(c,x,y,z):
        if classes[i] == 1:
            x_nobreak.append(x)
            y_nobreak.append(y)
            z_nobreak.append(z)
        elif classes[i] == 0:
            x_break.append(x)
            y_break.append(y)
            z_break.append(z)
    plt.figure('2D_spec')
    plt.scatter(x_nobreak, y_nobreak, 
                marker='o', color=nobreak_colors[cluster])
    plt.scatter(x_break, y_break, 
                marker='*', color=break_colors[cluster])
    
    plt.figure('3D_spec')
    ax_spec.scatter(x_nobreak, y_nobreak, z_nobreak, 
                    marker='o', color=nobreak_colors[cluster])
    ax_spec.scatter(x_break, y_break, z_break, 
                    marker='*',color=break_colors[cluster])
        
    # plot k means clusters
    rows_kMeans = [d == u_labels_kMeans[cluster] for d in kMeansLabels]
    x = data_3d[rows_kMeans, 0]
    y = data_3d[rows_kMeans, 1]
    z = data_3d[rows_kMeans, 2]
    c = classes[rows_kMeans]
    x_nobreak, y_nobreak, z_nobreak = [], [], []
    x_break, y_break, z_break = [], [], []
    for (i,x,y,z) in zip(c,x,y,z):
        if classes[i] == 1:
            x_nobreak.append(x)
            y_nobreak.append(y)
            z_nobreak.append(z)
        elif classes[i] == 0:
            x_break.append(x)
            y_break.append(y)
            z_break.append(z)
    plt.figure('2D_kMeans')
    plt.scatter(x_nobreak, y_nobreak, 
                marker='o', color=nobreak_colors[cluster])
    plt.scatter(x_break, y_break,
                marker='*', color=break_colors[cluster])
    
    plt.figure('3D_kMeans')
    ax_kMeans.scatter(x_nobreak, y_nobreak, z_nobreak, 
                      marker='o', color=nobreak_colors[cluster])
    ax_kMeans.scatter(x_break, y_break, z_break, 
                      marker='*', color=break_colors[cluster])


plt.figure('2D_spec')
plt.legend(my_clusters, fontsize=18)
plt.xlabel('dimension 1', fontsize=18)
plt.ylabel('dimension 2', fontsize=18)
plt.title('2D spectral clustering', fontsize=22)

plt.figure('2D_kMeans')
plt.legend(my_clusters, fontsize=18)
plt.xlabel('dimension 1', fontsize=18)
plt.ylabel('dimension 2', fontsize=18)
plt.title('2D k-means clustering', fontsize=22)

plt.figure('3D_spec')
ax_spec.set_xlabel('dimension 1')
ax_spec.set_ylabel('dimension 2')
ax_spec.set_ylabel('dimension 3')
ax_spec.set_title('2D spectral clustering')

plt.figure('3D_kMeans')
ax_kMeans.set_xlabel('dimension 1')
ax_kMeans.set_ylabel('dimension 2')
ax_kMeans.set_ylabel('dimension 3')
ax_kMeans.set_title('2D k-means clustering')

