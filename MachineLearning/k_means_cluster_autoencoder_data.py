import numpy as np
import os
import sys
import data_utils
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.gridspec as gridspec
import cv2
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.utils.data
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.cluster import MiniBatchKMeans
from autoencoder_mod import autoencoder
"""
Nicolas Castano 
Last revision: 7/11/18

Run k-means cluster off of autoencoded data. Indicate model_name and adjust
path in model_file

"""

################################ USEFUL METHODS ##############################
# and some preamble script
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
            index += 1
            if index % 20 == 0: print('encoding dataset', index)
    return dims


# to avoid memory problems partition data calls and loop over frame ranges
def chunks(l, n):
    """yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def pre_process_frame(frame, crop_at_constr=True, blur_im=True, invert=True):
    """just steps for preprocessing frame"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # remove gray border
    crop_locs = np.where((np.logical_or(frame == np.amin(frame),
                                        frame == np.amax(frame))))
    r1 = crop_locs[0][0]
    r2 = crop_locs[0][-1] + 1
    c1 = crop_locs[1][0]
    c2 = crop_locs[1][-1] + 1
    frame = frame[r1:r2, c1:c2]
    if crop_at_constr is True:
        constr_loc = 120# data_utils.find_constriction(frame)  # 120 for last set
        frame = data_utils.crop_my_frame(frame,[0,constr_loc,0,frame.shape[0]])
    if blur_im is True:
        frame = cv2.blur(frame,(3,3))
    # expect the pixel values to be 0 or 255 (some max) so...
    frame = np.uint8(frame)
    if invert is True:
        frame = np.invert(frame)
    return frame

#################################### SCRIPT ###################################
plt.close('all')
video_dir = 'C:/Users/nicas/Documents/' + \
                    'CS231N-ConvNNImageRecognition/' + \
                    'Project/datasets'
model_name = 'auto_encode_v4'
model_file = 'C:/Users/nicas/Documents/CS231N-ConvNNImageRecognition/' + \
             'Project/' + model_name + '.pt'
print('loading existing model...')
my_model = torch.load(model_file)

n_params = 50  # set number of parameters to encode to for clustering to use

all_files = {}     
count = 0     
for subdir, dirs, files in os.walk(video_dir):
    tags = subdir.split(os.sep)
    for file in files:
        file_name = subdir + os.sep + file
        # lets not assume all files found are .avi video files
        if '.avi' in file:
            fate = file_name.split(os.sep)[-2]
            name = fate + '_vid_' + str(count)
            count += 1
            all_files[name] = file_name
 
# dictionary has the same name key and points to a list of encoded vectors
# corresponding to each frame               
encoded_vals = {key:[] for key in list(all_files.keys())}
frame_count = 0
for i,key in enumerate(all_files):
    if i%25 == 0: print('encoding ', key)
    vid = cv2.VideoCapture(all_files[key])
    while(vid.isOpened()):
        # capture frame-by-frame
        ret, frame = vid.read()
        if ret == True:
            frame_count += 1
            # pre process the frame prior to encoding
            frame = pre_process_frame(frame)
            # get the frame dimensions from first frame of all the data
            if frame_count == 1: 
                frame_h = frame.shape[0]
                frame_w = frame.shape[1]
            # mimic data structure for data loader
            X_data = np.zeros((1, 1, frame_h, frame_w))
            y_data = np.zeros((1,1))
            X_data[0,0,:,:] = frame
            if 'nobreak'  in key:
                y_data[0] = 1
            else:
                y_data[0] = 0
                    
            X_data = torch.from_numpy(X_data)
            y_data = torch.from_numpy(y_data)
            data = torch.utils.data.TensorDataset(X_data, y_data)
            loader = torch.utils.data.DataLoader(data, shuffle=False)

            encoded_vals[key].append(encode_data(loader,my_model,
                        ret_n_params=n_params))
        
        else: 
            break
 
    # When everything done, release the video capture object
    vid.release()

# after the above code is excecuted we will have a dictionary of video tags
# pointing to vectors of encoded data; loop through these and collect all 
# vectors to then cluster
all_data = np.empty((frame_count,n_params),dtype='float')
# have to save all the keys for later reference; these names have same order
# of all_data
test_frame_names = [None] * frame_count
i = 0  # count for every vector
for key in encoded_vals:
    curr_frame = 0
    for vector in encoded_vals[key]:
        all_data[i,:] = np.ravel(vector)
        test_frame_names[i] = key + '_frame' + str(curr_frame)
        i += 1
        curr_frame += 1

# zero center and normalize
means = np.mean(all_data, axis=0)
sd = np.std(all_data, axis=0)
all_data = (all_data - means) / sd

# CLUSTER
N_CLUSTER = 5000# 800 give interesting distribution
RAND_STATE = 1
#specCluster = SpectralClustering(n_clusters=N_CLUSTER, random_state=1, 
#                                 n_init=30)
#specLabels = specCluster.fit_predict(all_data)

#kMeansCluster = KMeans(n_clusters=N_CLUSTER, random_state=1, n_init=30)
#kMeansLabels = kMeansCluster.fit_predict(all_data)

kMeansMiniBatch = MiniBatchKMeans(n_clusters=N_CLUSTER, random_state=RAND_STATE, 
                                  n_init=30)
kMeansMiniBatchLabels = kMeansMiniBatch.fit_predict(all_data)
plt.figure()
plt.hist(kMeansMiniBatchLabels, N_CLUSTER)
plt.title('k-means mini batch clustering distribution')

#u_labels_spec = np.unique(specLabels)
#u_labels_kMeans = np.unique(kMeansLabels)
u_labels_kMeansMiniBatch, counts = np.unique(kMeansMiniBatchLabels,
                                             return_counts=True)

# what are the n most represented clusters
sorted_labels = [label for _,label in sorted(zip(counts,
                                                 u_labels_kMeansMiniBatch),
                                                 reverse=True)]
sorted_names = [name for _,name in sorted(zip(counts,test_frame_names),
                                             reverse=True)]
sorted_counts = sorted(counts, reverse=True)

# show mean image of cluster
# set up plotting
n_cluster_rep = 25
subplot_rows = int(np.sqrt(n_cluster_rep))
subplot_cols = int(np.sqrt(n_cluster_rep))
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(25,12))
subplot_fig.subplots_adjust(hspace=0.2,wspace=0) 

for i in range(0,n_cluster_rep):
    ax = plt.subplot(subplot_rows, subplot_cols, i+1)
    curr_cluster = sorted_labels[i]
    print('compiling cluster', str(curr_cluster), 'with', sorted_counts[i],
          'counts' )
    locs = np.where(kMeansMiniBatchLabels == curr_cluster)
    mean_image = np.zeros((frame_h, frame_w), dtype='float')
    n = 0
    for loc in locs[0]:            
        if n <= 20:
            key = test_frame_names[loc]
            frame_key = key.split('frame')[-1]
            file_key = key.split('_frame')[0]
            vid = cv2.VideoCapture(all_files[file_key])
            vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_key))
            ret, frame = vid.read()
            frame = pre_process_frame(frame)
            mean_image += frame
        n += 1
    mean_image /= len(locs[0])
    
    ax.imshow(mean_image)#, aspect='auto')
    title_str = 'cluster %d, %d frames' % (curr_cluster, len(locs[0]))
    ax.set_title(title_str)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

# show some sample of a cluster
# set up plotting
subplot_rows = 5
subplot_cols = 5

for i in range(0,5):
    subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(25,12))
    subplot_fig.subplots_adjust(hspace=0.0,wspace=0.0) 
    curr_cluster = sorted_labels[i]
    print('extracting from cluster', str(curr_cluster))
    locs = np.where(kMeansMiniBatchLabels == curr_cluster)
    rand_locs = np.random.permutation(locs[0])
    for j in range(subplot_cols*subplot_rows):
        ax = plt.subplot(subplot_rows, subplot_cols, j+1)
        loc = rand_locs[j]
        key = test_frame_names[loc]
        frame_key = key.split('frame')[-1]
        file_key = key.split('_frame')[0]
        vid = cv2.VideoCapture(all_files[file_key])
        vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_key))
        ret, frame = vid.read()
        frame = pre_process_frame(frame)
        ax.imshow(frame)
        if j == 0:
            title = 'cluster %d, %d clusters' % (curr_cluster, N_CLUSTER)
            ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
            
plt.show()


#plt.close('all')
#
## break and no break color maps
#cmap_break = colormap.get_cmap('winter')
#cmap_nobreak = colormap.get_cmap('autumn')
#break_colors = []
#nobreak_colors = []
#for c in np.arange(1,N_CLUSTER+1):
#    pull_color = float(c/N_CLUSTER)
#    break_colors.append(cmap_break(pull_color))
#    nobreak_colors.append(cmap_nobreak(pull_color))
#    
#
#fig2D_spec = plt.figure('2D_spec')
#fig2D_kMeans = plt.figure('2D_kMeans')
#fig3D_spec = plt.figure('3D_spec')
#ax_spec = fig3D_spec.add_subplot(111, projection='3d')
#fig3D_kMeans = plt.figure('3D_kMeans')
#ax_kMeans = fig3D_kMeans.add_subplot(111, projection='3d')
#
#my_clusters = []
#
#for cluster in np.arange(N_CLUSTER):
#    my_clusters.append('cluster ' + str(cluster))
#    # plot spectral clusters with o markers for no break and * for break
#    rows_spec = [d == u_labels_spec[cluster] for d in specLabels]
#    x = data_3d[rows_spec, 0]
#    y = data_3d[rows_spec, 1]
#    z = data_3d[rows_spec, 2]
#    c = classes[rows_spec]
#    x_nobreak, y_nobreak, z_nobreak = [], [], []
#    x_break, y_break, z_break = [], [], []
#    for (i,x,y,z) in zip(c,x,y,z):
#        if classes[i] == 1:
#            x_nobreak.append(x)
#            y_nobreak.append(y)
#            z_nobreak.append(z)
#        elif classes[i] == 0:
#            x_break.append(x)
#            y_break.append(y)
#            z_break.append(z)
#    plt.figure('2D_spec')
#    plt.scatter(x_nobreak, y_nobreak, 
#                marker='o', color=nobreak_colors[cluster])
#    plt.scatter(x_break, y_break, 
#                marker='*', color=break_colors[cluster])
#    
#    plt.figure('3D_spec')
#    ax_spec.scatter(x_nobreak, y_nobreak, z_nobreak, 
#                    marker='o', color=nobreak_colors[cluster])
#    ax_spec.scatter(x_break, y_break, z_break, 
#                    marker='*',color=break_colors[cluster])
#        
#    # plot k means clusters
#    rows_kMeans = [d == u_labels_kMeans[cluster] for d in kMeansLabels]
#    x = data_3d[rows_kMeans, 0]
#    y = data_3d[rows_kMeans, 1]
#    z = data_3d[rows_kMeans, 2]
#    c = classes[rows_kMeans]
#    x_nobreak, y_nobreak, z_nobreak = [], [], []
#    x_break, y_break, z_break = [], [], []
#    for (i,x,y,z) in zip(c,x,y,z):
#        if classes[i] == 1:
#            x_nobreak.append(x)
#            y_nobreak.append(y)
#            z_nobreak.append(z)
#        elif classes[i] == 0:
#            x_break.append(x)
#            y_break.append(y)
#            z_break.append(z)
#    plt.figure('2D_kMeans')
#    plt.scatter(x_nobreak, y_nobreak, 
#                marker='o', color=nobreak_colors[cluster])
#    plt.scatter(x_break, y_break,
#                marker='*', color=break_colors[cluster])
#    
#    plt.figure('3D_kMeans')
#    ax_kMeans.scatter(x_nobreak, y_nobreak, z_nobreak, 
#                      marker='o', color=nobreak_colors[cluster])
#    ax_kMeans.scatter(x_break, y_break, z_break, 
#                      marker='*', color=break_colors[cluster])
#
#
#plt.figure('2D_spec')
#plt.legend(my_clusters, fontsize=18)
#plt.xlabel('dimension 1', fontsize=18)
#plt.ylabel('dimension 2', fontsize=18)
#plt.title('2D spectral clustering', fontsize=22)
#
#plt.figure('2D_kMeans')
#plt.legend(my_clusters, fontsize=18)
#plt.xlabel('dimension 1', fontsize=18)
#plt.ylabel('dimension 2', fontsize=18)
#plt.title('2D k-means clustering', fontsize=22)
#
#plt.figure('3D_spec')
#ax_spec.set_xlabel('dimension 1')
#ax_spec.set_ylabel('dimension 2')
#ax_spec.set_ylabel('dimension 3')
#ax_spec.set_title('2D spectral clustering')
#
#plt.figure('3D_kMeans')
#ax_kMeans.set_xlabel('dimension 1')
#ax_kMeans.set_ylabel('dimension 2')
#ax_kMeans.set_ylabel('dimension 3')
#ax_kMeans.set_title('2D k-means clustering')
#
