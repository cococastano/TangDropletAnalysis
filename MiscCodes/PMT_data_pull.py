import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter
import tkinter.filedialog as filedialog
from scipy import signal
from detect_peaks import detect_peaks
from scipy.interpolate import interp1d

"""
Nicolas Castano
Last revision: 6/17/18

Pulling time data from long data file (~1.3 GB/text file) and cleaning up 
the signal while preserving all peaks to get a count of peaks

"""

################################ USEFUL METHODS ##############################
def chunks(l, n):
    """yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def downsample_for_peaks(sig, mph=None, mpd=1, interp_type='linear', 
                         mean_mar=0.05, ds_factor = 0.1):
    """return downsampled data that preserves peaks"""
    npc = 1
    if interp_type is 'linear': npc = 1
    if interp_type is 'quadratic': npc = 2
    if interp_type is 'cubic': npc = 3
    
    # first get all peaks and determine mean of portion of peaks for mph
    peak_i_init = detect_peaks(sig, mph=None, mpd=mpd, show=False)
    peaks_init = [sig[pk_i] for pk_i in peak_i_init]
    lower_mar = peaks_init[:100]#int(len(sig)*mean_mar)]
    
    if mph == None: mph = np.mean(lower_mar) + 0.007
    
    peak_i = detect_peaks(sig, mph=mph, mpd=mpd, show=False)
    fit_time = np.linspace(part_time[j][0], part_time[j][-1], 
                           int(len(sig)*ds_factor))
    
    if len(peak_i) <= npc:  # for cases when no peaks are found
        peaks = mph * np.ones_like(fit_time)
        if len(peak_i) == 1:
            peaks[int(peak_i[0]*ds_factor)] = sig[peak_i[0]]

    else:  
        # else downsample to upper envelope (peaks)
        peaks = mph * np.ones_like(fit_time)
        for pk_i in peak_i:
            peaks[int(pk_i*ds_factor)] = sig[pk_i]
#        peaks = [sig[k] for k in peak_i]
#        peaks_time = [part_time[j][k] for k in peak_i]
#        # fit to peaks
#        u_env = interp1d(peaks_time, peaks, kind=interp_type, 
#                         bounds_error=False, fill_value=mph)
#        peaks = u_env(fit_time)
    return (fit_time, peaks)

################################# PULL DATA ###################################
##################### CLEAN SIGNAL WHILE PRESERVING PEAKS #####################

# see Allergies/Codes/PMT_data_pull_v0.py for the original version

# plot PMT signal for flow cytometer; store values by step
plt.close('all')
line_step = 2

# get directory with data files
root = tkinter.Tk()
root.dir_name = filedialog.askdirectory()
root.destroy()  
file_list = [x[2] for x in os.walk(root.dir_name)]

# store all names of the tests
test_names = [name.split('.txt')[0] for name in file_list[0] \
              if '.txt' in name]
# each item is a tuple with time and signal and store means too
signal_means = {key.split('.txt')[0]:0 for key in file_list[0] \
                if '.txt' in key}

# downsample by two passes of peak detection
# vector to collect first, second, and third downsample for repartitioning
# along with their mean values
downsample_data = {key.split('.txt')[0]:([],[]) for key in file_list[0] \
                   if '.txt' in key}
peaks = {key.split('.txt')[0]:([],[]) for key in file_list[0] \
         if '.txt' in key}
ds_lower_mean = {key:0 for key in list(signal_means.keys())}

# number of data points in partition and storing peak counts
num_in_parts = 80000  
my_peak_count = {key:0 for key in list(signal_means.keys())}

# set up plotting
i = 0  # index for subplotting
subplot_rows = 2
subplot_cols = len(test_names)
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(25,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2) 

for test in test_names:
    i += 1
    # temporary container for each signal ([time],[sig])
    PMT_signal = ([],[])
    file_name = root.dir_name + os.sep + test + '.txt'
    print('pulling from file: ', test + '.txt')
    with open(file_name, 'r') as f:
        sub_count = 0
        for count, line in enumerate(f, start=1):
            if count % line_step == 0:
                data = line.split()
                PMT_signal[0].append(float(data[0]))
                PMT_signal[1].append(float(data[1]))
                sub_count += 1
                signal_means[test] += float(data[1])
            if count % 5000000 == 0: print('reading line',count)
        signal_means[test] /= sub_count
           
    # make partitions of data
    print('partitioning ',test)
    part_PMT_signal = list(chunks(PMT_signal[1],num_in_parts))
    part_time = list(chunks(PMT_signal[0],num_in_parts))
    len_part = len(part_PMT_signal)
    ax1 = plt.subplot(subplot_rows, subplot_cols, i)
    ax1.plot(PMT_signal[0],PMT_signal[1],'k',linewidth=2)
    for j, sig in enumerate(part_PMT_signal):
        #  let the first mimimum peak height be determined in the method
        mph = None
        
        if j % 300 == 0: 
            print('downsampling by peaks for partition %d of %d' %(j,len_part))
        # call downsample by peaks method and store data for next pass
        ds_time,ds_data = downsample_for_peaks(sig, mph=mph, mpd=20, 
                                               interp_type='linear',
                                               mean_mar = 0.05)
        downsample_data[test][0].extend(ds_time)
        downsample_data[test][1].extend(ds_data)
        
        ax1.plot(ds_time,ds_data,linewidth=2)
    ax1.set_title(test)
    ax1.set_ylim((0,0.4))
    # this will be the mean of some lower margin of data   
    ds_lower_mean[test] = np.mean(sorted(downsample_data[test][1])[:10])


#    ax2 = plt.subplot(subplot_rows, subplot_cols, i+len(test_names))
#    ax2.plot(PMT_signal[0],PMT_signal[1],'k',linewidth=2,label='raw data')
#    ax2.plot(downsample_data[test][0],downsample_data[test][1],linewidth=2,
#             label='downsampled data')
#    ax2.set_ylim((0,0.4))

    # find peaks on this downsampled signal
    peak_i = detect_peaks(downsample_data[test][1], mph=ds_lower_mean[test]+0.003, 
                          mpd=1, show=False)  
    pk = [downsample_data[test][1][pk_i] for pk_i in peak_i]
    t = [downsample_data[test][0][pk_i] for pk_i in peak_i]
    peaks[test][1].extend(pk)
    peaks[test][0].extend(t)
    ax3 = plt.subplot(subplot_rows, subplot_cols, i+len(test_names))
    ax3.plot(downsample_data[test][0],downsample_data[test][1],'k',
             linewidth=2,label='downsampled data')
    ax3.plot(peaks[test][0],peaks[test][1],'o',linewidth=2,
             label='counted peaks')
    ax3_title = '%d peaks counted' % (len(peak_i))
    ax3.set_title(ax3_title)
    
    ax3.set_ylim((0,0.4))
    ax3.legend()
                
plt.show()

# plot distribution of peak intensities
num_bins = 50
bins = np.linspace(0, 0.4, num_bins)
subplot_rows = 1
subplot_cols = len(test_names)
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(25,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2) 
for i, test in enumerate(downsample_data, start=1):
    ax = plt.subplot(subplot_rows, subplot_cols, i)
    plt.hist(peaks[test][1], bins, alpha=1)
    plt.xlabel('Intensity [-]')
    plt.ylabel('Number of data points [-]')
    plt.title(test)
    plt.grid()
plt.show()