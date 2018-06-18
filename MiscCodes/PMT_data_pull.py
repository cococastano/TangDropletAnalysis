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

Pulling time data from long data file (~1.3 GB/text file)

"""


def chunks(l, n):
    """yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

plt.close('all')

# plot PMT signal for flow cytometer; store values by step
line_step = 10

# get directory with data files
root = tkinter.Tk()
root.dir_name = filedialog.askdirectory()
root.destroy()  

file_list = [x[2] for x in os.walk(root.dir_name)]
# each item is a tuple with time and signal
PMT_signals = {key.split('.txt')[0]:([],[]) for key in file_list[0] \
               if '.txt' in key}
signal_means = {key:0 for key in list(PMT_signals.keys())}

for subdir, _, files in os.walk(root.dir_name):
    for file in files:
        if '.txt' in file:
            file_name = subdir + os.sep + file
            key = file.split('.txt')[0]
            print('pulling from file: ', file)
            with open(file_name, 'r') as f:
                sub_count = 0
                for count, line in enumerate(f, start=1):
                    if count % line_step == 0:
                        data = line.split()
                        PMT_signals[key][0].append(float(data[0]))
                        PMT_signals[key][1].append(float(data[1]))
                        sub_count += 1
                        signal_means[key] += float(data[1])
                    if count % 5000000 == 0: print('reading line',count)
                signal_means[key] /= sub_count

# np.save('PMT_signals.npy', PMT_signals)

  
    
# whole signal
subplot_cols = 1
subplot_rows = int(np.ceil(len(PMT_signals)/subplot_cols))
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(15,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2)                 
for i, key in enumerate(PMT_signals, start=1):
    print('plotting:', key)
    plt.subplot(subplot_rows, subplot_cols, i)
    plt.plot(PMT_signals[key][0],PMT_signals[key][1],'k',linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Fluorescent signal')
    plt.title(key)
    plt.ylim((0,0.4))
    plt.grid()

plt.show()

# histogram plot
num_bins = 150
bins = np.linspace(0, 0.4, num_bins)
subplot_cols = 1
subplot_rows = int(np.ceil(len(PMT_signals)/subplot_cols))
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(15,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2) 
default_color = (1.0, 0.4980392156862745, 0.054901960784313725, 1)                
for i, key in enumerate(PMT_signals, start=1):
    print('plotting:', key)
    ax = plt.subplot(subplot_rows, subplot_cols, i)
    plt.hist(PMT_signals[key], bins, alpha=1)
    plt.yscale('log', nonposy='clip')
    for obj in ax.get_children():
        try:
            if obj.get_facecolor() != default_color:
                obj.set_facecolor((1,1,1,1))  
            elif obj.get_facecolor() == default_color:
                obj.set_facecolor((0,0.5,0.5,1)) 
        except:
            pass
    plt.xlabel('Intensity [-]')
    plt.ylabel('Number of data points [-]')
    plt.title(key)
    plt.grid()
plt.show()
    

# plot variance of each partition in subplot
subplot_rows = 2
subplot_cols = len(PMT_signals)
subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(25,12))
subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2)    
num_in_parts = 10000  # number of data points in partition     
dummy = list(PMT_signals.keys())[1]
my_peak_count = {key:0 for key in list(PMT_signals.keys())}
for i, key in enumerate(PMT_signals, start=1):
    # make partitions of data
    print('partitioning:',key)
    part_PMT_signal = list(chunks(PMT_signals[key][1],num_in_parts))
    part_time = list(chunks(PMT_signals[key][0],num_in_parts))
    len_part = len(part_PMT_signal)
    ax1 = plt.subplot(subplot_rows, subplot_cols, i)
    ax2 = plt.subplot(subplot_rows, subplot_cols, i+len(PMT_signals))
    num_peaks = 0
    part_var = 0
    for j, sig in enumerate(part_PMT_signal):
#        if j % 1000 == 0: print('filtering partition %d of %d' %(j,len_part))
#        filtered_data = signal.medfilt(sig,31)
        ax1.plot(part_time[j],sig,'k',linewidth=2)
        
        
        # take hilbert transform
#        hilb_trans_sig = signal.hilbert(sig,N=100)
#        my_hilbs[key].extend(hilb_trans_sig)

#        ax2 = ax1.twinx()
#        ax2.plot(hilb_trans_sig,'b',linewidth=2)
        
#        fig, ax1 = plt.subplots()
#        t = np.arange(0.01, 10.0, 0.01)
#        s1 = np.exp(t)
#        ax1.plot(t, s1, 'b-')
#        ax1.set_xlabel('time (s)')
#        # Make the y-axis label, ticks and tick labels match the line color.
#        ax1.set_ylabel('exp', color='b')
#        ax1.tick_params('y', colors='b')
#        
#        ax2 = ax1.twinx()
#        s2 = np.sin(2 * np.pi * t)
#        ax2.plot(t, s2, 'r.')
#        ax2.set_ylabel('sin', color='r')
#        ax2.tick_params('y', colors='r')
#        
#        fig.tight_layout()
        
        
        # get peaks
        # set minimum peak height based on mean of signal
        mph = signal_means[key] + 0.01
        if j % 300 == 0: 
            print('finding peaks for partition %d of %d' %(j,len_part))
        peak_i = detect_peaks(sig, mph=mph, mpd=20, show=False)
        if j % 300 == 0: print('found %d peaks in partition' % (len(peak_i)))
        if len(peak_i) <= 5:
            peaks = mph*np.ones_like(sig)
            peaks_time = np.linspace(part_time[j][0], 
                                     part_time[j][-1], len(sig))
            ax1.plot(peaks_time,peaks,linewidth=2)
        else:
            peaks = [sig[k] for k in peak_i]
            peaks_time = [part_time[j][k] for k in peak_i]
        
            u_env = interp1d(peaks_time, peaks, kind='cubic', 
                             bounds_error=False, fill_value=0.0)
            peaks = u_env(peaks_time)
            num_peaks += len(peak_i)
            
        ax2.plot(j,np.var(peaks),'ko')
        
        if part_var < np.var(peaks): 
            part_var = np.var(peaks)
            ax1.plot(peaks_time,peaks,'k*',linewidth=2)
            print(part_var)
#            8.170008676748596e-07
#            1.0288007340720225e-06
#            1.398725876388892e-06
#            1.5986916074380143e-06
#            1.7561223493749962e-06
#            1.7690038723966954e-06
#            1.827427283446712e-06
            
        ax1.set_ylim((0,0.4))
        
    my_peak_count[key] = num_peaks
    
#    ax1.xlabel('Time [s]')
#    ax1.plt.ylabel('Fluorescent signal')
#    ax1.title(key)
#    ax1.ylim((0,0.4))
#    ax1.grid()
    

plt.show()
   





#subplot_cols = 1
#subplot_rows = int(np.ceil(len(PMT_signals)/subplot_cols))
#subplot_fig, axs = plt.subplots(subplot_rows, subplot_cols,figsize=(15,12))
#subplot_fig.subplots_adjust(hspace=0.5,wspace=0.2)    
#num_in_parts = 10000  # number of data points in partition     
#dummy = list(PMT_signals.keys())[1]
#my_hilbs = {key:[] for key in list(PMT_signals.keys())}
#my_peak_count = {key:0 for key in list(PMT_signals.keys())}
##for i, key in enumerate(PMT_signals, start=1):
#for i, key in enumerate([dummy], start=1):
#    # make partitions of data
#    print('partitioning:',key)
#    part_PMT_signal = list(chunks(PMT_signals[key][1],num_in_parts))
#    part_time = list(chunks(PMT_signals[key][0],num_in_parts))
#    len_part = len(part_PMT_signal)
#    ax1 = plt.subplot(subplot_rows, subplot_cols, i)
#    num_peaks = 0
#    part_var = 0
#    for j, sig in enumerate(part_PMT_signal):
##        if j % 1000 == 0: print('filtering partition %d of %d' %(j,len_part))
##        filtered_data = signal.medfilt(sig,31)
#        ax1.plot(part_time[j],sig,'k',linewidth=2)
#        
#        # take hilbert transform
##        hilb_trans_sig = signal.hilbert(sig,N=100)
##        my_hilbs[key].extend(hilb_trans_sig)
#
##        ax2 = ax1.twinx()
##        ax2.plot(hilb_trans_sig,'b',linewidth=2)
#        
##        fig, ax1 = plt.subplots()
##        t = np.arange(0.01, 10.0, 0.01)
##        s1 = np.exp(t)
##        ax1.plot(t, s1, 'b-')
##        ax1.set_xlabel('time (s)')
##        # Make the y-axis label, ticks and tick labels match the line color.
##        ax1.set_ylabel('exp', color='b')
##        ax1.tick_params('y', colors='b')
##        
##        ax2 = ax1.twinx()
##        s2 = np.sin(2 * np.pi * t)
##        ax2.plot(t, s2, 'r.')
##        ax2.set_ylabel('sin', color='r')
##        ax2.tick_params('y', colors='r')
##        
##        fig.tight_layout()
#        
#        
#        # get peaks
#        if j % 300 == 0: 
#            print('finding peaks for partition %d of %d' %(j,len_part))
#        peak_i = detect_peaks(sig, mph=0.043, mpd=20, show=False)
#        if j % 300 == 0: print('found %d peaks in partition' % (len(peak_i)))
#        if len(peak_i) <= 5:
#            print('no peaks in partition %d' %(j))
#            peaks = 0.043*np.ones_like(sig)
#            peaks_time = np.linspace(part_time[j][0], 
#                                     part_time[j][-1], len(sig))
#            ax1.plot(peaks_time,peaks,linewidth=2)
#        else:
#            peaks = [sig[k] for k in peak_i]
#            peaks_time = [part_time[j][k] for k in peak_i]
#        
#            u_env = interp1d(peaks_time, peaks, kind='cubic', 
#                             bounds_error=False, fill_value=0.0)
#            peaks = u_env(peaks_time)
#            num_peaks += len(peak_i)
#            
#        if part_var < np.var(peaks): 
#            part_var = np.var(peaks)
#            ax1.plot(peaks_time,peaks,'k*',linewidth=2)
#            print(part_var)
##            8.170008676748596e-07
##            1.0288007340720225e-06
##            1.398725876388892e-06
##            1.5986916074380143e-06
##            1.7561223493749962e-06
##            1.7690038723966954e-06
##            1.827427283446712e-06
#        else:
#            ax1.plot(peaks_time,peaks,linewidth=2)
#
#        
#    my_peak_count[key] = num_peaks
#    
#    plt.xlabel('Time [s]')
#    plt.ylabel('Fluorescent signal')
#    plt.title(key)
#    plt.ylim((0,0.4))
#    plt.grid()
#    
#hilb_fig = plt.figure()
##plt.rcParams['agg.path.chunksize'] = 100000
##plt.plot(np.real(my_hilbs[key]),np.imag(my_hilbs[key]))
#envelope = np.abs(my_hilbs[key])
#
#plt.plot(envelope)
#    
#
#plt.show()
#   
