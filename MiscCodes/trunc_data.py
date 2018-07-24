import numpy as np
import matplotlib.pyplot as plt

# truncate data files by a time

file_in = 'E:/BAT/071818_flow_cyto_v6_my_blood/pos_stim_3_Av_1pt5_0pt6_20000sps.txt'
file_out = 'E:/BAT/071818_flow_cyto_v6_my_blood/pos_stim_3_Av_1pt5_0pt6_20000sps_trunc.txt'

with open(file_in,'r') as infile, open(file_out,'w') as outfile:
    for i,line in enumerate(infile):
        time = float(line.split()[0])
        if time < 530:
            outfile.write(line)
            if i % 5000000 == 0: print('writing line ',i)
        else:
            if i % 10000000 == 0: print('omitting line ',i)
    
        
# and remove erroneous spikes (i.e. from ambient light)
file_in = 'E:/BAT/071818_flow_cyto_v6_my_blood/pos_stim_3_Av_1pt5_0pt6_20000sps.txt'        
file_out = 'E:/BAT/071818_flow_cyto_v6_my_blood/pos_stim_3_Av_1pt5_0pt6_20000sps_trunc.txt'
trunc_t_1 = 0
trunc_t_2 = 4.6
with open(file_in,'r') as infile, open(file_out,'w') as outfile:
    for i,line in enumerate(infile):
        time = float(line.split()[0])
        val = float(line.split()[1])
        if not time > trunc_t_1:
            outfile.write(line)
            if i % 500000 == 0: print('writing line ',i)
        elif time >= trunc_t_1 and time <= trunc_t_2:
            if i % 100000 == 0: print('omitting line ',i)
        elif time > trunc_t_2:
            t = time - (trunc_t_2 - trunc_t_1)
            my_line = '%.6f\t%.6f\n' % (t,val)
            outfile.write(my_line)
            if i % 100000 == 0: print('writing adjusted line ',i)    
#        elif time >= 2823.884 and time <= 2890.3:
##            if i % 100000 == 0: print('omitting line ',i)
#        elif time > 2890.3:
#            t = time - (2890.3 - 2823.884)
#            my_line = '%.6f\t%.6f\n' % (t,val)
#            outfile.write(my_line)
#            if i % 500000 == 0: print('writing line ',i)


# remove gaps in data
# when the PMT labview program is stopped and started the time keeps counting
# so the next data point will be seperated by more than the sampling period
file_in = 'E:/BAT/062818_flow_cyto_v4_bryan_blood/pos_stim_1_Av_350mbar_0pt65_10000sps.txt'         
file_out = 'E:/BAT/062818_flow_cyto_v4_bryan_blood/pos_stim_1_Av_350mbar_0pt65_10000sps_adj.txt'
with open(file_in,'r') as infile, open(file_out,'w') as outfile:
    thresh_t = 1.0
    prev_t = 0.0
    shift_t = 0.0
    for i, line in enumerate(infile):
        time = float(line.split()[0])
        time = time - shift_t
        val = float(line.split()[1])
        # if the delta t is more than a second, shift data over
        if time - prev_t > thresh_t:
            shift_t = shift_t + (time - prev_t)
            print('adjusting for gap in data... new shift time: ', shift_t)
        prev_t = time
        my_line = '%.6f\t%.6f\n' % (time,val)
        outfile.write(my_line)
        if i % 500000 == 0: print('writing line ',i)


# and combine data files
files_in = ['E:/BAT/062518_flow_cyto_v2_bryan_blood/pos_stim_2_Av_0pt1_0pt3_10000sps.txt',
            'E:/BAT/062518_flow_cyto_v2_bryan_blood/pos_stim_2b_Av_0pt1_0pt3_10000sps_trunc_adj.txt']
file_out = 'E:/BAT/062518_flow_cyto_v2_bryan_blood/pos_stim_2_Av_0pt1_0pt3_10000sps_comb.txt'      
with open(file_out,'w') as outfile:
    with open(files_in[0],'r') as infile_1:
        for i,line in enumerate(infile_1):
            outfile.write(line)
            time = float(line.split()[0])
            if i % 500000 == 0: print('writing line ',i,' from ', files_in[0])
    
    end_time = time
    with open(files_in[1],'r') as infile_2:
        for i,line in enumerate(infile_2):
            time = float(line.split()[0])
            val = float(line.split()[1])
            t = time + end_time
            my_line = '%.6f\t%.6f\n' % (t,val)
            outfile.write(my_line)
            if i % 500000 == 0: print('writing line ',i,' from ', files_in[1])
            
            
############################ PLOT CHANGES YOU MADE ############################
file_in = 'E:/BAT/062818_flow_cyto_v4_bryan_blood/neg_stim_1_Av_100mbar_0pt3_10000sps.txt'         
file_out = 'E:/BAT/062818_flow_cyto_v4_bryan_blood/neg_stim_1_Av_100mbar_0pt3_10000sps_adj.txt'
my_files = {file_in, file_out}
# each item is a tuple with time and signal
PMT_signals = {key.split('.txt')[0]:([],[]) for key in my_files}

for file in my_files:
    file_name = file
    key = file.split('.txt')[0]
    with open(file_name, 'r') as f:
        sub_count = 0
        for count, line in enumerate(f, start=1):
            if count % 2 == 0:
                data = line.split()
                PMT_signals[key][0].append(float(data[0]))
                PMT_signals[key][1].append(float(data[1]))
                sub_count += 1
            if count % 5000000 == 0: print('reading line',count)

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
    
plt.rcParams['agg.path.chunksize'] = 10000
plt.show()