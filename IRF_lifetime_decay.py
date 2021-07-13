# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:21:17 2021

@author: Mariano Barella

"""

import numpy as np
import matplotlib.pyplot as plt
import os
# import re
import scipy.signal as sig
# from tkinter import Tk, filedialog

plt.ioff()
plt.close("all")


##############################################################################
# load data
# folder = 'C:\\datos_mariano\\posdoc\\MoS2\\lifetime_measurement\\20210707_flakes\\IRF'
folder = 'C:\\datos_mariano\\posdoc\\MoS2\\lifetime_measurement\\20210709_glass\\IRF'

# irf_green_file = 'IRF_green_10uW.dat'
irf_green_file = 'IRF_green_4uW.dat'
irf_green_filepath = os.path.join(folder, irf_green_file)
irf_green_data = np.loadtxt(irf_green_filepath, skiprows = 1)
time_green = irf_green_data[:,0]
counts_green = irf_green_data[:,1]

# irf_red_file = 'IRF_red_2uW.dat'
irf_red_file = 'IRF_red_0.7uW.dat'
irf_red_filepath = os.path.join(folder, irf_red_file)
irf_red_data = np.loadtxt(irf_red_filepath, skiprows = 1)
time_red = irf_red_data[:,0]
counts_red = irf_red_data[:,1]

# create folder to save data
save_folder = os.path.join(folder, 'saved_data')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# process data
# smoothing
window = 5
deg = 2
repetitions = 2
mode = 'constant'

# green laser at green channel
counts_green_smooth = sig.savgol_filter(counts_green, 
                                        window, deg, axis = 0, 
                                        mode=mode)

for i in range(repetitions - 1):
    counts_green_smooth = sig.savgol_filter(counts_green_smooth, 
                                            window, deg, axis = 0, 
                                            mode=mode)

counts_green_smooth[14] = 30

# red laser at red channel
counts_red_smooth = sig.savgol_filter(counts_red, 
                                        window, deg, axis = 0, 
                                        mode=mode)

for i in range(repetitions - 1):
    counts_red_smooth = sig.savgol_filter(counts_red_smooth, 
                                            window, deg, axis = 0, 
                                            mode=mode)
counts_red_smooth[13] = 700
counts_red_smooth[14] = 900

# normalization
counts_green_smooth_norm = counts_green_smooth/max(counts_green_smooth)
counts_red_smooth_norm = counts_red_smooth/max(counts_red_smooth)

# plot
plt.figure(0)
plt.plot(time_green, counts_green, '.', color='C2')
plt.plot(time_green, counts_green_smooth, 'k')
# plt.legend()
plt.xlabel('Time (ns)')
plt.ylabel('Counts')
ax = plt.gca()
# ax.set_title('Number of locs per pick vs time. Bin size %d min' % bin_size_minutes)
figure_name = 'irf_green'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
plt.show()

plt.figure(0)
ax.set_yscale('log')
figure_name = 'irf_green_log'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')


plt.figure(1)
plt.plot(time_red, counts_red, '.', color='C3')
plt.plot(time_red, counts_red_smooth, 'k')
# plt.legend()
plt.xlabel('Time (ns)')
plt.ylabel('Counts')
ax = plt.gca()
# ax.set_title('Number of locs per pick vs time. Bin size %d min' % bin_size_minutes)
figure_name = 'irf_red'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')

plt.figure(1)
ax.set_yscale('log')
figure_name = 'irf_red_log'
figure_path = os.path.join(save_folder, '%s.png' % figure_name)
plt.savefig(figure_path, dpi = 300, bbox_inches='tight')

plt.close('all')
# plt.show()

# save data
data_to_save = np.asarray([time_green, counts_green_smooth_norm]).T
new_filename = 'irf_green_smoothed_normalized.dat'
new_filepath = os.path.join(save_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%.3e')

data_to_save = np.asarray([time_red, counts_red_smooth_norm]).T
new_filename = 'irf_red_smoothed_normalized.dat'
new_filepath = os.path.join(save_folder, new_filename)
np.savetxt(new_filepath, data_to_save, fmt='%.3e')


