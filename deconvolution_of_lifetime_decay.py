# -*- coding: utf-8 -*-
"""
Monday Jul 12 2021
Fribourg, Switzerland
@author: Mariano Barella

Deconvolution of lifetime histogram decay with IRF

This code process lifetime histograms obtained with the LabView software 
in order to remove the instrument response function (IRF) from the acquired data.
The IRF distorts the true lifetime of the detected molecule due to timing 
accuracy, timing error (phase noise) and not-so-sharp excitation source.

The code uses another code where auxiliary functions are placed.
Another code for processing the IRF is given. 
It's called "IRF_lifetime_decay.py" (*).

Inputs: 
    - The IRF of the laser you used must be measured.
    - Place the IRF file in a subfolder, inside the data's folder.
    - Name and filepath of the IRF.
    - Folder of the data. 
    - Window, degee and number of repetitions for the smoothing of the 
    histogram. Only for presentation puposes.
    - Extension of the lifetime hisogram files.

Analysis pipeline:
    1) The program looks for data with the selected extension.
    2) It replaces all commas present in the files with dots.
    2) Then, loads the IRF.
    3) Then, opens each file and process it (typical these files contain 
    several histograms appended as were extracted using the LabView software).
    4) Data processing:
        a) crop the tail of the histogram (usually has higher counts than expected)
        b) smoothing
        c) find maximum
        d) crop data from maximum to last point
        e) deconvolve (**)
        f) grab deconvolved lifetime
        g) plot raw data, smoothed data and deconvolved data
        h) save data

** Deconvolution is performed as follows:
    - A single decay model is assumed (exponential decay)
    - With the obtained IRF (code is also provided)
    - Convolute model with IRF
    - Fit the convoluted decay (model+IRF) with the acquired data
    - This fitting step makes the deconvolution process an iterative process.
    - Initial parameteres are given and minimization of the residuals
    (convolved model minus data) gives the best lifetime that fits the data.
    - As this code performs a fitting of the acquired data a figure of merit 
    for this process, a goodess of the fit, R-squared is given.
    
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.signal as sig
import scipy.optimize as opt
# from tkinter import Tk, filedialog
from auxiliary_functions_for_deconvolution import exp_decay, conv, s_squared, \
                                                    calc_r2, replace_comma_dot

plt.ioff()
plt.close("all")

# INPUT extension of files to replace comma by dot
extension = '\.dat'

# INPUT parameter definition for smoothing
window = 13
deg = 1
repetitions = 1

##############################################################################
# INPUT data's folder
folder = 'C:\\datos_mariano\\posdoc\\MoS2\\lifetime_measurement\\20210707_flakes'
folder = 'C:\\datos_mariano\\posdoc\\MoS2\\lifetime_measurement\\20210709_glass'

# replace comma by dot in the specified folder
replace_comma_dot(folder, extension)
# make a list of the files to be analized
list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search(extension, f)]
list_of_files.sort()
L = len(list_of_files)

# create folders to save data
save_folder = os.path.join(folder, 'saved_data')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_folder_figures = os.path.join(folder, 'saved_data\\figures')
if not os.path.exists(save_folder_figures):
    os.makedirs(save_folder_figures)

# load IRF in green for deconvolution
irf_filename = 'irf_green_smoothed_normalized.dat'
irf_subfolder = 'IRF\\saved_data'
irf_folder = os.path.join(folder, irf_subfolder)
irf_filepath = os.path.join(irf_folder, irf_filename)
irf_data = np.loadtxt(irf_filepath, skiprows = 1)
time_irf = irf_data[:,0]
counts_irf = irf_data[:,1]

# iterate over files
for i in range(L):
    # load data
    filename = list_of_files[i]
    print(filename, '\n')
    arrival_time_filepath = os.path.join(folder, filename)
    data = np.loadtxt(arrival_time_filepath, skiprows = 1)
    # removing flat tail + accumulation of counts at the end of the histogram 
    # (artifact of pre-analysis)
    time = data[:-17,0] 
    arrival_times = data[:-17,1:] 
    number_of_hists = np.shape(arrival_times)[1]
    
    # allocation
    lifetime_array = np.zeros(number_of_hists)
    R2_array = np.zeros(number_of_hists)
    col_array = np.zeros(number_of_hists)
    # deconvolve each histogram
    for j in range(number_of_hists):
        # grab signal
        counts_signal = arrival_times[:,j]
        # smooth to find maximum and crop data
        counts_signal_smooth = sig.savgol_filter(counts_signal, 
                                          window, deg, axis = 0, 
                                          mode='constant')
        # find maximum
        index_max = np.argmax(counts_signal_smooth)
        # keep only the decay
        time_decay = time[index_max:]
        counts_decay = counts_signal[index_max:]
        N_decay = len(counts_decay)
        # smooth only the decay
        counts_decay_smooth = sig.savgol_filter(counts_decay, 
                                          window, deg, axis = 0, 
                                          mode='interp')
        for l in range(repetitions - 1):
            counts_decay_smooth = sig.savgol_filter(counts_decay_smooth, 
                                              window, deg, axis = 0, 
                                              mode='interp')
        
        print('\n------- Finding lifetime of histogram number %03d\n' % j)
        # minimize difference between convolved signal and data
        # guess initial parameters
        init_params = [2, 10, 0] # tau, amplitude, offset
        # prepare function to store points the method pass through
        road_to_convergence = list()
        road_to_convergence.append(init_params)
        def callback_fun(X):
            road_to_convergence.append(list(X))
            return 
        # define bounds of the minimization problem (any bounded method)
        bnds = opt.Bounds([0.1, 0, -100], [10, 5000, 500]) # [lower bound array], [upper bound array]
        # now minimize
        out = opt.minimize(s_squared, 
                            init_params, 
                            args = (time, counts_decay, counts_irf, N_decay),
                            method = 'Nelder-Mead',
                            bounds = bnds,
                            callback = callback_fun,
                            options = {'maxiter':2000,
                                      'maxfev':2000,
                                       # 'disp':True})
                                       'disp':False})
                            # other options are set to default and 
                            # are suitable for this problem
        # print(out)
        # grab fitted parameters
        best_params = out.x
        # recover convoluted signal to check goodess of the fit and plot
        fitted_decay = exp_decay(best_params, time)
        fitted_conv = conv(fitted_decay, counts_irf, 'full')
        index_max_fitted_conv = np.argmax(fitted_conv)
        fitted_conv_decay = fitted_conv[index_max_fitted_conv:index_max_fitted_conv + N_decay]
        # figure of merit of the fit
        R2 = calc_r2(counts_decay, fitted_conv_decay)
        print('R-squared %.2f' % R2)
        print('Lifetime %.2f ns' % best_params[0])
        print('Amplitude %.2f' % best_params[1])
        print('Offset %.2f' % best_params[2])
        # send parameters out of loop to save later
        lifetime_array[j] = best_params[0]
        R2_array[j] = R2
        col_array[j] = number_of_hists - j
        # plot
        if True:
            plt.figure()
            plt.plot(time, counts_signal, '.', color='C0', label='Raw data')
            plt.plot(time_decay, counts_decay_smooth, '--', color='k', label='Decay smoothed')
            plt.plot(time_decay, fitted_conv_decay, '-', color='C3', label='Best convoluted decay (fit)')
            plt.legend(loc=1)
            plt.xlabel('Time (ns)')
            plt.ylabel('Counts')
            plt.xlim([-0.2, 11])
            ax = plt.gca()
            title = '%s_counts%03d' % (filename, j)
            ax.set_title(title)
            figure_name = '%s_counts%03d' % (filename, j)
            figure_path = os.path.join(save_folder_figures, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            ax.set_yscale('log')
            figure_name = 'log_%s_counts%03d' % (filename, j)
            figure_path = os.path.join(save_folder_figures, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')        
            plt.close()
         
    # save data     
    data_to_save = np.asarray([col_array, lifetime_array, R2_array]).T
    new_filename = 'lifetime_of_deconvoluted_data_%s' % ('_'.join(filename.split('_')[3:]))
    new_filepath = os.path.join(save_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%.2f', header = 'Column Lifetime(ns) R-squared')

# plt.close('all')
# plt.show()

