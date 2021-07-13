# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:21:17 2021

@author: Mariano Barella

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.signal as sig
import scipy.optimize as opt
# from tkinter import Tk, filedialog

plt.ioff()
plt.close("all")

# model decay as mono-exponential
def exp_decay(params, time):
    tau, amplitude, offset = params
    y = amplitude*np.exp(-time/tau) + offset
    return y

# convolution of the exponential decay with the IRF
def conv(decay_model, irf, mode):
    c = np.convolve(decay_model, irf, mode=mode)
    return c

# Function to minimize: S-squared. We will find the best parameters (tau, 
# amplitude, offset) that minimize the difference between the acquired decay 
# histogram of arrival times and the convoluted one
def s_squared(params, time, decay_data, irf, number_of_points):
    func = exp_decay(params, time)
    convolved_model = conv(func, irf, 'full')
    index_max_conv = np.argmax(convolved_model)
    convolved_decay = convolved_model[index_max_conv:index_max_conv + number_of_points]
    if False:
        # DO NOT set ti True unless debugging
        plt.figure()
        plt.plot(func, 'o', color='C0', label='Model')
        plt.plot(decay_data, 's', color='C1', label='Raw')
        plt.plot(convolved_decay, '-', color='C3', label='Conv decay')
        plt.plot(convolved_model, '^', color='C4', label='Conv full')
        plt.legend()
        plt.xlabel('Array index')
        plt.ylabel('Counts')
        plt.show()
    s2 = 0
    residuals = decay_data - convolved_decay
    s2_residuals = residuals**2
    s2 = np.sum(s2_residuals)
    return s2

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    # sum of squares of residuals
    ssres = ((observed - fitted)**2).sum()
    # total sum of squares
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

# process data
# smoothing
window = 13
deg = 1
repetitions = 1

##############################################################################
# make a list of files
folder = 'C:\\datos_mariano\\posdoc\\MoS2\\lifetime_measurement\\20210707_flakes'
folder = 'C:\\datos_mariano\\posdoc\\MoS2\\lifetime_measurement\\20210709_glass'
list_of_files = os.listdir(folder)
list_of_files = [f for f in list_of_files if re.search('\.dat',f)]
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
        
        # print('\n------- Finding lifetime of histogram number %03d\n' % j)
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
        # print('R-squared %.2f' % R2)
        # print('Lifetime %.2f ns' % best_params[0])
        # print('Amplitude %.2f' % best_params[1])
        # print('Offset %.2f' % best_params[2])
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

