# -*- coding: utf-8 -*-
"""
Monday Jul 12 2021
Fribourg, Switzerland
@author: Mariano Barella

Auxiliary functions for "deconvolution_of_lifetime_decay.py"

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re

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

def replace_comma_dot(folder, extension):
    print('\nReplacing comma by dot...')
    list_of_files = os.listdir(folder)
    list_of_files.sort()
    list_of_files = [f for f in list_of_files\
                     if not os.path.isdir(os.path.join(folder, f))]
    list_of_files = [f for f in list_of_files if re.search(extension, f)]
    
    L = len(list_of_files)
    
    for i in range(L):
        filename = list_of_files[i]
        print(filename,'edited.')
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r+') as f:
            text = f.read()
            f.seek(0)
            f.truncate()
            f.write(text.replace(',','.'))
            f.close()
    print('----- Replacing done -----')
    return

