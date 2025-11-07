# Import functions
import numpy as np
import scipy.io as sio
import os
from self_py_fun.HW8Fun import *

# Global Variables
bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
E_val = 16

parent_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Programming Software\\GitHub\\BIOS-584'
parent_data_dir = '{}/data'.format(parent_dir)
time_index = np.linspace(0, 800, 25) # This is a hypothetic time range up to 800 ms after each stimulus.

subject_name = 'K114'
session_name = '001_BCI_TRN'

# Create K114 directory to store plots:
if os.path.exists('{}/subject_name'.format(parent_dir)):
    print('Folder subject_name already exists')
else:
    os.mkdir('{}/subject_name'.format(parent_dir))
    print('Folder subject_name created')

# Load dataset from MATLAB file
eeg_trunc_obj = sio.loadmat('{}/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat'.format(parent_data_dir))

# Extract signal and type objects from eeg_trunc_obj
# Rename Signal and Type variables
eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = eeg_trunc_obj['Type']

# Convert eeg_trunc_type to a 1D array
eeg_trunc_type = np.squeeze(eeg_trunc_type, axis = 1)

# Call the functions
output_arr = produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, E_val)

plot_trunc_mean(
        output_arr[0], output_arr[1], subject_name, time_index, E_val, electrode_name_ls,
        y_limit = np.array([-5, 8]), fig_size=(12, 12))

plot_trunc_cov(
        output_arr[2], 'Target', time_index, subject_name, E_val, electrode_name_ls,
        fig_size=(14,12))

plot_trunc_cov(
        output_arr[3], 'Non-target', time_index, subject_name, E_val, electrode_name_ls,
        fig_size=(14,12))

plot_trunc_cov(
        output_arr[4], 'All', time_index, subject_name, E_val, electrode_name_ls,
        fig_size=(14,12))