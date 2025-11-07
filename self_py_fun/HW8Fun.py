# Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf # This will be used to create a PDF to store multiple plots in the same file

# Define current directory
plots_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Programming Software\\GitHub\\BIOS-584\\K114'

# Calculate Mean and Covariance
def produce_trun_mean_cov(input_signal, input_type, E_val):
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)
        E_val: integer, (number of electrodes)

    return:
    -----
        A list of 5 arrays including
            signal_tar_mean, (E_val, length_per_electrode)
            signal_ntar_mean, (E_val, length_per_electrode)
            signal_tar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_ntar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_all_cov, (E_val, length_per_electrode, length_per_electrode)

    note:
    -----
        descriptive mean and sample covariance statistics from real data
        In this case, E_val=16, length_per_electrode=25.
        But you should pass them as arguments or calculate them inside the function.
    """
    # Define length_per_electrode
    length_per_electrode = int((input_signal.shape[1] / E_val))
    # Break input_signal into target and non-target components
    signal_tar = input_signal[input_type == 1]
    signal_ntar = input_signal[input_type == -1]
    # Initialize output arrays
    signal_tar_mean = np.empty((E_val, length_per_electrode))
    signal_ntar_mean = np.empty((E_val, length_per_electrode))
    signal_tar_cov = np.empty((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.empty((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.empty((E_val, length_per_electrode, length_per_electrode))
    # Calculate the summary statistics for each column, placed into groups by the length of each electrode
    for electrode in range(E_val):
        this_column_group = electrode * length_per_electrode
        next_column_group = (electrode + 1) * length_per_electrode

        signal_tar_mean[electrode, :] = np.mean(signal_tar[:, this_column_group:next_column_group], axis=0)
        signal_ntar_mean[electrode, :] = np.mean(signal_ntar[:, this_column_group:next_column_group], axis=0)

        signal_tar_cov[electrode, ...] = np.cov(signal_tar[:, this_column_group:next_column_group], rowvar=False)
        signal_ntar_cov[electrode, ...] = np.cov(signal_ntar[:, this_column_group:next_column_group], rowvar=False)
        signal_all_cov[electrode, ...] = np.cov(input_signal[:, this_column_group:next_column_group], rowvar=False)
    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]

# Plot Means
def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit = np.array([-5, 8]), fig_size = (12, 12)
):
    r"""
    :param eeg_tar_mean:
    :param eeg_ntar_mean:
    :param subject_name:
    :param time_index:
    :param E_val:
    :param electrode_name_ls:
    :param y_limit: optional parameter, a list or an array of two numbers
    :param fig_size: optional parameter, a tuple of two numbers
    :return:
    """
    # Start by initializing the mean plots, main title, plot titles, and plot indices
    fig1, mean_plots = plt.subplots(4, 4, figsize = fig_size)
    fig1.suptitle('{} Mean'.format(subject_name), fontsize = 14)
    electrode_name_arr = np.array(electrode_name_ls).reshape(4, 4)
    col_index = 0
    row_index = 0
    # Iteratively fill the x's, y's, titles, and axes labels of each plot
    for electrode in range(E_val):
        mean_plots[row_index, col_index].plot(32 * np.arange(0, 25), eeg_ntar_mean[electrode, :], label = 'Non-Target', color = 'blue')
        mean_plots[row_index, col_index].plot(32 * np.arange(0, 25), eeg_tar_mean[electrode, :], label = 'Target', color = 'red')
        mean_plots[row_index, col_index].set_title('{}'.format(electrode_name_arr[row_index, col_index]))
        mean_plots[row_index, col_index].set_xlabel('Time (ms)')
        mean_plots[row_index, col_index].set_ylabel('Amplitude (muV)')
        mean_plots[row_index, col_index].set_ylim(y_limit)
        mean_plots[row_index, col_index].set_xlim(np.array([0, 800]))
        if col_index < 3:
            col_index = col_index + 1
        else:
            col_index = 0
            row_index = row_index + 1
    # Print the plot
    plt.tight_layout()
    plt.subplots_adjust(wspace = 1.0)
    plt.savefig('{}\\Mean.png'.format(plots_dir))

# Plot Covariance
def plot_trunc_cov(
        eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(14,12)
):
    # Create the meshgrid of time points: (25x25) for each
    x, y = np.meshgrid(time_index, time_index)
    # Initialize the cov plots, main title, plot titles, and plot indices
    fig1, cov_plots = plt.subplots(4, 4, figsize = fig_size)
    fig1.suptitle('{} Covariance {}'.format(subject_name, cov_type), fontsize = 14)
    electrode_name_arr = np.array(electrode_name_ls).reshape(4, 4)
    col_index = 0
    row_index = 0
    # Iteratively fill the x's, y's, titles, and axes labels of each plot
    for electrode in range(E_val):
        cov_plots[row_index, col_index].contourf(x, y, eeg_cov[electrode], levels = 20, cmap = 'coolwarm')
        cov_plots[row_index, col_index].set_title('{}'.format(electrode_name_arr[row_index, col_index]))
        cov_plots[row_index, col_index].set_xlabel('Time (ms)')
        cov_plots[row_index, col_index].set_ylabel('Time (ms)')
        if col_index < 3:
            col_index = col_index + 1
        else:
            col_index = 0
            row_index = row_index + 1
    # Print the plot
    plt.tight_layout()
    plt.subplots_adjust(wspace = 1.0)
    plt.savefig('{}\\Covariance_{}.png'.format(plots_dir, cov_type))