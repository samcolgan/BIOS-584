import numpy as np


# This function is merely for BIOS 584 debugging purposes
def compute_D_partial(input_signal):
    r"""
    :param input_signal:
    """
    T_len = len(input_signal)
    signal_diff_one = input_signal[-1] - input_signal[1:]
    D_val = np.sum(np.sqrt(1+signal_diff_one**2)) / (T_len - 1)
    return D_val


def compute_D_correct(input_signal):
    r"""
    :param input_signal:
    """
    T_len = len(input_signal)
    signal_diff_one = np.empty(T_len-1)
    for iter in np.arange(0, T_len-1):
        signal_diff_one[iter] = np.sqrt(1 + (input_signal[iter] - input_signal[iter+1])**2)
    D_val = np.sum(signal_diff_one)
    return D_val