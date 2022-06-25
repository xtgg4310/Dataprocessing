import matplotlib.pyplot as plt
import time

from scipy.signal import stft

import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg
import numpy as np
import PortSTM32
from numpy import linalg as lin
import signal_process as sg

if __name__ == '__main__':
    suffix = "_1.txt"
    time_suffix = "_1.txt"
    file_name = r"\test"
    path_for_data = r"C:\Users\Enigma_2020\Desktop\collect_data"
    data_signal, timestamp = txt.txt_to_matrix_DAQ_RPI(path_for_data + file_name + suffix)
    time_pulse = txt.txt_to_matrix_speed_time(path_for_data + file_name + time_suffix)
    print(time_pulse[0][0], time_pulse[len(time_pulse) - 1][0])
    print(data_signal[1][0])
    L = len(data_signal)
    data = [0] * L
    time = [0] * L
    for i in range(L):
        data[i] = data_signal[i][0]
        time[i] = timestamp[0]


    ##plt.show()
    f, t, zxx = stft(data, 100000, nperseg=150, noverlap=120, nfft=100000)
    plt.pcolormesh(t, f[40500:42500], zxx[40500:42500])
