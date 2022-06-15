import matplotlib.pyplot as plt
import time
import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg
import numpy as np
import PortSTM32
from numpy import linalg as lin

if __name__ == '__main__':
    suffix = ".txt"
    time_suffix = "_time.txt"
    file_name = "data1"
    path_for_data = r"C:\Users\Enigma_2020\Desktop\collect_data\2022"
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

    plt.plot(data)
    plt.show()
