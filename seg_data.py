import matplotlib.pyplot as plt
import time
import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg
import numpy as np
import PortSTM32
from numpy import linalg as lin

if __name__ == '__main__':
    path_for_data = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220425_breath_motion"
    path_for_data_suffix = r".txt"
    path_for_data_suffix_time = r"_time.txt"
    path_for_noise = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                     r"Haozheng\PC\Desktop\newDATA\220425_breath_motion_noise.txt "
    path_for_noise_time = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                          r"Haozheng\PC\Desktop\newDATA\220425_breath_motion_noise_time.txt"
    path_for_main = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220425_breath_motion"
    path_for_noise_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                            r"Haozheng\PC\Desktop\newDATA\220425_breath_motion_noise"
    for file_index in range(1):
        index_file = str(file_index + 1)
        if index_file != '1':
            data_signal, timestamp = txt.txt_to_matrix_DAQ_RPI(path_for_data + index_file + path_for_data_suffix)
            time_pulse = txt.txt_to_matrix_speed_time(path_for_data + index_file + path_for_data_suffix_time)
        else:
            data_signal, timestamp = txt.txt_to_matrix_DAQ_RPI(path_for_data + path_for_data_suffix)
            time_pulse = txt.txt_to_matrix_speed_time(path_for_data + path_for_data_suffix_time)

        noise_, noise_time = txt.txt_to_matrix_DAQ_RPI(path_for_noise)
        # noise_ = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
        #                                  r"Haozheng\PC\Desktop\newDATA\noise_box.txt", False)

        noise_time_pulse = txt.txt_to_matrix_speed_time(path_for_noise_time)
        time1 = float(timestamp[0])
        time2 = float(timestamp[1])
        L = len(data_signal)
        L_n = len(noise_)
        St_len = len(time_pulse)
        rate = L / (time2 - time1)
        data, data_1, time = sg.data_remove_aver(data_signal, timestamp)
        data_n, data_n_1, time_n = sg.data_remove_aver(noise_, noise_time)

        ener = sg.energy_harvest(data_1)
        ener_nor, dic = sg.echo_get_(ener, 0.8, length=147)
        ener_n = sg.energy_harvest(data_n_1)
        ener_n_nor, dic_n = sg.echo_get_(ener_n, 0.80, length=147)

        index_t = 0
        index_t_n = 0

        m = 0
        i = 0
        count = 0
        while i < len(dic) - 1 and m < len(dic_n) - 1 and index_t < len(time_pulse) - 1 and index_t_n < len(
                noise_time_pulse) - 1:
            # print((record_timestamp[i+1]-record_timestamp[i])/(dic[i+1]-dic[i]))

            len_index = 0
            index = 0
            start, end, k = sg.search_start_index(i, dic)
            start_n, end_n, m_n = sg.search_start_index(m, dic_n)
            new_re, time_re, _, _, index_t, rate = sg.echo_extract(data, start, end, time, time_pulse, index_t)
            new_re_n, time_re_n, _, _, index_t_n, rate_noise = sg.echo_extract(data_n, start_n, end_n, time_n,
                                                                               noise_time_pulse, index_t_n)

            new_re = sg.butter_bandpass_filter(new_re, 40500, 42500, rate)
            new_re_n = sg.butter_bandpass_filter(new_re_n, 40500, 42500, rate_noise)
            en = sg.energy_harvest(new_re)
            # en_n = sg.energy_harvest(new_re_n)
            en_s = sg.echo_get_(en, 0.8, 80, False)
            # en_n_s = sg.echo_get_(en_n, 0.8, 147, False)
            if index_file != '1':
                path_file_for_signal = path_for_main + index_file + r"\breath_" + index_file + r"_signal_" + str(
                    i) + ".txt "
                path_file_for_energy = path_for_main + index_file + r"\breath_" + index_file + r"_energy_" + str(
                    i) + ".txt "
                path_file_for_energy_sum = path_for_main + index_file + r"\breath_" + index_file + r"_energy_sum_" + str(
                    i) + ".txt"
            else:
                path_file_for_signal = path_for_main + r"\breath_" + index_file + r"_signal_" + str(
                    i) + ".txt "
                path_file_for_energy = path_for_main + r"\breath_" + index_file + r"_energy_" + str(
                    i) + ".txt "
                path_file_for_energy_sum = path_for_main + r"\breath_" + index_file + r"_energy_sum_" + str(
                    i) + ".txt"
            path_file_for_noise = path_for_noise_folder + r"\220425_noise_" + str(m + count * (len(dic_n) - 1)) + ".txt"
            sg.record_echos(path_file_for_signal, new_re, time_re)
            sg.record_echos(path_file_for_energy, en, time_re)
            sg.record_echos(path_file_for_energy_sum, en_s, time_re)
            sg.record_echos(path_file_for_noise, new_re_n, time_re_n)
            plt.show()
            i = k
            m = m_n
            if m == len(dic_n) - 1:
                count += 1
                m = m % (len(dic_n) - 1)
                index_t_n = 0

        file_index += 1
