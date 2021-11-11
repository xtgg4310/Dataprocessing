import matplotlib.pyplot as plt
import time
import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg
import numpy as np
import PortSTM32
from numpy import linalg as lin

if __name__ == '__main__':
    name = ['front', 'left', 'back', 'right']
    name2 = ['f', 'l', 'b', 'r']
    path_for_data = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220425_breath"
    # path_for_data2 = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220421_thinking_"
    path_for_data2_time = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220421_t_"
    path_for_data_suffix = r".txt"
    path_for_data_suffix_time = r"_time.txt"
    # path_for_data_suffix_time2 = r""
    path_for_noise = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220425_noise.txt"
    path_for_noise_time = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                          r"Haozheng\PC\Desktop\newDATA\220421_static_time.txt "
    # path_for_main = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220421_"
    path_for_noise_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220421_noise"
    for file in range(4):
        data_signal, timestamp = txt.txt_to_matrix_DAQ_RPI(path_for_data + name[file] + path_for_data_suffix)
        noise_, noise_time = txt.txt_to_matrix_DAQ_RPI(path_for_noise)
        # noise_ = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
        #                                  r"Haozheng\PC\Desktop\newDATA\noise_box.txt", False)
        time_pulse = txt.txt_to_matrix_speed_time(path_for_data + name[file] + path_for_data_suffix_time)
        noise_time_pulse = txt.txt_to_matrix_speed_time(path_for_noise_time)
        time1 = float(timestamp[0])
        time2 = float(timestamp[1])
        L = len(data_signal)
        L_n = len(noise_)
        St_len = len(time_pulse)
        rate = L / (time2 - time1)
        print(rate)
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
        while i < len(dic) - 1 and m < len(dic_n) - 1 and index_t < len(time_pulse) - 1 and index_t_n < len(
                noise_time_pulse) - 1:
            # print((record_timestamp[i+1]-record_timestamp[i])/(dic[i+1]-dic[i]))

            len_index = 0
            index = 0
            start, end, k = sg.search_start_index(i, dic)
            start_n, end_n, m_n = sg.search_start_index(m, dic_n)
            new_re, time_re, emit, time_em, index_t, rate = sg.echo_extract(data, start, end, time, time_pulse, index_t)
            new_re_n, time_re_n, emit_n, time_em_n, index_t_n, rate_noise = sg.echo_extract(data_n, start_n, end_n,
                                                                                            time_n,
                                                                                            noise_time_pulse, index_t_n)
            print(i, k, m_n)
            print(len(time_re) / (time_re[len(time_re) - 1] - time_re[0]))
            print(rate_noise)
            new_re = sg.butter_bandpass_filter(new_re, 40500, 42500, rate)
            new_re_n = sg.butter_bandpass_filter(new_re_n, 40500, 42500, rate_noise)
            en = sg.energy_harvest(new_re)
            # en_n = sg.energy_harvest(new_re_n)
            en_s = sg.echo_get_(en, 0.8, 80, False)
            # en_n_s = sg.echo_get_(en_n, 0.8, 147, False)
            path_file_for_signal = path_for_data + name[file] + r"\220421_signal_" + str(i) + ".txt "
            path_file_for_emit = path_for_data + name[file] + r"\220421_emit_" + str(i) + ".txt "
            path_file_for_energy = path_for_data + name[file] + r"\220421_energy_" + str(i) + ".txt "
            path_file_for_energy_sum = path_for_data + name[file] + r"\220421_energy_sum_" + str(i) + ".txt"
            path_file_for_noise = path_for_noise_folder + r"\220421_static_" + str(i) + ".txt"
            path_file_for_noise_emit = path_for_noise_folder + r"\220421_static_emit_" + str(i) + ".txt"
            sg.record_echos(path_file_for_signal, new_re, time_re)
            sg.record_echos(path_file_for_emit, emit, time_em)
            sg.record_echos(path_file_for_energy, en, time_re)
            sg.record_echos(path_file_for_energy_sum, en_s, time_re)
            sg.record_echos(path_file_for_noise, new_re_n, time_re_n)
            sg.record_echos(path_file_for_noise_emit, emit_n, time_em_n)
            plt.show()
            i = k
            m = m_n

    for file in range(4):
        data_signal, timestamp = txt.txt_to_matrix_DAQ_RPI(path_for_data2 + name[file] + path_for_data_suffix)
        noise_, noise_time = txt.txt_to_matrix_DAQ_RPI(path_for_noise)
        # noise_ = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
        #                                  r"Haozheng\PC\Desktop\newDATA\noise_box.txt", False)
        time_pulse = txt.txt_to_matrix_speed_time(path_for_data2_time + name2[file] + path_for_data_suffix_time)
        noise_time_pulse = txt.txt_to_matrix_speed_time(path_for_noise_time)
        time1 = float(timestamp[0])
        time2 = float(timestamp[1])
        L = len(data_signal)
        L_n = len(noise_)
        St_len = len(time_pulse)
        rate = L / (time2 - time1)
        print(rate)
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
        while i < len(dic) - 1 and m < len(dic_n) - 1 and index_t < len(time_pulse) - 1 and index_t_n < len(
                noise_time_pulse) - 1:
            # print((record_timestamp[i+1]-record_timestamp[i])/(dic[i+1]-dic[i]))

            len_index = 0
            index = 0
            start, end, k = sg.search_start_index(i, dic)
            start_n, end_n, m_n = sg.search_start_index(m, dic_n)
            new_re, time_re, emit, time_em, index_t, rate = sg.echo_extract(data, start, end, time, time_pulse, index_t)
            new_re_n, time_re_n, emit_n, time_em_n, index_t_n, rate_noise = sg.echo_extract(data_n, start_n, end_n,
                                                                                            time_n,
                                                                                            noise_time_pulse, index_t_n)
            print(i, k, m_n)
            print(len(time_re) / (time_re[len(time_re) - 1] - time_re[0]))
            print(rate_noise)
            new_re = sg.butter_bandpass_filter(new_re, 40500, 42500, rate)
            new_re_n = sg.butter_bandpass_filter(new_re_n, 40500, 42500, rate_noise)
            en = sg.energy_harvest(new_re)
            # en_n = sg.energy_harvest(new_re_n)
            en_s = sg.echo_get_(en, 0.8, 80, False)
            # en_n_s = sg.echo_get_(en_n, 0.8, 147, False)
            path_file_for_signal = path_for_data2 + name[file] + r"\220421_signal_" + str(i) + ".txt "
            path_file_for_emit = path_for_data2 + name[file] + r"\220421_emit_" + str(i) + ".txt "
            path_file_for_energy = path_for_data2 + name[file] + r"\220421_energy_" + str(i) + ".txt "
            path_file_for_energy_sum = path_for_data2 + name[file] + r"\220421_energy_sum_" + str(i) + ".txt"
            path_file_for_noise = path_for_noise_folder + r"\220421_static_" + str(i) + ".txt"
            path_file_for_noise_emit = path_for_noise_folder + r"\220421_static_emit_" + str(i) + ".txt"
            sg.record_echos(path_file_for_signal, new_re, time_re)
            sg.record_echos(path_file_for_emit, emit, time_em)
            sg.record_echos(path_file_for_energy, en, time_re)
            sg.record_echos(path_file_for_energy_sum, en_s, time_re)
            sg.record_echos(path_file_for_noise, new_re_n, time_re_n)
            sg.record_echos(path_file_for_noise_emit, emit_n, time_em_n)
            plt.show()
            i = k
            m = m_n
