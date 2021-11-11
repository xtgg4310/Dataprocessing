import random
import matplotlib.pyplot as plt
import time
import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg
import numpy as np
import PortSTM32
from numpy import linalg as lin
import os
from scipy.signal import stft


def select_data_random(number, num, flag=True):
    index = list(range(0, number))
    if not flag:
        return index
    random.shuffle(index)
    return index[0:num]


def target_function(dis, a, b, n):
    return a * np.cos(b) / dis ** n


if __name__ == '__main__':

    name = ['front', 'left', 'back', 'right']

    color_ = ["black", "grey", "red", "yellow", "lime", "forestgreen", "teal", "blue", "tan", "violet"]
    dis_re = {}
    enve_re = {}
    record_signal = {}
    path_for_save1 = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\fig\human_"
    path_for_save2 = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\fig\thinking_"
    for index_file in range(8):

        file_index_folder = name[index_file % 4]
        if index_file < 4:
            path_for_data_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                                   r"Haozheng\PC\Desktop\newDATA\220421_" + \
                                   file_index_folder
            path_for_save = path_for_save1
        else:
            path_for_data_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                                   r"Haozheng\PC\Desktop\newDATA\220421_thinking_" + \
                                   file_index_folder
            path_for_save = path_for_save2
        path_for_noise_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                                r"Haozheng\PC\Desktop\newDATA\220421_static"
        path_for_signal_suffix = r"\220421_signal_"
        path_for_signal_emit_suffix = r"\220421_signal_emit"
        path_for_energy_suffix = r"\220421_energy_"
        path_for_energy_sum_suffix = r"\220421_energy_sum_"
        path_for_noise_suffix = r"\220421_static_"
        path_for_noise_emit_suffix = r"\220421_static_emit_"

        L = 49
        num = 1
        index_list = select_data_random(L, num, False)
        length = len(index_list)
        dis_record = [0] * length
        energy_record = [0] * length
        index = 0

        for i in range(length):
            file_index = str(index_list[i]) + ".txt"
            print(i)
            if not os.path.exists(path_for_data_folder + path_for_signal_suffix + file_index):
                continue
            data, time = txt.txt_to_matrix_feature(path_for_data_folder + path_for_signal_suffix + file_index)
            ener, time_en = txt.txt_to_matrix_feature(path_for_data_folder + path_for_energy_suffix + file_index)
            ener_s, time_en_s = txt.txt_to_matrix_feature(path_for_data_folder + path_for_energy_sum_suffix + \
                                                          file_index)
            noise, time_n = txt.txt_to_matrix_feature(path_for_noise_folder + path_for_noise_suffix + file_index)
            rate = len(time) / (time[len(time) - 1] - time[0])
            rate_n = len(time_n) / (time_n[len(time_n) - 1] - time_n[0])
            dic = sg.get_object_index(ener_s[0:900], 0.2, length=3)

            en_n = sg.energy_harvest(noise)
            dis = sg.index2dis(0, 899, rate)
            dis_n = sg.index2dis(0, 899, rate_n)
            print(dic)
            print(dic[0][0])
            dis_record[index] = int(dic[0][0] * 34000 / (rate * 2))
            energy_record[index] = dic[0][1]
            en_n_s = sg.echo_get_(en_n, 0.8, 80, False)
            data_enve = sg.envelop_extraction_hilbert(data)
            noise_enve = sg.envelop_extraction_hilbert(noise)
            data_new_enve = sg.envelope_extraction_hilbert_peak(data_enve)
            noise_new_enve = sg.envelope_extraction_hilbert_peak(noise_enve)
            plt.plot(dis, data[0:900], label='signal', color='b')
            plt.plot(dis_n, noise[0:900], label='noise', alpha=0.4, color='y')
            plt.ylabel("Amplitude")
            plt.xlabel("Distance[cm]")
            plt.legend()
            plt.grid(axis="y")
            plt.ylim(-4, 4)
            plt.savefig(path_for_save + file_index_folder + r"\signal_" + file_index + ".png")
            plt.close()
            plt.plot(dis, ener_s[0:900], label='signal', color='b')
            plt.plot(dis_n, en_n_s[0:900], label='noise', alpha=0.4, color='y')
            plt.ylabel("Amplitude")
            plt.xlabel("Distance[cm]")
            plt.legend()
            plt.grid(axis="y")
            plt.ylim(0, 130)
            plt.savefig(path_for_save + file_index_folder + r"\energy_" + file_index + ".png")
            plt.close()
            plt.plot(dis, data_new_enve[0:900], label='signal', color='b')
            plt.plot(dis_n, noise_new_enve[0:900], label='noise', color='y')
            plt.ylabel("Amplitude")
            plt.xlabel("Distance[cm]")
            plt.legend()
            plt.grid(axis="y")
            plt.ylim(0, 4)
            plt.savefig(path_for_save + file_index_folder + r"\signal_envelope_" + file_index + ".png")
            plt.close()

            data = sg.matrix2list(data)
            noise = sg.matrix2list(noise)
            record_signal[index] = data[0:900]
            enve_re[index] = data_new_enve[0:900]
            dis_re[index] = dis

            '''
            data = sg.matrix2list(data)
            noise = sg.matrix2list(noise)
            
            f, t, zxx = stft(data[0:900], rate, nperseg=200, noverlap=120, nfft=100000)
            f_n, t_n, zxx_n = stft(noise[0:900], rate_n, nperseg=200, noverlap=120, nfft=100000)
            record_zxx = np.abs(zxx)
            record_noise = np.abs(zxx_n)
            
            for m in range(0, min(len(t), len(t_n))):
                for j in range(40000, 50000):
                    if record_zxx[j][m] > record_noise[j][m]:
                        record_zxx[j][m] -= record_noise[j][m]
                    else:
                        record_zxx[j][m] = 0
            record_zxx = sg.normal_energy(record_zxx)
            plt.subplot(311)
            plt.pcolormesh(t, f[40500:42500], record_zxx[40500:42500])
            plt.title("signal without noise")
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.subplot(312)
            plt.pcolormesh(t, f[40500:42500], sg.normal_energy(np.abs(zxx)[40500:42500]))
            plt.title("signal")
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.subplot(313)
            plt.pcolormesh(t_n, f_n[40500:42500], sg.normal_energy(np.abs(zxx_n)[40500:42500]))
            plt.title("noise")
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.tight_layout()
            plt.rcParams['pcolor.shading'] = 'nearest'
            # plt.show()
            plt.savefig(path_for_save + r"\stft_" + file_index_folder + "_" + file_index + ".png")
            plt.close()
            '''
            index += 1

        f1 = open(path_for_save + file_index_folder + r"\record_" + "dis.txt", 'w')
        f2 = open(path_for_save + file_index_folder + r"\record_" + "energy.txt", 'w')
        for k in range(index):
            f1.writelines(str(dis_record[k]) + "\n")
        for k in range(index):
            f2.writelines(str(energy_record[k]) + "\n")
        f1.close()
        f2.close()
