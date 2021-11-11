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
import calucate as ca

if __name__ == '__main__':
    path_for_data = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220425_breath_motion"
    path_for_data_suffix = r".txt"
    path_for_data_suffix_time = r"_time.txt"
    path_for_main = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\newDATA\220425_breath_motion"
    path_for_noise_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                            r"Haozheng\PC\Desktop\newDATA\220425_breath_motion_noise"
    path_for_save1 = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\fig\220425_breath_motion"
    for index_file in range(1):
        file_index_folder = str(index_file + 1)
        if file_index_folder != '1':
            path_for_data_folder = path_for_data + file_index_folder
            save_path = path_for_save1 + file_index_folder
        else:
            path_for_data_folder = path_for_data
            save_path = path_for_save1
        path_for_noise_folder = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou " \
                                r"Haozheng\PC\Desktop\newDATA\220425_breath_motion_noise"
        path_for_signal_suffix = r"\breath_" + file_index_folder + r"_signal_"
        path_for_energy_suffix = r"\breath_" + file_index_folder + r"_energy_"
        path_for_energy_sum_suffix = r"\breath_" + file_index_folder + r"_energy_sum_"
        path_for_noise_suffix = r"\220425_noise_"
        # path_for_noise_emit_suffix = r"\220425_static_emit_"
        # suffix = "_impulse.txt"
        if file_index_folder == '1':
            L = 200
        else:
            L = 399
        # num = 400
        # index_list = ca.select_data_random(L, num, False)
        # length = len(index_list)
        re_central = {}
        re_shift = {}
        re_diff = {}
        re_sec_shift = {}
        re_sec_diff = {}
        re_index = 0
        aver_length = {}
        ener_enev = {}
        for i in range(L):
            file_index = str(i) + ".txt"
            print(i)
            if not os.path.exists(path_for_data_folder + path_for_signal_suffix + file_index) or not os.path.exists(
                    path_for_noise_folder + path_for_noise_suffix + file_index):
                continue
            data, time = txt.txt_to_matrix_feature(path_for_data_folder + path_for_signal_suffix + file_index)
            noise, time_n = txt.txt_to_matrix_feature(path_for_noise_folder + path_for_noise_suffix + file_index)

            # emit, time_em = txt.txt_to_matrix_feature(path_for_data_folder + path_for_signal_emit_suffix + file_index)
            # emit_n, time_em_n = txt.txt_to_matrix_feature(
            #    path_for_noise_folder + path_for_noise_emit_suffix + file_index)
            rate = len(time) / (time[len(time) - 1] - time[0])
            rate_n = len(time_n) / (time_n[len(time_n) - 1] - time_n[0])
            dis = sg.index2dis(0, 799, rate)
            dis_n = sg.index2dis(0, 799, rate_n)
            data = sg.matrix2list(data)
            # emit = sg.matrix2list(emit)
            # emit_n = sg.matrix2list(emit_n)
            noise = sg.matrix2list(noise)
            data_enve = sg.envelop_extraction_hilbert(data)
            noise_enve = sg.envelop_extraction_hilbert(noise)
            new_re = data[0:800]
            new_time = time[0:800]
            new_static = noise[0:800]
            new_time_s = time_n[0:800]
            data_new_enve = sg.envelope_extraction_hilbert_peak(data_enve)
            noise_new_enve = sg.envelope_extraction_hilbert_peak(noise_enve)
            dic = sg.envelope_threshold_index(data_new_enve, 0.3)
            dis_re, result = sg.spectrum_background_noise(file_index_folder, i, data[0:2000], rate, noise[0:2000],
                                                          rate_n)
            print(dic[0] * 34000 / (rate * 2))
            plt.plot(dis, data[0:800], label='signal', color='b')
            plt.plot(dis_n, noise[0:800], label='noise', alpha=0.4, color='y')
            plt.ylabel("Amplitude")
            plt.xlabel("Distance[cm]")
            plt.legend()
            plt.grid(axis="y")
            plt.ylim(-4, 4)
            plt.savefig(save_path + r"\signal_" + file_index + ".png")
            plt.close()
            plt.plot(dis, data_new_enve[0:800], label='signal', color='b')
            plt.plot(dis_n, noise_new_enve[0:800], label='noise', color='y')
            plt.ylabel("Amplitude")
            plt.xlabel("Distance[cm]")
            plt.legend()
            plt.grid(axis="y")
            plt.ylim(0, 4)
            plt.savefig(save_path + r"\signal_envelope_" + file_index + ".png")
            plt.close()
            temp = [0] * len(result)
            temp1 = [0] * len(result)
            temp2 = [0] * len(result)
            temp3 = [0] * len(result)
            temp4 = [0] * len(result)
            print(len(result))
            for m in range(0, len(result)):
                temp[m] = result[m][0]
            re_central[re_index] = temp
            for m in range(0, len(result)):
                temp1[m] = result[m][1]
                temp2[m] = result[m][2]
                temp3[m] = temp1[m] - temp[m] if -250 < temp1[m] - temp[m] < 250 else 0
                temp4[m] = temp2[m] - temp[m]
            re_shift[re_index] = temp1
            re_sec_shift[re_index] = temp2
            re_diff[re_index] = temp3
            re_sec_diff[re_index] = temp4
            ener_enev[re_index] = np.max(data_new_enve[0:800])
            re_index += 1
            aver_length.update({i: time[0]})
            print(re_index)
        relative_move_time = [0] * len(aver_length)
        re_i = 0
        en_re = [0] * len(ener_enev)
        for i in range(len(en_re)):
            en_re[i] = ener_enev[i]
        for keys in aver_length.keys():
            relative_move_time[re_i] = aver_length[keys]
            re_i += 1
        for i in range(0, len(re_central)):
            path = path_for_main + r"\fig_n" + str(i) + ".png"
            plt.subplot(511)
            plt.plot(re_central[i])
            plt.title("central")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Numbers")
            plt.subplot(512)
            plt.plot(re_shift[i])
            plt.title("shift")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Numbers")
            plt.subplot(513)
            plt.plot(re_diff[i])
            plt.title("diff")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Numbers")
            plt.subplot(514)
            plt.plot(re_sec_shift[i])
            plt.title("Sec_shift")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Numbers")
            plt.subplot(515)
            plt.plot(re_sec_diff[i])
            plt.title("Sec_diff")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Numbers")
            plt.tight_layout()
            # plt.show()
            plt.savefig(save_path + r"\result_shift_" + str(i) + ".png")
            plt.close()
        aver = [0] * len(re_diff)
        speed2 = [0] * len(re_diff)
        speed = [0] * len(re_diff)
        for i in range(0, len(re_diff)):
            aver[i] = sg.aver_motion(re_diff[i])
            speed2[i] = sg.doppler_move_shift(sg.aver_motion(re_sec_diff[i]), 41500)
            speed[i] = sg.doppler_move_shift(aver[i], 41500)
        plt.plot(relative_move_time, aver)
        plt.title("shift_total")
        plt.ylabel("fre_shift [Hz]")
        plt.xlabel("time[sec]")
        # plt.show()
        plt.savefig(save_path + r"\total_shift.png")
        plt.close()
        plt.plot(relative_move_time, speed)
        plt.title("shift_total")
        plt.ylabel("speed")
        plt.xlabel("time[sec]")
        # plt.show()
        plt.savefig(save_path + r"\speed.png")
        plt.close()
        plt.plot(relative_move_time, speed2)
        plt.title("shift_total")
        plt.ylabel("speed2")
        plt.xlabel("time[sec]")
        # plt.show()
        plt.savefig(save_path + r"\speed2.png")
        plt.close()
        fft_data, fft_rate, rate_speed = sg.fft_data(aver, relative_move_time, 0, len(aver))
        plt.plot(fft_rate, fft_data)
        plt.title("move_fre_feature")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency[Hz]")
        # plt.show()
        plt.savefig(save_path + r"\motion_fre.png")
        plt.close()

        plt.plot(relative_move_time, en_re)
        plt.title("move_ener_feature")
        plt.ylabel("Amplitude")
        plt.xlabel("Time[sec]")
        # plt.show()
        plt.savefig(save_path + r"\motion_enev.png")
        plt.close()
        # sg.object_movement_movement(speed, rate_movement_sample, relative_move_time, speed2)
        fd = open(save_path + r"\record_speed.txt", "w")
        fd2 = open(save_path + r"\record_speed2.txt", "w")
        fd3 = open(save_path + r"\record_motion_fre.txt", "w")
        for i in range(0, len(re_diff)):
            fd.writelines(str(speed[i]) + "," + str(relative_move_time[i]) + "\n")
            fd2.writelines(str(speed2[i]) + "," + str(relative_move_time[i]) + "\n")
        fd3.writelines("count: " + str(len(re_central)) + "\n")
        # fd3.writelines("fre_motion2: " + str(fre_motion2) + "\n")
