import matplotlib.pyplot as plt
import time
import txt2matrix as txt
import RPI_data as RPI
import signal_process as sg
import numpy as np
import PortSTM32
from numpy import linalg as lin

if __name__ == '__main__':
    # data, time = sg.sin_wave(10, 50, 10000, 30, 1)
    # plt.plot(time, data)
    # plt.show()
    # sg.time_phase_draw(data,10000,None,time)
    # time.sleep(5)
    # print(11111111111111111111111111111111111111111111)
    # record, start, end = PortSTM32.read_port(1)

    # record = PortSTM32.decode_port(record)
    # print(300000.0 / (end - start))

    # record[0] = PortSTM32.digit_segment(record[0])
    # number = PortSTM32.get_all_data(record)
    # data = PortSTM32.get_all_data(record)
    # PortSTM32.error_rate(data)
    # time_interval = (end - start) / len(data)
    # time = len(data) * [0]
    # for i in range(0, len(data)):
    #   time[i] = start + time_interval * i

    data_signal, timestamp = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                       r"Haozheng\PC\Desktop\newDATA\220411_stand.txt")
    data_o2, time_o2 = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                 r"Haozheng\PC\Desktop\newDATA\220411_stand_hand.txt")
    data_o3, time_o3 = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                 r"Haozheng\PC\Desktop\newDATA\220411_stand_lean.txt")
    data_o4, time_o4 = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                 r"Haozheng\PC\Desktop\newDATA\220407_45.txt")
    data_o5, time_o5 = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                 r"Haozheng\PC\Desktop\newDATA\220407_30.txt")
    noise_, noise_time = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                   r"Haozheng\PC\Desktop\newDATA\220411_noise.txt")
    # noise_ = txt.txt_to_matrix_DAQ_RPI(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
    #                                  r"Haozheng\PC\Desktop\newDATA\noise_box.txt", False)
    stand_time_pulse = txt.txt_to_matrix_speed_time(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                    r"Haozheng\PC\Desktop\newDATA\220411_std_time.txt")
    stand_time_pulse_2 = txt.txt_to_matrix_speed_time(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                      r"Haozheng\PC\Desktop\newDATA\220411_std_hand_time.txt")
    stand_time_pulse_3 = txt.txt_to_matrix_speed_time(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                      r"Haozheng\PC\Desktop\newDATA\220411_std_lean_time.txt")
    stand_time_pulse_4 = txt.txt_to_matrix_speed_time(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                      r"Haozheng\PC\Desktop\newDATA\220407_45_time.txt")
    stand_time_pulse_5 = txt.txt_to_matrix_speed_time(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                      r"Haozheng\PC\Desktop\newDATA\220407_30_time.txt")
    noise_time_pulse = txt.txt_to_matrix_speed_time(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou "
                                                    r"Haozheng\PC\Desktop\newDATA\220411_noise_time.txt")
    path_for_main = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\fig\width3"
    time1 = float(timestamp[0])
    time2 = float(timestamp[1])
    L = len(data_signal)
    L_n = len(noise_)
    St_len = len(stand_time_pulse)
    data = [0] * L
    data_1 = [0] * L
    time = [0] * L
    rate = L / (time2 - time1)
    print(rate)
    for i in range(0, L):
        time[i] = time1 + i * (time2 - time1) / L
        data[i] = data_signal[i][0]
    re = np.average(data)
    for i in range(0, L):
        data_1[i] = data[i] - re
    data2, data_2, time_2 = sg.data_remove_aver(data_o2, time_o2)
    data3, data_3, time_3 = sg.data_remove_aver(data_o3, time_o3)
    data4, data_4, time_4 = sg.data_remove_aver(data_o4, time_o4)
    data5, data_5, time_5 = sg.data_remove_aver(data_o5, time_o5)
    # data2, data_new_2, time_2 = sg.data_remove_aver(data_2, time_2)
    # plt.plot(data)
    # plt.show()
    # time_n1 = float(noise_time[0])
    # time_n2 = float(noise_time[1])
    data_n = [0] * L_n
    data_n_1 = [0] * L_n
    time_n = [0] * L_n
    # rate_n = L_n / (time_n2 - time_n1)
    # print(rate_n)
    for i in range(0, L_n):
        #    time_n[i] = time_n1 + i * (time_n2 - time_n1) / L
        data_n[i] = noise_[i][0]
    re_n = np.average(data_n)
    for i in range(0, L_n):
        data_n_1[i] = data_n[i] - re_n
    # sg.stft_scalar_calculate(data, 100000)
    ener = sg.energy_harvest(data_1)
    ener_nor, dic = sg.echo_get_(ener, 0.8, length=147)
    ener_nor_2, dic_2 = sg.echo_get_(sg.energy_harvest(data_2), 0.8, length=147)
    ener_nor_3, dic_3 = sg.echo_get_(sg.energy_harvest(data_3), 0.8, length=147)
    ener_nor_4, dic_4 = sg.echo_get_(sg.energy_harvest(data_4), 0.8, length=147)
    ener_nor_5, dic_5 = sg.echo_get_(sg.energy_harvest(data_5), 0.8, length=147)
    # print()
    # print(len(dic))
    # print(len(dic_2))
    # print(len(dic_3))
    # print(len(dic_4))
    # print(len(dic_5))
    ener_n = sg.energy_harvest(data_n_1)
    ener_n_nor, dic_n = sg.echo_get_(ener_n, 0.80, length=147)

    count = [0] * len(dic)
    index_c = 0
    index_t = 0
    index_t_n = 0
    emit = [0] * 80
    time_em = [0] * 80
    i = 0
    i_2 = 0
    i_3 = 0
    i_4 = 0
    i_5 = 0
    m = 0
    re_central = {}
    re_shift = {}
    re_diff = {}
    re_sec_shift = {}
    re_sec_diff = {}
    re_index = 0
    aver_length = {}

    while i < len(dic) - 1 and m < len(dic_n) - 1 and i_2 < len(dic_2) - 1 and i_3 < len(dic_3) - 1 and i_4 < len(
            dic_4) - 1 and i_5 < len(dic_5) - 1 and index_t < len(stand_time_pulse) - 1 and index_t_n < len(
        noise_time_pulse) - 1:
        # print((record_timestamp[i+1]-record_timestamp[i])/(dic[i+1]-dic[i]))
        len_index = 0
        k = i + 1
        while k < len(dic):
            if dic[k] > dic[i] + 6000:
                start = dic[i]
                end = dic[k]
                break
            else:
                k += 1
        k_2 = i_2 + 1
        while k_2 < len(dic_2):
            if dic_2[k_2] > dic_2[i_2] + 6000:
                start_2 = dic_2[i_2]
                end_2 = dic_2[k_2]
                break
            else:
                k_2 += 1
        k_3 = i_3 + 1
        while k_3 < len(dic_3):
            if dic_3[k_3] > dic_3[i_3] + 6000:
                start_3 = dic_3[i_3]
                end_3 = dic_3[k_3]
                break
            else:
                k_3 += 1

        new_re = [0] * (end - start + 1)
        time_re = [0] * (end - start + 1)
        new_re_2 = [0] * (end_2 - start_2 + 1)
        time_re_2 = [0] * (end_2 - start_2 + 1)
        new_re_3 = [0] * (end_3 - start_3 + 1)
        time_re_3 = [0] * (end_3 - start_3 + 1)

        index = 0
        index_2 = 0
        index_3 = 0

        inter = 1 / (stand_time_pulse[index_t + 1][0] - stand_time_pulse[index_t][0])
        inter_2 = 1 / (stand_time_pulse_2[index_t + 1][0] - stand_time_pulse_2[index_t][0])
        inter_3 = 1 / (stand_time_pulse_3[index_t + 1][0] - stand_time_pulse_3[index_t][0])

        for j in range(start, end + 1):
            if 0 <= index <= 170:
                if 0 <= index <= 79:
                    emit[index] = data[j]
                    time_em[index] = time[j]
                new_re[index] = 0
                time_re[index] = stand_time_pulse[index_t][0] + inter * index
            else:
                new_re[index] = data[j]
                time_re[index] = stand_time_pulse[index_t][0] + inter * index
            index += 1

        for j in range(start_2, end_2 + 1):
            if 0 <= index_2 <= 170:
                if 0 <= index_2 <= 79:
                    emit[index_2] = data2[j]
                    time_em[index_2] = time_2[j]
                new_re_2[index_2] = 0
                time_re_2[index_2] = stand_time_pulse_2[index_t][0] + inter_2 * index_2
            else:
                new_re_2[index_2] = data2[j]
                time_re_2[index_2] = stand_time_pulse_2[index_t][0] + inter_2 * index_2
            index_2 += 1

        for j in range(start_3, end_3 + 1):
            if 0 <= index_3 <= 170:
                if 0 <= index_3 <= 79:
                    emit[index_3] = data3[j]
                    time_em[index_3] = time_3[j]
                new_re_3[index_3] = 0
                time_re_3[index_3] = stand_time_pulse_3[index_t][0] + inter_3 * index_3
            else:
                new_re_3[index_3] = data3[j]
                time_re_3[index_3] = stand_time_pulse_3[index_t][0] + inter_3 * index_3
            index_3 += 1

        m_n = m + 1
        while m_n < len(dic_n):
            if dic_n[m_n] > dic_n[m] + 6000:
                start_n = dic_n[m]
                end_n = dic_n[m_n]
                break
            else:
                m_n += 1
        time_re_noise = [0] * (end_n - start_n + 1)
        new_re_n = [0] * (end_n - start_n + 1)
        inter_n = 1 / (noise_time_pulse[index_t_n + 1][0] - noise_time_pulse[index_t_n][0])
        index_n = 0
        for j in range(start_n, end_n + 1):
            if 0 <= index_n <= 170:
                new_re_n[index_n] = 0
            else:
                new_re_n[index_n] = data_n[j]
            time_re_noise[index_n] = noise_time_pulse[index_t_n][0] + index_n * index_n
            index_n += 1

        rate_new = (end - start + 1) / (stand_time_pulse[index_t + 1][0] - stand_time_pulse[index_t][0])
        rate_new_2 = (end_2 - start_2 + 1) / (stand_time_pulse_2[index_t + 1][0] - stand_time_pulse_2[index_t][0])
        rate_new_3 = (end_3 - start_3 + 1) / (stand_time_pulse_3[index_t + 1][0] - stand_time_pulse_3[index_t][0])
        rate_new_noise = (end_n - start_n + 1) / (noise_time_pulse[index_t_n + 1][0] - noise_time_pulse[index_t_n][0])
        index_t += 1
        index_t_n += 1
        new_re = sg.butter_bandpass_filter(new_re, 40500, 42500, rate_new)
        new_re_n = sg.butter_bandpass_filter(new_re_n, 40500, 42500, rate_new_noise)
        new_re_2 = sg.butter_bandpass_filter(new_re_2, 40500, 42500, rate_new_2)
        new_re_3 = sg.butter_bandpass_filter(new_re_3, 40500, 42500, rate_new_3)
        # new_re_4 = sg.butter_bandpass_filter(new_re_2, 40500, 42500, rate_new_2)
        # new_re_5 = sg.butter_bandpass_filter(new_re_3, 40500, 42500, rate_new_3)
        dis1 = sg.index2dis(start, end, rate_new)
        dis2 = sg.index2dis(start_2, end_2, rate_new_2)
        dis3 = sg.index2dis(start_3, end_3, rate_new_3)
        disn = sg.index2dis(start_n, end_n, rate_new_noise)
        print(rate_new, rate_new_2, rate_new_3, rate_new_noise)
        ener_1 = sg.energy_harvest(new_re)
        ener_2 = sg.energy_harvest(new_re_2)
        ener_3 = sg.energy_harvest(new_re_3)
        ener_n = sg.energy_harvest(new_re_n)
        # print(rate_new_noise)
        plt.subplot(221)
        plt.plot(dis1[100:1500], new_re[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_std_" + str(i))
        plt.subplot(222)
        plt.plot(dis2[100:1500], new_re_2[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_stdhand_" + str(i))
        plt.subplot(223)
        plt.plot(dis3[100:1500], new_re_3[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_stdlean_" + str(i))
        plt.subplot(224)
        plt.plot(disn[100:1500], new_re_n[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_noise_" + str(i))
        plt.tight_layout()
        plt.savefig(path_for_main + r"\fig_" + str(i) + r"_time_domain.png")
        plt.close()

        plt.subplot(221)
        plt.plot(dis1[100:1500], ener_1[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_std_en_" + str(i))
        plt.subplot(222)
        plt.plot(dis2[100:1500], ener_2[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_stdhand_en_" + str(i))
        plt.subplot(223)
        plt.plot(dis3[100:1500], ener_3[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_stdlean_en_" + str(i))
        plt.subplot(224)
        plt.plot(disn[100:1500], ener_n[100:1500])
        plt.ylabel("Amplitude")
        plt.xlabel("Dis[cm]")
        plt.title("time_domain_noise_en_" + str(i))
        plt.tight_layout()
        plt.savefig(path_for_main + r"\fig_en_" + str(i) + r"_time_domain.png")
        plt.close()
        # new_re = sg.butter_bandpass_filter(new_re, 40500, 42500, rate_new)
        # new_re_n = sg.butter_bandpass_filter(new_re_n, 40500, 42500, rate_new_noise)
        # length = 80
        '''
        dis, result = sg.spectrum_background_noise(i, new_re[0:2000], rate_new, new_re_n[0:2000], rate_new_noise)
        fre_motion = 0
        count[index_c] = dis
        index_c += 1
        temp = [0] * len(result)
        temp1 = [0] * len(result)
        temp2 = [0] * len(result)
        temp3 = [0] * len(result)
        temp4 = [0] * len(result)
        print(len(result))
        for i in range(0, len(result)):
            temp[i] = result[i][0]
        re_central[re_index] = temp
        for i in range(0, len(result)):
            temp1[i] = result[i][1]
            temp2[i] = result[i][2]
            temp3[i] = temp1[i] - temp[i] if -250 < temp1[i] - temp[i] < 250 else 0
            temp4[i] = temp2[i] - temp[i] 
        re_shift[re_index] = temp1
        re_sec_shift[re_index] = temp2
        re_diff[re_index] = temp3
        re_sec_diff[re_index] = temp4
        re_index += 1
        aver_length.update({len_index: end - start + 1})
        len_index += 1
        '''
        i = k
        i_2 = k_2
        i_3 = k_3
        m = m_n

    '''
    total_len = 0
    for i in range(0, len(aver_length)):
        total_len += aver_length[i]
    rate_movement_sample = total_len / (len(aver_length))
    print(rate_movement_sample)
    interval = rate_movement_sample / 100000
    relative_move_time = [0] * len(re_diff)
    for i in range(0, len(re_diff)):
        relative_move_time[i] = interval * i
    print(len(count))
    print(len(re_diff))
    plt.plot(relative_move_time[0:min(len(count), len(re_diff))], count[0:min(len(count), len(re_diff))])
    plt.title("distance")
    plt.ylabel("distance [cm]")
    plt.xlabel("number of echos")
    # plt.show()
    plt.savefig(path_for_main + r"\dis_change.png")
    plt.close()
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
        plt.savefig(path)
        plt.close()
    aver = [0] * len(re_diff)
    speed2 = [0] * len(re_diff)
    speed = [0] * len(re_diff)
    for i in range(0, len(re_diff)):
        aver[i] = sg.aver_motion(re_diff[i])
        speed2[i] = sg.doppler_move_shift(sg.aver_motion(re_sec_diff[i]), 41500)
        speed[i] = sg.doppler_move_shift(aver[i], 41500)
    plt.plot(aver)
    plt.title("shift_total")
    plt.ylabel("fre_shift [Hz]")
    plt.xlabel("number of echos")
    # plt.show()
    plt.savefig(path_for_main + r"\total_shift.png")
    plt.close()
    sg.object_movement_movement(speed, rate_movement_sample, relative_move_time, speed2)
    fd = open(path_for_main + r"\record_speed.txt", "w")
    fd2 = open(path_for_main + r"\record_speed2.txt", "w")
    fd3 = open(path_for_main + r"\record_motion_fre.txt", "w")
    fd4 = open(path_for_main + r"\distance_measurement.txt", "w")
    for i in range(0, len(re_diff)):
        fd.writelines(str(speed[i]) + "," + str(relative_move_time[i]) + "\n")
        fd2.writelines(str(speed2[i]) + "," + str(relative_move_time[i]) + "\n")
    L_dis = min(len(count), len(re_diff))
    for i in range(0, min(len(count), len(re_diff))):
        if count[i] == 0 and i > L_dis * 0.95:
            break
        fd4.writelines(str(count[i]) + "," + str(relative_move_time[i]) + "\n")
    fd3.writelines("count: " + str(len(re_central)) + "\n")
    fd3.writelines("fre_motion: " + str(fre_motion) + "\n")
    # fd3.writelines("fre_motion2: " + str(fre_motion2) + "\n")
    '''
