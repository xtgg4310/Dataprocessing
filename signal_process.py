import math
import random
import matplotlib.pyplot as plt
import numpy as np
from neurodsp import timefrequency as tfq
from scipy import signal
from scipy import stats as st
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import hilbert as hlb
from scipy.signal import stft
import point

path_for = r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\fig\220425_breath_motion"


def envelope_extraction_hilbert_peak(data):
    index = {}
    id = 0
    for i in range(len(data)):
        if i == 0:
            if data[i] - data[i + 1] >= 0:
                index[id] = i
                id += 1
        elif i == len(data) - 1:
            if data[i] - data[i - 1] >= 0:
                index[id] = i
                id += 1
        else:
            if data[i] - data[i - 1] > 0 and data[i] - data[i + 1] >= 0:
                index[id] = i
                id += 1
    new_enve = [0] * len(data)
    i = 0
    k = 0
    while k < len(index):
        if i == index[k]:
            k += 1
            continue
        interval = (data[index[k]] - data[i]) / (index[k] - i)
        # print(i, index[k], interval, data[index[k]], data[i])
        for j in range(i, index[k]):
            new_enve[j] = data[i] + interval * (j - i)
        i = index[k]
        k += 1

    return new_enve


def envelop_extraction_hilbert(signal):
    analytic_signal = hlb(signal)
    envelope_signal = np.abs(analytic_signal)
    return envelope_signal


def get_object_index(ener, threshold, length):
    L = len(ener)
    dic = {}
    index = 0
    ener_new = normal_energy(ener)
    for i in range(0, L):
        if ener_new[i] > threshold:
            if i - length <= 0:
                s = 0
            else:
                s = i - length
            if i + length >= L:
                e = L
            else:
                e = i + length
            if ener_new[i] == max(ener_new[s:e]):
                dic[index] = [i, ener[i][0]]
                index += 1
            else:
                continue
        else:
            continue
    return dic


def record_echos(filename, data, time):
    fd = open(filename, 'w')
    L = len(data)
    print(L)
    for i in range(L):
        fd.writelines(str(data[i]) + "," + str(time[i]) + "\n")
    fd.close()


def echo_extract(data, start, end, time, time_pulse, index_time):
    index = 0
    new_re = [0] * (end - start + 1)
    time_re = [0] * (end - start + 1)
    emit = [0] * 171
    time_em = [0] * 171
    inter = (time_pulse[index_time + 1][0] - time_pulse[index_time][0]) / len(new_re)
    rate = (end - start + 1) / (time_pulse[index_time + 1][0] - time_pulse[index_time][0])
    print(rate)
    for j in range(start, end + 1):
        if 0 <= index <= 170:
            emit[index] = data[j]
            time_em[index] = time_pulse[index_time][0] + inter * index
            new_re[index] = 0
            time_re[index] = time_pulse[index_time][0] + inter * index
        else:
            new_re[index] = data[j]
            time_re[index] = time_pulse[index_time][0] + inter * index
        index += 1
    return new_re, time_re, emit, time_em, index_time + 1, rate


def search_start_index(index, dic):
    index_new = index + 1
    while index_new < len(dic):
        if dic[index_new] > dic[index] + 6000:
            start = dic[index]
            end = dic[index_new]
            break
        else:
            index_new += 1
    return start, end, index_new


def index2dis(start, end, rate):
    dis = [0] * (end + 1 - start)
    for i in range(0, len(dis)):
        dis[i] = i * 34000 / (2 * rate)
    return dis


def matrix2list(data):
    data_new = [0] * len(data)
    for i in range(0, len(data)):
        data_new[i] = data[i][0]
    return data_new


def data_remove_aver(data, time):
    L = len(data)
    data_new = [0] * L
    time_new = [0] * L
    data_re = [0] * L
    time1 = float(time[0])
    time2 = float(time[1])
    for i in range(0, L):
        data_new[i] = data[i][0]
        time_new[i] = time1 + i * (time2 - time1) / L
    re = np.average(data_new)
    for i in range(0, L):
        data_re[i] = data_new[i] - re
    return data_new, data_re, time_new


def cos_wave(A, f, fs, phi, t):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A * np.cos(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y, n


def signal_mag_draw(data, time, start, length):
    plt.plot(time[start:start + length], data[start:start + length], 'c*-')
    plt.title('time_domain')
    plt.show()


def FFT_draw(fft_data, fft_rate, start, length, index=-1, part=False):
    if not part:
        plt.plot(fft_rate[start:start + length], fft_data[start:start + length], 'c*-')
    else:
        plt.plot(fft_rate[start + int(length * 2 / 3):start + length],
                 fft_data[start + int(length * 2 / 3):start + length], 'c*-')
    if index == -1:
        plt.title('fre_domain')
    else:
        plt.title("fre_" + str(index) + "_domain")
    plt.show()


def compute_phase(data, fs, fc=None):
    pha = tfq.phase_by_time(data, fs, fc)
    return pha


def time_phase_draw(data, fs, fc, time):
    pha = compute_phase(data, fs, fc)
    plt.plot(time, pha)
    plt.show()


def relative_phase_draw(data, fs, fc, time):
    pha = compute_phase(data, fs, fc)
    temp = pha[0]
    for i in range(0, len(pha)):
        pha[i] -= temp
    plt.plot(time, pha)
    plt.show()


def stft_scalar_calculate(scalar, rate, windows=100, noverlay=30, length=10000, nflag=False):
    f, t, Zxx = stft(scalar, rate, nperseg=windows, noverlap=noverlay, nfft=length, return_onesided=nflag)
    plt.pcolormesh(t, f[0:600], normal_energy(np.abs(Zxx))[0:600])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def compare_fre_draw(data1_fft, data1_rate, data2_fft, data2_rate):
    plt.subplot(211)
    plt.plot(data1_rate, data1_fft, 'c*-')
    plt.title('STFT_MODULE')
    plt.subplot(212)
    plt.plot(data2_rate, data2_fft, 'c*-')
    plt.title('STFT_data')
    plt.tight_layout()
    plt.show()


def distance_cal(nsamples, rate, speed):
    return ((nsamples / rate) * speed) / 2


def cal_spec_energy(data, start, end):
    ener = [0] * len(data[0])
    # print(len(data[0]))
    for i in range(0, len(data[0])):
        for j in range(start, end):
            ener[i] += data[j][i] * data[j][i]
    return ener


def echo_index_select(ener, thershold):
    ener_new = normal_energy(ener)
    index = 0
    dic = {}
    for i in range(0, len(ener_new)):
        if ener_new[i] > thershold:
            dic.update({index: i})
            index += 1
    return ener_new, dic


def search_central(central, fre):
    index = 0
    for i in range(1, len(fre)):
        if central > fre[i]:
            continue
        else:
            index = i
            break
    return index


def doppler_shift_peak(data, index, start, end, threshold=0.1):
    if ((data[index] - data[start]) / data[index]) > threshold and (
            (data[index] - data[end]) / data[index]) > threshold:
        return True
    else:
        return False


def feature_doppler_central_search(central_up, central_down, time, fre, zxx):
    index_down = search_central(central_down, fre)
    index_up = search_central(central_up, fre)
    print(fre[index_down], fre[index_up])
    for i in range(0, len(time)):
        for j in range(index_down, index_up):
            zxx[j][i] = 0
    New_zxx = normal_energy(zxx)
    dic_right = {}
    dic_left = {}
    index_right = 0
    index_left = 0
    for i in range(0, len(time)):
        record = [0] * len(fre)
        for j in range(0, len(fre)):
            record[j] = New_zxx[j][i]
        for j in range(index_up + 1, len(fre)):
            if record[j] > 0.3 and judge_peak(record, j, length=20):
                # and doppler_shift_peak(record, j, j - 5, j + 5, 0.3):
                dic_right.update({index_right: fre[j]})
                # index_right += 1
                break
            elif j == len(fre) - 1:
                dic_right.update({index_right: (central_up + central_down) / 2})
                break
            else:
                continue
        index_right += 1
        j = index_down - 1
        while j >= 40000:
            if record[j] > 0.3 and judge_peak(record, j, length=10):
                # and doppler_shift_peak(record, j, j - 5, j + 5, 0.3):
                dic_left.update({index_left: fre[j]})
                # index += 1
                break
            elif j == index_down - 1:
                dic_left.update({index_left: (central_up + central_down) / 2})
                break
            else:
                j -= 1
                continue
        index_left += 1
        # index = 0
    return New_zxx, dic_left, dic_right


def search_peak(data, f, start, end, low_thershold, high_thershold=1):
    re = {}
    # data=normal_energy(data)
    for i in range(0, end - start + 1):
        if judge_peak(data, i, length=5) and low_thershold < data[i] < high_thershold:
            re.update({f[start + i]: data[i]})
        else:
            continue
    value = sorted(re.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return value


def object_movement_movement(data, fc, time, data2):
    # data1 = butter_highpass_filter(data, 1, fc)
    temp = [0] * len(data)
    for i in range(0, len(data)):
        flag = 1
        if data[i] < 0:
            flag = -1
        if abs(data[i]) < 10:
            temp[i] = 0
        elif abs(data[i]) < 30:
            temp[i] = 0.3 * flag
        elif abs(data[i]) < 60:
            temp[i] = 0.6 * flag
        elif abs(data[i]) < 90:
            temp[i] = 0.9 * flag
        elif abs(data[i]) < 120:
            temp[i] = 1.2 * flag
        else:
            temp[i] = 1.5 * flag
    print(np.var(data))
    print(np.mean(data))
    print(np.var(temp))
    print(np.mean(temp))

    fd = open(path_for + "\para_record.txt", "w")
    fd.write(str(np.var(data)) + "\n")
    fd.write(str(np.mean(data)) + "\n")
    fd.write(str(np.var(temp)) + "\n")
    fd.write(str(np.mean(temp)) + "\n")
    fd.write(str(np.var(data2)) + "\n")
    fd.write(str(np.mean(data2)) + "\n")
    # plt.subplot(311)
    # plt.plot(time, data1)
    # plt.title("high-fre-motion")
    # plt.ylabel("speed[cm/s]")
    # plt.xlabel("Time[sec]")
    plt.subplot(311)
    plt.plot(time, temp)
    plt.title("motion")
    plt.ylabel("motion level")
    plt.xlabel("Time[sec]")
    plt.subplot(312)
    plt.plot(time, data)
    plt.title("raw data for motion speed")
    plt.ylabel("speed[cm/s]")
    plt.xlabel("Time[sec]")
    plt.subplot(313)
    plt.plot(time, data2)
    plt.title("raw data for motion speed2")
    plt.ylabel("speed[cm/s]")
    plt.xlabel("Time[sec]")
    plt.tight_layout()
    plt.savefig(path_for + "\motion.png")
    plt.close()


def distribution_of_energy_fre_bin(data, start, end, index=-1, range_=-1, range_e=-1):
    # length = len(data)
    width = len(data[0])
    band1_energy = [0] * width
    band2_energy = [0] * width
    band3_energy = [0] * width
    band4_energy = [0] * width
    band5_energy = [0] * width
    if range_ != -1:
        index_start = index - range_ if index - range_ > 0 else 0
        index_end = index + range_e if index + range_e < width else width
    else:
        index_start = 0
        index_end = width
    band = int((end - start + 1) / 5)
    for i in range(index_start, index_end):
        sum_re_1 = 0
        for j in range(start, start + band):
            sum_re_1 += data[j][i]
        band1_energy[i] = sum_re_1
    for i in range(index_start, index_end):
        sum_re_2 = 0
        for j in range(start + band, start + 2 * band):
            sum_re_2 += data[j][i]
        band2_energy[i] = sum_re_2
    for i in range(index_start, index_end):
        sum_re_3 = 0
        for j in range(start + 2 * band, start + 3 * band):
            sum_re_3 += data[j][i]
        band3_energy[i] = sum_re_3
    for i in range(index_start, index_end):
        sum_re_4 = 0
        for j in range(start + 3 * band, start + 4 * band):
            sum_re_4 += data[j][i]
        band4_energy[i] = sum_re_4
    for i in range(index_start, index_end):
        sum_re_5 = 0
        for j in range(start + 4 * band, start + 5 * band):
            sum_re_5 += data[j][i]
        band5_energy[i] = sum_re_5
    return band1_energy, band2_energy, band3_energy, band4_energy, band5_energy


def feature_extraction(data, f, start, end, index, shift_left, shift_right):
    # print(record)
    temp = [0] * (end - start + 1)
    target_start = index - shift_left if index - shift_left > 0 else 0
    target_end = index + shift_right if index + shift_right < len(data[0]) - 1 else len(data[0]) - 1
    print(target_start, target_end)
    record = [[0 for i in range(3)] for j in range(target_end - target_start + 1)]
    in_re = 0
    for i in range(target_start, target_end + 1):
        index_temp = 0

        for j in range(start, end + 1):
            temp[index_temp] = data[j][i]
            index_temp += 1
        value = search_peak(temp, f, start, end, 0.2)
        record[in_re][1] = 41500
        record[in_re][0] = 41500
        record[in_re][2] = 41500
        if len(value) != 0:
            if value[0][1] < 0.85:
                record[in_re][1] = 41500
            elif value[0][0] < 40900 or value[0][0] > 42100:
                record[in_re][1] = 41500
            else:
                record[in_re][1] = value[0][0]
            if len(value) < 2:
                record[in_re][2] = 41500
            else:
                if 0.55 < value[0][1] < 0.85 and 40900 < value[0][0] < 42100:
                    record[in_re][2] = value[0][0]
                elif value[1][1] <= 0.55:
                    record[in_re][2] = 41500
                elif value[1][1] >= 0.85:
                    L = len(value)
                    i = 0
                    while i < L and value[i][1] >= 0.85:
                        i += 1
                    if i < L and value[i][1] > 0.55:
                        record[in_re][2] = value[i][1]
                    else:
                        record[in_re][2] = record[in_re][0]
                elif value[1][0] < 40900 or value[1][0] > 42100:
                    record[in_re][2] = 41500
                else:
                    record[in_re][2] = value[1][0]
        in_re += 1
    # print(record)
    # print(len(record))
    # print(len(record[0]))
    return record


def spectrum_background_noise(file_index, num, data, rate, data_n, rate_n, windows=150, overlap=120,
                              length=100000, noise_flag=True):
    f, t, zxx = stft(data, rate, nperseg=windows, noverlap=overlap, nfft=length)
    f_n, t_n, zxx_n = stft(data_n, rate_n, nperseg=windows, noverlap=overlap, nfft=length)
    # print(f.size)
    record_zxx = np.abs(zxx)
    record_noise = np.abs(zxx_n)
    # e1, dic1 = echo_index_select(cal_spec_energy(np.abs(zxx), 40000, 50000), 0.1)
    # e2, dic2 = echo_index_select(cal_spec_energy(np.abs(zxx_n), 40000, 45000), 0.05)
    if noise_flag:
        for i in range(0, min(len(t), len(t_n))):
            for j in range(40000, 50000):
                if record_zxx[j][i] > record_noise[j][i]:
                    record_zxx[j][i] -= record_noise[j][i]
                else:
                    record_zxx[j][i] = 0
    # print(num)

    record_zxx = normal_energy(record_zxx)
    e1, dic1 = echo_index_select(cal_spec_energy(record_zxx, 40500, 42500), 0.06)
    print(dic1[0])
    re = feature_extraction(record_zxx, f, 41000, 42000, dic1[0], 1, 5)
    d1, d2, d3, d4, d5 = distribution_of_energy_fre_bin(record_zxx, 40500, 42500, dic1[0], 3, 8)
    if file_index == '1':
        path = path_for + r"\fig_" + str(num) + ".png"
    else:
        path = path_for + file_index + r"\fig_" + str(num) + ".png"
    plt.subplot(311)
    plt.pcolormesh(t, f[40500:42500], record_zxx[40500:42500])
    plt.title("signal without noise")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.subplot(312)
    plt.pcolormesh(t, f[40500:42500], normal_energy(np.abs(zxx)[40500:42500]))
    plt.title("signal")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.subplot(313)
    plt.pcolormesh(t_n, f_n[40500:42500], normal_energy(np.abs(zxx_n)[40500:42500]))
    plt.title("noise")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.rcParams['pcolor.shading'] = 'nearest'
    # plt.show()
    plt.savefig(path)
    plt.close()

    if file_index == '1':
        path_for_energy = path_for + r"\fig_" + str(num) + r"_energy.png"
    else:
        path_for_energy = path_for + file_index + r"\fig_" + str(num) + r"_energy.png"
    plt.subplot(111)
    plt.plot(t, d1, label="band1", color='b')
    plt.plot(t, d2, label="band2", color='r')
    plt.plot(t, d3, label="band3", color='g')
    plt.plot(t, d4, label="band4", color='black')
    plt.plot(t, d5, label="band5", color='yellow')
    plt.legend()
    plt.title("fre_band power distribution")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.savefig(path_for_energy)
    plt.close()

    # print(dic1)
    s = dic1[0] * (windows - overlap)
    dis = distance_cal(s, rate, 34000)
    return dis, re


def doppler_move_shift(data, cen_fre):
    speed = (data / cen_fre) * 34000
    return speed


def data_addzero(length, data):
    new_data = [0] * (len(data) + length)
    for i in range(0, len(data)):
        new_data[i] = data[i]
    return new_data


def fft_data(data, time, start1, length, zero=False, length_zero=0):
    rate_temp = cal_sampling_rate(data, time)
    # print(rate_temp)
    # rate_temp = 100000
    if zero:
        data = data_addzero(length_zero, data)
        length += length_zero
    r1 = abs(fft(data[start1:start1 + length - 1]) / length)
    re_r1 = 2 * r1[range(int(length / 2))]

    _rate = [0] * (int(length / 2))
    for i in range(0, int(length / 2)):
        _rate[i] = i * rate_temp / length
    return re_r1, _rate, rate_temp


def signal_draw_save(data, time, fft_data, rate, start, length):
    plt.subplot(211)
    plt.plot(time[start:start + length], data[start:start + length], 'b')
    plt.title('time_domain')

    plt.subplot(212)
    plt.plot(rate, fft_data, 'b')
    plt.title('fre_domain')

    plt.tight_layout()
    plt.savefig('sit_result_signal.jpg')

    plt.show()


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, Wn=normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_bandpass(cutoff1, cutoff2, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = [cutoff1 / nyq, cutoff2 / nyq]
    b, a = signal.butter(order, normal_cutoff, btype='bandpass', analog=False)
    return b, a


def butter_bandpass_filter(data, cutoff1, cutoff2, fs, order=5):
    b, a = butter_bandpass(cutoff1, cutoff2, fs, order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def phase_draw(data, fs, fc):
    plt.phase_spectrum(data, fs, fc, pad_to=1000)
    plt.title("Phase Spectrum of the Signal")
    plt.show()


def cal_sampling_rate(data, time):
    return len(data) / (time[len(data) - 1] - time[0])


def data_removal_noise(data, rate, time, start=0):
    windows = int(math.ceil((time / 1000000) * rate))
    temp = np.mean(data[start:start + windows - 1])
    for i in range(start, start + windows):
        data[i] = temp
    # print(len(data))
    return data


def energy_harvest(data):
    data_energy = [0] * len(data)
    for i in range(0, len(data)):
        data_energy[i] = np.sqrt(data[i] * data[i])
    # plt.plot(data_energy)
    # plt.show()
    return data_energy


def echo_get_(data, threshold, length=80, flag=True):
    index = 0
    energy_sum = [0] * len(data)
    while index + length < len(data):
        energy_sum[index] = np.sum(data[index:index + length])
        index += 1
    if flag:
        energy_sum_new = normal_energy(energy_sum)
        dic = {}
        index = 0
        for i in range(0, len(energy_sum_new)):
            # print(energy_sum[i])
            if energy_sum_new[i] > threshold and judge_peak(energy_sum_new, i):
                # print(i)
                dic.update({index: i - 3})
                index += 1
        return energy_sum, dic
    return energy_sum


def judge_peak(data, index, length=40):
    start = (index - length) if (index > length) else 0
    end = index + length if (index + length < len(data)) else len(data)

    return data[index] == np.max(data[start:end])


def normal_energy(data):
    _range = np.max(data) - np.min(data)
    # print(_range)
    return (data - np.min(data)) / _range


def data_thresold_judge(signal_record, data):
    thresold = np.mean(signal_record)
    if data > thresold + 0.5 or data < thresold - 0.5:
        return True
    else:
        return False


def search_time_for_segmatation(timestamp, record_time, start=0, index=0):
    i = start
    if start >= len(timestamp):
        return -1
    if timestamp[i] > record_time[index]:
        return -1
    while i < len(timestamp) and timestamp[i] < record_time[index]:
        i += 1
    return i


def data_segmentation(data, time, timestamp, rate, remove=True):
    dic = {}
    new_time = {}
    temp = 0
    index = 0
    j = 0
    while j < len(time) - 1:
        if search_time_for_segmatation(timestamp, time, temp, j) == -1:
            j += 1
            continue
        elif j + 1 != len(time):
            temp = search_time_for_segmatation(timestamp, time, temp, j)
            temp_end = search_time_for_segmatation(timestamp, time, temp, j + 1)
            if temp == -1 or temp_end == -1:
                continue
            temp_re = data[temp:temp_end]
            if remove:
                temp_re = data_removal_noise(temp_re, rate, 6500, 0)
            dic.update({index: temp_re})
            new_time.update({index: timestamp[temp:temp_end]})
            # plt.plot(new_time[index], dic[index], 'b')
            # plt.show()
            # print(len(temp_re), temp_end-temp)
            temp = temp_end - 1000
            j += 1
            index += 1
        else:
            continue
    return dic, new_time, index


def data_segmentation_simple(data, time, rate, interval=0.101700):
    length = math.ceil(interval * rate)
    number_seg = int(math.floor(len(data) / length))
    data_new = [0] * number_seg
    time_new = [0] * number_seg
    index = 0
    for i in range(0, number_seg):
        if index + length < len(data):
            data_new[i] = data[index:index + length]
            time_new[i] = time[index:index + length]
            index += length
        else:
            break
    return data_new, time_new


def get_center_frequency(data, time, timestamp, rate):
    dic, new_time, index = data_segmentation(data, time, timestamp, rate, False)
    results = [0] * index
    record_index = 0
    for i in range(0, index):
        fft_re, fft_rate, _ = fft_data(dic[i], new_time[i], 0, len(dic[i]))
        results[record_index] = fft_rate[np.argmax(fft_re[10:len(fft_re)])]
        record_index += 1
    return results


def windows_fft(data, timestamp, start, windows, shift=10):
    index = start
    fft_data_re = {}
    fft_rate = {}
    record = 0
    while index < len(data):
        if index + windows < len(data):
            temp, temp1, _ = fft_data(data, timestamp, index, windows, True, length_zero=10000)
            FFT_draw(temp, temp1, 0, len(temp), index, True)
            fft_data_re.update({record: temp})
            fft_rate.update({record: temp1})
            record += 1
            index += shift
            # print(index)
        else:
            temp, temp1, _ = fft_data(data, timestamp, index, len(data) - index, True, length_zero=10000)
            FFT_draw(temp, temp1, 0, len(temp), index, True)
            fft_data_re.update({record: temp})
            fft_rate.update({record: temp1})
            record += 1
    return fft_data_re, fft_rate, record


def slide_windows(data, n):
    new_data = [0] * len(data)
    if n % 2 == 0:
        left = int(n / 2 - 1)
        right = int(n / 2)
    else:
        left = int(math.floor(n / 2))
        right = int(math.floor(n / 2))
    for i in range(0, len(data)):
        if i - left < 0 or i + right > len(data) - 1:
            if i - left < 0:
                count = 0
                for j in range(0, i + right + 1):
                    new_data[i] += data[j]
                    count += 1
                new_data[i] = new_data[i] / count
            else:
                count = 0
                for j in range(i - left, len(data)):
                    new_data[i] += data[j]
                    count += 1
                new_data[i] = new_data[i] / count
        else:
            count = 0
            for j in range(i - left, i + right + 1):
                new_data[i] += data[j]
                count += 1
            new_data[i] = new_data[i] / count
    return new_data


def distribution_of_signal(data, bin):
    max_value = max(data)
    min_value = min(data)
    interval = (max_value - min_value) / bin
    distri = [0] * bin
    bin_index = [0] * bin

    for i in range(0, len(data)):
        index = int((data[i] - min_value) / bin)
        distri[index] += 1
    for i in range(0, len(bin)):
        bin_index[i] = i * interval
    return distri, bin_index


def pearson_correlartion(data, data2):
    corr1 = [0] * 20
    corr2 = [0] * 20
    for i in range(0, 20):
        corr1[i] = st.pearsonr(data[i:len(data)], data2[0:len(data2) - i])[0]
    for i in range(0, 20):
        corr2[i] = st.pearsonr(data2[i:len(data)], data[0:len(data2) - i])[0]
    plt.plot(corr1, 'b')
    plt.plot(corr2, 'r')
    plt.show()


def KalMan_filter():
    return


def aver_motion(data):
    zero_flag = 0
    count = 0
    re = 0
    for i in range(len(data)):
        if data[i] != 0:
            zero_flag = 1
            re += data[i]
            count += 1
        else:
            continue
    if zero_flag:
        return re / count
    else:
        return 0


def envelope_threshold_index(data, threshold):
    data_new = normal_energy(data)
    dic = {}
    index = 0
    for i in range(len(data_new)):
        if data_new[i] >= threshold:
            dic[index] = i
    return dic


def discrete_impulse_response(emit, data, static, emit_n):
    f = fft(data)
    new_em = data_addzero(len(data) - len(emit), emit)
    new_em_n = data_addzero(len(data) - len(emit_n), emit_n)
    f_b = fft(static)
    f_e = fft(new_em)
    f_e_n = fft(new_em_n)
    Back_re = [0] * len(f_b)
    for i in range(len(Back_re)):
        Back_re[i] = f_b[i] / f_e_n[i]
    Rece_re = [0] * len(f)
    for i in range(len(Rece_re)):
        Rece_re[i] = f[i] / f_e[i]
    diff = [0] * len(Rece_re)
    for i in range(len(diff)):
        diff[i] = Rece_re[i] - Back_re[i]
    result = ifft(diff)
    back_s = ifft(Back_re)
    return result, back_s


def amp2db(data):
    for i in range(len(data)):
        data[i] = 10 * math.log10(data[i])
    return data


def x_y_cal(height, theta, dis):
    k = np.tan(theta)
    if k > 0:
        flag = 1
    else:
        flag = -1
    # print(dis, height)
    x = flag * np.sqrt(dis ** 2 - height ** 2) / (np.sqrt(1 + k ** 2))
    y = k * x
    return x, y


def search_sonar_line_min_dis_theta(x, y, x_1, y_1, z, z_1, target_theta_1):
    k_1 = np.tan(target_theta_1)
    if x == x_1:
        A = (z - z_1) / (y - y_1)
        x_min_1 = x
        y_min_1 = k_1 * x_min_1
        z_min_1 = A * (y_min_1 - y_1) + z_1
    else:
        A = (y - y_1) / (x - x_1)
        B = (z - z_1) / (x - x_1)
        x_min_1 = (y - A * x) / (k_1 - A)
        z_min_1 = B * (x_min_1 - x) + z
        y_min_1 = k_1 * x_min_1
    # target_1 = point.point(x_min_1+sonar.x, y_min_1+sonar.y, z_min_1+sonar.z).r

    target_1 = point.point(x_min_1, y_min_1, z_min_1).r
    return target_1


def search_theta_ground_truth(point_set, theta_target, sonar=None, Flag=False):
    for i in range(len(point_set) - 1):
        if Flag:
            point_set[i].sonar_axis_convert(sonar)
            ##print(point_set[i].sonar_theta)
        if (point_set[i].sonar_theta - theta_target) * (point_set[i + 1].sonar_theta - theta_target) <= 0:
            if point_set[i].sonar_theta == theta_target:
                return point_set[i].sonar_r, point_set[i].sonar_theta
            elif point_set[i + 1].sonar_theta == theta_target:
                return point_set[i + 1].sonar_r, point_set[i + 1].sonar_theta
            else:
                return point_set[i].sonar_r, point_set[i].sonar_theta
    if Flag:
        point_set[len(point_set) - 1].sonar_axis_convert(sonar)
    # print(point_set[0].sonar_theta, point_set[len(point_set) - 1].sonar_theta)
    if point_set[len(point_set) - 1].sonar_theta == theta_target:
        return point_set[len(point_set) - 1].sonar_r, point_set[len(point_set) - 1].sonar_theta
    else:
        return -1


def check_length(x, y, z, x1, y1, z1, threshold_low, threshold_high):
    real_len = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
    if real_len < threshold_low or real_len > threshold_high:
        return False
    else:
        return True


def point_set_cross(dataset, dataset2, resolv=0.015):
    target_dataset = np.array([])
    flag = False
    for i in range(len(dataset)):
        x_temp = dataset[i][0]
        y_temp = dataset[i][1]
        value_temp = dataset[i][2]
        for j in range(len(dataset2)):
            x_temp_2 = dataset2[j][0]
            y_temp_2 = dataset2[j][1]
            if np.sqrt((x_temp - x_temp_2) ** 2 + (y_temp - y_temp_2) ** 2) <= resolv:
                if not flag:
                    target_dataset = np.array([[x_temp, y_temp, value_temp]])
                    flag = True
                else:
                    target_dataset = np.append(target_dataset, np.array([[x_temp, y_temp, value_temp]]), axis=0)
                # print(target_dataset)
                break

    len_target_set = len(target_dataset)
    #print(len_target_set)
    for i in range(len(dataset2)):
        x_temp_2 = dataset2[i][0]
        y_temp_2 = dataset2[i][1]
        value_temp_2 = dataset2[i][2]
        for j in range(len_target_set):
            # print(i, j, target_dataset[j][1])
            if np.sqrt((x_temp_2 - target_dataset[j][0]) ** 2 + (y_temp_2 - target_dataset[j][1]) ** 2) <= resolv:
                target_dataset = np.append(target_dataset, np.array([[x_temp_2, y_temp_2, value_temp_2]]), axis=0)
                break
    #print(len(target_dataset))
    result = sorted(target_dataset, key=lambda x: (x[0], x[1]))
    return result


def weight_recalculate(result):
    sum_weight = 0
    for i in range(len(result)):
        sum_weight += result[i][2]
    x_sum = 0
    y_sum = 0
    for i in range(len(result)):
        x_sum += (result[i][0] * result[i][2])/sum_weight
        y_sum += (result[i][1] * result[i][2])/sum_weight
    #x_sum /= len(result)
    #y_sum /= len(result)
    return x_sum, y_sum
