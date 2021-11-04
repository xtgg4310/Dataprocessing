# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.signal import stft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import sys

sys.setrecursionlimit(10000)


# Press the green button in the gutter to run the script.
def cal_sampling_rate(time1, time2, data):
    return 1000 / ((time2 - time1) / len(data))


def cal_scal(data):
    length_scal = len(data)
    re1 = [0] * length_scal
    for i in range(0, length_scal):
        re1[i] = np.sqrt(np.power(data["acc_x"][i], 2) + np.power(data["acc_y"][i], 2) + np.power(data["acc_z"][i], 2))
    return re1


def fft_file_doc(data, start1, length):
    re1 = [0] * length
    for i in range(0, length):
        re1[i] = np.sqrt(np.power(data["acc_x"][i + start1], 2) + np.power(data["acc_y"][i + start1], 2) + np.power(
            data["acc_z"][i + start1] - 9.8, 2))
    record_x = data["acc_x"].values.tolist()
    record_y = data["acc_y"].values.tolist()
    record_z = data["acc_z"].values.tolist()
    x = abs(fft(record_x[start1:start1 + length - 1]) / length)
    y = abs(fft(record_y[start1:start1 + length - 1]) / length)
    z = abs(fft(record_z[start1:start1 + length - 1]) / length)
    r1 = abs(fft(re1) / length)
    re_x = 2 * x[range(int(length / 2))]
    re_y = 2 * y[range(int(length / 2))]
    re_z = 2 * z[range(int(length / 2))]
    re_r1 = 2 * r1[range(int(length / 2))]
    rate_temp = cal_sampling_rate(data['timestamp_acc'][0], data['timestamp_acc'][len(data) - 1], data)
    _rate = [0] * (int(length / 2))
    for i in range(0, int(length / 2)):
        _rate[i] = i * rate_temp / length;
    return re_x, re_y, re_z, re1, re_r1, _rate, rate_temp;


def fft_file(data, data_x, data_y, data_z, scal_1, start1, length):
    # re1 = [0] * length
    # for i in range(0, length):
    # re1[i] = np.sqrt(np.power(data_x[i + start1], 2) + np.power(data_y[i + start1], 2) + np.power(
    #        data_z[i + start1], 2))

    x = abs(fft(data_x[start1:start1 + length - 1]) / length)
    y = abs(fft(data_y[start1:start1 + length - 1]) / length)
    z = abs(fft(data_z[start1:start1 + length - 1]) / length)
    r1 = abs(fft(scal_1) / length)
    re_x = 2 * x[range(int(length / 2))]
    re_y = 2 * y[range(int(length / 2))]
    re_z = 2 * z[range(int(length / 2))]
    re_r1 = 2 * r1[range(int(length / 2))]
    rate_temp = cal_sampling_rate(data['timestamp_acc'][0], data['timestamp_acc'][len(data) - 1], data)
    _rate = [0] * (int(length / 2))
    for i in range(0, int(length / 2)):
        _rate[i] = i * rate_temp / length;
    return re_x, re_y, re_z, re_r1, _rate, rate_temp;


def signal_draw_save_doc(data, data1, r, r1, start, start1, length):
    plt.subplot(411)
    plt.plot(data['acc_x'][start:start + length], 'b')
    plt.plot(data1['acc_x'][start1:start1 + length], 'r')
    plt.title('x-axis')

    plt.subplot(412)
    plt.plot(data['acc_y'][start:start + length], 'b')
    plt.plot(data1['acc_y'][start1:start1 + length], 'r')
    plt.title('y-axis')

    plt.subplot(413)
    plt.plot(data['acc_z'][start:start + length], 'b')
    plt.plot(data1['acc_z'][start1:start1 + length], 'r')
    plt.title('z-axis')

    plt.subplot(414)
    plt.plot(r, 'b')
    plt.plot(r1, 'r')
    plt.title('scalar')

    plt.tight_layout()
    plt.savefig('test_result_signal.jpg')

    plt.show()


def signal_draw_save(data_x, data_y, data_z, data1_x, data1_y, data1_z, r, r1, start, start1, length):
    plt.subplot(411)
    plt.plot(data_x[start:start + length], 'b')
    plt.plot(data1_x[start1:start1 + length], 'r')
    plt.title('x-axis')

    plt.subplot(412)
    plt.plot(data_y[start:start + length], 'b')
    plt.plot(data1_y[start1:start1 + length], 'r')
    plt.title('y-axis')

    plt.subplot(413)
    plt.plot(data_z[start:start + length], 'b')
    plt.plot(data1_z[start1:start1 + length], 'r')
    plt.title('z-axis')

    plt.subplot(414)
    plt.plot(r, 'b')
    plt.plot(r1, 'r')
    plt.title('scalar')

    plt.tight_layout()
    plt.savefig('test_result_signal.jpg')

    plt.show()


def fft_draw_save(x, y, z, r, x1, y1, z1, r1, rate1, rate2, rate, rate_a):
    plt.subplot(421)
    plt.plot(rate1[int(len(rate1) / rate) - 1:len(rate1) - 1], x[int(len(rate1) / rate) - 1:len(rate1) - 1], 'b')
    plt.title('x-axis_nocage')

    plt.subplot(422)
    plt.plot(rate2[int(len(rate2) / rate_a) - 1:len(rate2) - 1], x1[int(len(rate2) / rate_a) - 1:len(x1) - 1], 'r')
    plt.title('x-axis_cage')

    plt.subplot(423)
    plt.plot(rate1[int(len(rate1) / rate) - 1:len(rate1) - 1], y[int(len(rate1) / rate) - 1:len(y) - 1], 'b')
    plt.title('y-axis_nocage')

    plt.subplot(424)
    plt.plot(rate2[int(len(rate2) / rate_a) - 1:len(rate2) - 1], y1[int(len(rate2) / rate_a) - 1:len(y1) - 1], 'r')
    plt.title('y-axis_cage')

    plt.subplot(425)
    plt.plot(rate1[int(len(rate1) / rate) - 1:len(rate1) - 1], z[int(len(rate1) / rate) - 1:len(z) - 1], 'b')
    plt.title('z-axis_nocage')

    plt.subplot(426)
    plt.plot(rate2[int(len(rate2) / rate_a) - 1:len(rate2) - 1], z1[int(len(rate2) / rate_a) - 1:len(z1) - 1], 'r')
    plt.title('z-axis_nocage')

    plt.subplot(427)
    plt.plot(rate1[int(len(rate1) / rate) - 1:len(rate1) - 1], r[int(len(rate1) / rate) - 1:len(r) - 1], 'b')
    plt.title('scalar_nocage')

    plt.subplot(428)
    plt.plot(rate2[int(len(rate2) / rate_a) - 1:len(rate2) - 1], r1[int(len(rate2) / rate_a) - 1:len(r1) - 1], 'r')
    plt.title('scalar_cage')

    plt.tight_layout()
    plt.savefig('test_result_frequency.jpg')

    plt.show()


def high_pass(doc1):
    rate_temp = cal_sampling_rate(doc1['timestamp_acc'][0], doc1['timestamp_acc'][len(doc1) - 1], doc1)
    sos = signal.butter(2, Wn=4 / rate_temp, btype='highpass', analog=False, output='sos')
    new_x = signal.sosfilt(sos, doc1["acc_x"])
    new_y = signal.sosfilt(sos, doc1["acc_y"])
    new_z = signal.sosfilt(sos, doc1["acc_z"])
    return new_x, new_y, new_z


def scalar_high_pass(r, doc1):
    rate_temp = cal_sampling_rate(doc1['timestamp_acc'][0], doc1['timestamp_acc'][len(doc1) - 1], doc1)
    sos = signal.butter(10, Wn=4 / rate_temp, btype='highpass', analog=False, output='sos')
    new_scalar = signal.sosfilt(sos, r)
    return new_scalar


def stft_calculate(data, rate, start, end):
    f, t, Zxx = stft(data['acc_z'][start:end], rate, nperseg=500)
    print(np.abs(Zxx))
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz] z-axis')
    plt.xlabel('Time [sec]')
    plt.show()


def stft_scalar_calculate(scalar, rate):
    f, t, Zxx = stft(scalar, rate, nperseg=500)
    print(np.abs(Zxx))
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
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


def txt_to_matrix(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 4))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(' ')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line[:]
        row += 1

    return datamat


def pre_process(doc1):
    for i in range(0, len(doc1)):
        for j in range(0, 4):
            doc1[i][j] = (doc1[i][j] / 16384) * 9.8
    scalar = [0] * len(doc1);
    for i in range(0, len(doc1)):
        scalar[i] = np.sqrt(np.power(doc1[i][0], 2) + np.power(doc1[i][1], 2), np.power(doc1[i][3], 2))


def cal_corr_len(scal, scal1, start, start2, length):
    return np.corrcoef(scal[start:start + length], scal1[start2:start2 + length])


def cal_corr_len_win(data, data1, scal, scal1, start, length_win):
    length = min(len(scal), len(scal1))
    i = start
    j = start
    result = []
    while i + length_win < length and j + length_win < length:
        i, j = search_the_same_time(data, data1, i, j)
        result.append(cal_corr_len(scal, scal1, i, j, length_win)[0][1])
        i = i + 50
        j = j + 50

    return result


def align_time(data, data2, start, start2):
    if (data['timestamp_acc'][start] == data2['timestamp_acc'][start2] or (
            data['timestamp_acc'][start] == data2['timestamp_acc'][start2] + 1) or
            data['timestamp_acc'][start] == data2['timestamp_acc'][start2] - 1):
        return True
    else:
        return False


def search_the_same_time(data, data2, start, start2):
    if align_time(data, data2, start, start2):
        return start, start2;
    else:
        if data['timestamp_acc'][start] < data2['timestamp_acc'][start2]:
            for i in range(start + 1, len(data)):
                if align_time(data, data2, i, start2):
                    return i, start2
                else:
                    if data['timestamp_acc'][i] > data2['timestamp_acc'][start2]:
                        return -1, -1
        else:
            for i in range(start2 + 1, len(data2)):
                if align_time(data, data2, start, i):
                    return start, i
                else:
                    if data['timestamp_acc'][start] < data2['timestamp_acc'][i]:
                        return -1, -1


if __name__ == '__main__':
    doc = pd.read_csv(r"your path")
    doc1 = pd.read_csv(r"anthor path")
    s1 = cal_scal(doc)
    s2 = cal_scal(doc1)
    # out = np.corrcoef(s1[500:12000], s2[500:12000])
    # print(out)
    re = cal_corr_len_win(doc, doc1, s1, s2, 500, 1000)
    plt.plot(re)
    plt.show()
    # time, time2 = search_the_same_time(doc, doc1, 0, 0);
    # length = 1000;
    # time = search_the_same_time(doc, doc1, time + 100, time2 + 100);
    # print(time)
    # record1 = doc1.values.tolist()
    # x, y, z, scal, r, rate, rate_a = fft_file_doc(doc1, 500, 3000)
    # x1, y1, z1, scal_1, r1, rate1, rate_a1 = fft_file_doc(doc2, 653, 3000)
    # signal_draw_save_doc(doc1, doc2, scal, scal_1, 500, 653, 3000)
    # doc1_x, doc1_y, doc1_z = high_pass(doc1)
    # doc2_x, doc2_y, doc2_z = high_pass(doc2)
    # scalar_1 = scalar_high_pass(scal, doc1)
    # scalar_2 = scalar_high_pass(scal_1, doc2)
    # x, y, z, r, rate, rate_a = fft_file(doc1, doc1_x, doc1_y, doc1_z,scalar_1, 500, 6000)
    # x1, y1, z1, r1, rate1, rate_a1 = fft_file(doc2, doc2_x, doc2_y, doc2_z,scalar_2, 6000, 6000)
    # signal_draw_save(doc1_x, doc1_y, doc1_z, doc2_x, doc2_y, doc2_z, scalar_1, scalar_2, 500, 6000, 6000)

    # fft_draw_save(x, y, z, r, x1, y1, z1, r1, rate, rate1, rate_a, rate_a1)

    # stft_calculate(doc1, rate_a,4000,10000)
    # stft_calculate(doc2, rate_a1,4000,10000)
    # stft_scalar_calculate(scalar_1, rate_a)
    # stft_scalar_calculate(scalar_2, rate_a1)
