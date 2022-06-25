import numpy as np


def txt_to_matrix_optimization(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数
    datamat = np.zeros((rows, 2))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line
        row += 1
    x = np.ndarray([])
    y = np.ndarray([])
    for i in range(len(datamat)):
        x = np.append(x, datamat[i][0])
        y = np.append(y, datamat[i][1])
    return datamat, x, y


def txt_to_matrix_speed_time(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数
    row = 0
    datamat = np.zeros((rows, 1))
    for line in lines:
        if row == 0:
            continue
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        # print(line)
        datamat[row, :] = line[0]
        row += 1
    # print(time)
    return datamat


def txt_to_matrix_feature(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数
    row = 0
    datamat = np.zeros((rows, 1))
    time = np.zeros((rows, 1))
    for line in lines:
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        # print(line)
        # print(line[0])
        datamat[row, :] = line[0]
        time[row, :] = line[1]
        row += 1
    # print(time)
    return datamat, time


def txt_to_matrix_DAQ_RPI(filename, time_flag=True):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数
    time = np.zeros((1, 2))
    if time_flag:
        datamat = np.zeros((rows - 2, 1))  # 初始化矩阵
    else:
        datamat = np.zeros((rows - 1, 1))
    row = 0
    record = 0
    for line in lines:
        if row == 0:
            row += 1

            continue
        if row == rows - 1:
            line = line.strip().split(',')
            if time_flag:
                time = line
            else:
                datamat[record, :] = line[0]
                record += 1
            break
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        # print(line)
        datamat[record, :] = line[0]
        row += 1
        record += 1
    # print(time)
    if time_flag:
        return datamat, time
    else:
        return datamat


def txt_to_matrix(filename, length):
    file = open(filename)
    lines = file.readlines()
    # rows = length  # 文件行数

    datamat = np.zeros((length, 2))  # 初始化矩阵
    row = 0
    record = 0
    for line in lines:
        if row == 1 or row == 0:
            row += 1
            continue
        if record >= length:
            break
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        datamat[record, :] = line[1:]
        row += 1
        record += 1
    return datamat


def txt_to_matrix_notime(filename, length):
    file = open(filename)
    lines = file.readlines()
    # rows = length  # 文件行数

    datamat = np.zeros((length, 1))  # 初始化矩阵
    row = 0
    record = 0
    for line in lines:
        if row == 1 or row == 0:
            row += 1
            continue
        if record >= length:
            break
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        datamat[record, :] = line[1:]
        row += 1
        record += 1
    return datamat


def txt_to_matrix_dis(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 2))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line
        row += 1
    return datamat


def txt_to_matrix_realtime(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 2))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(' ')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line
        row += 1
    return datamat


def txt_to_matrix_sonartime(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 1))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(' ')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line
        row += 1
    return datamat


def txt_to_matrix_stm32(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 2))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(' ')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line
        row += 1
    return datamat


def txt_to_matrix_seg(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 2))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line
        row += 1
    return datamat
