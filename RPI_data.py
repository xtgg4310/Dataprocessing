def pre_process_RPI(doc1):
    scalar = [0] * len(doc1)
    timestamp = [0] * len(doc1)
    for i in range(0, len(doc1)):
        scalar[i] = doc1[i][1]
        timestamp[i] = doc1[i][0]
    rate = len(doc1) / (timestamp[len(doc1) - 1] - timestamp[0])
    return scalar, timestamp, rate


def pre_process_time(time):
    time_new = [0] * len(time)
    count = 1
    diff = 0
    for i in range(0, len(time) - 1):
        if time[i] == time[i + 1] and i != len(time) - 2:
            count += 1
            continue
        elif i != len(time) - 2:
            temp = time[i]
            diff = (time[i + 1] - time[i]) / count
            for j in range(0, count):
                time_new[i - count + 1 + j] = temp + diff * j
            count = 1
        else:
            temp = time[i]
            for j in range(0, count):
                time_new[i - count + 1 + j] = temp + diff * j
    time_new[len(time) - 1] = time[len(time) - 1] + diff * count
    return time_new


def pre_relative_time(time):
    temp = time[0]
    for i in range(0, len(time)):
        time[i] = time[i] - temp
    return time


def search_pulse(data_time, length):
    data = [0] * length
    index = 0
    for i in range(0, len(data_time)):
        if i == 0 and data_time[i][0] == 1:
            data[index] = data_time[i][1]
            index += 1
        elif i > 0 and data_time[i - 1][0] == 0.0 and data_time[i][0] == 1.0:
            data[index] = data_time[i][1]
            index += 1
    return data


def error_dis_measure(data, dis_real):
    error = 0
    count = 0
    for i in range(0, len(data)):
        if data[i][1] >= dis_real - 5.0 or data[i][1] <= dis_real + 5.0:
            error = error + data[i][1]
            count += 1
    error /= count
    return error - dis_real
