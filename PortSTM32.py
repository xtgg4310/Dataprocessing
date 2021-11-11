import serial
from serial import EIGHTBITS
import time


def read_port(length=100000):
    number = length * [0]
    port1 = serial.Serial('COM35', 2000000, EIGHTBITS)
    i = 0
    currentTime = time.time()
    while i < length:
        number[i] = port1.read(600000)
        i += 1
    endTime = time.time()
    port1.close()
    return number, currentTime, endTime


def decode_port(num):
    l = len(num)
    i = 0
    strnum = l * [0]
    while i < l:
        strnum[i] = num[i].hex()
        i += 1
    return strnum


def digit_segment(number):
    l = len(number)
    index = 0
    new_code = ""
    while index < l:
        if index + 4 > l:
            break
        # print(number[index:index + 4])
        record = bin(int(number[index:index + 4], 16))[2:].zfill(16)
        # print(record)
        head1, head2 = get_head(record)
        if valid_package(head1, head2) == 1:
            new_code = new_code + number[index:index + 4]
            index = index + 4
        elif valid_package(head1, head2) == 0 or valid_package(head1, head2) == -1:
            index = index + 2
        else:
            index = index + 4
    return new_code


def check_bit(num):
    length = len(num)
    index = 0
    count = 0
    while index < length:
        if num[index] == '1':
            count += 1
        index += 1
    return count % 2 == 0


def get_head(num):
    return num[0], num[8]


def Sixteen2Two(num, total_len=16):
    # temp = decode_port(num)
    number = bin(int(num, 16))[2:].zfill(total_len)
    # check_flag = False
    check_flag = check_bit(number)
    new_num = 12 * [0]
    i = 0
    index = 0
    while i < 16:
        if i == 8 or i == 0 or i == 9 or i == 1:
            i += 1
            continue
        else:
            new_num[index] = number[i]
            i += 1
            index += 1

    if check_flag:
        # new_num[1] = '0'
        result = ''.join(new_num)
        return check_flag, result
    else:
        # new_num[1] = '0'
        result = ''.join(new_num)
        return check_flag, result


def Two2Int(num):
    flag, num_temp = Sixteen2Two(num)
    # print(flag,num_temp)
    if flag:
        # print(num_temp)
        return -1
    else:
        return int(num_temp, 2)


def get_all_data(record):
    length = len(record)
    count = 0
    for i in range(0, length):
        count += (len(record[i]) / 4)

    data = int(count) * [0]
    index = 0
    record_index = 0
    while index < length:
        i = 0
        while i + 4 <= len(record[index]):
            data[record_index] = Two2Int(record[index][i:i + 4])
            # print(record[index][i:i + 4])
            i += 4
            record_index += 1
        index += 1
    return data


def error_rate(data):
    total = len(data)
    count = 0
    index = 0
    while index < total:
        if data[index] == -1:
            count += 1
            index += 1
        else:
            index += 1
            continue
    print(count / total)


def valid_package(head1, head2):
    if head1 == '1' and head2 == '0':
        return 1
    elif head1 == '1' and head2 == '1':
        return 0
    elif head1 != '1' and head2 == '1':
        return -1
    elif head1 != '1' and head2 == '0':
        return -2
