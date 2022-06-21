import random
import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import optimizer as op
import point
import simulation as sl


def K_zx(x, y):
    return (2 * np.cos(x) - np.cos(y)) / (1.8 * np.sin(x) - 0.7 * np.cos(y))


def A_zk(x, y):
    return (1.4 * np.sin(y) * np.cos(x) - 1.8 * np.sin(x) * np.cos(y)) / (1.8 * np.sin(x) - 0.7 * np.cos(y))


def function_k_A(x, y):
    return np.sqrt((K_zx(x, y)) ** 2 + 2 * K_zx(x, y) * A_zk(x, y) + A_zk(x, y) ** 2 + 10)


if __name__ == '__main__':
    # a = random.randint(1, 10)
    # print(a)
    # data, x, y = txt2matrix.txt_to_matrix_optimization(
    #    r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_-4,2,3_-2,8,-2_20_200_0.8.txt")

    # plt.scatter(x, y, color='blue')
    # plt.scatter(3, -2, color='red')
    # plt.scatter(-3, 2, color='yellow')
    # plt.grid()
    # plt.show()
    fig = plt.figure()
    ax1 = Axes3D(fig)

    x_1 = np.linspace(0, np.pi, 300)
    y_1 = np.linspace(0, np.pi, 300)
    x, y = np.meshgrid(x_1, y_1)
    z = np.zeros((len(x_1), len(y_1)))
    for i in range(len(x_1)):
        for j in range(len(y_1)):
            z[i][j] = function_k_A(x[i][j], y[i][j])
            if z[i][j] > 1000:
                z[i][j] = 1000
    ax1.plot_surface(x, y, z)
    plt.show()
    '''
    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    Sonar1 = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-1.974015767970707, 5.806383687370714, 0.8454659008490583)
    end = point.point(-0.14024426907606635, 4.410684022700307, 0.7704260625197861)

    line_target = sl.Line(start, end, 1000)
    Sonar1.scaning_result_simple(line_target)
    scan_result2 = Sonar1.get_result()
    scan_result2[0].sonar_axis_convert(Sonar1)
    scan_result2[len(scan_result2) - 1].sonar_axis_convert(Sonar1)
    op.set_global_parameter2(scan_result2[0].sonar_theta, scan_result2[len(scan_result2) - 1].sonar_theta,
                             scan_result2[0].sonar_r, scan_result2[len(scan_result2) - 1].sonar_r, scan_result2,
                             np.pi * 2 / 3, Sonar1, True)
    x_2 = np.linspace(op.h_min_2[0], op.h_max_2[0], 201)
    print(op.h_min_2[0], op.h_max_2[0])
    print(op.h_min_2[1], op.h_max_2[1])
    y_2 = np.linspace(op.h_min_2[1], op.h_max_2[1], 201)
    x2, y2 = np.meshgrid(x_2, y_2)
    value2 = np.zeros((len(x_2), len(y_2)))
    dic_result2 = {}
    index_2 = 0
    for i in range(len(x_2)):
        for j in range(len(y_2)):
            value2[i][j] = op.function_to_target_sonar2(x2[i][j], y2[i][j])
            # print(value2[i][j])
            if value2[i][j] > -5:
                print(x2[i][j], y2[i][j])
                dic_result2.update({index_2: [x2[i][j], y2[i][j]]})
                index_2 += 1
    # print(op.function_to_target_sonar2(line_target.start.z, line_target.end.z))

    for i in range(len(dic_result2)):
        plt.scatter(dic_result2[i][0], dic_result2[i][1],color="blue")
    if line_target.start.z >= 0:
        plt.scatter(line_target.start.z, line_target.end.z, color="red")
    else:
        plt.scatter(-1 * line_target.start.z, -1 * line_target.end.z, color="red")
    plt.show()
    '''
    '''
    result = {}
    op.set_global_parameter(scan_result[0].theta,
                            scan_result[len(scan_result) - 1].theta,
                            scan_result[0].r,
                            scan_result[len(scan_result) - 1].r, scan_result,
                            np.pi / 2)

    h_min_1 = op.h_start_min
    h_max_1 = op.h_start_max
    h_min_2 = op.h_end_min
    h_max_2 = op.h_end_max
    time1 = time.time()
    count = 0
    x = np.linspace(h_min_1, h_max_1, 101)
    y = np.linspace(h_min_2, h_max_2, 101)
    x1, y1 = np.meshgrid(x, y)
    value = np.zeros((len(x), len(y)))
    print(x1)
    print(y1)
    for i in range(len(x)):
        for j in range(len(y)):
            value[i][j] = op.function_to_target(x1[i][j], y1[i][j])

    value = np.array(value)
    value = (value - np.min(value)) / (np.max(value) - np.min(value))
    index_max = np.argmax(value)
    x_max = int(np.floor(index_max / 101))
    y_max = index_max % 101
    print(np.max(value), value[x_max][y_max])
    dic_result = {}
    result_index = 0

    for i in range(len(x)):
        for j in range(len(y)):
            if value[i][j] > 0.99:
                dic_result.update({result_index: [x1[i][j], y1[i][j], value[i][j]]})
                result_index += 1

    time2 = time.time()
    print(time2 - time1)
    threshold = [0.99, 0.97, 0.95, 0.93, 0.9, 0.7]
    filer_count = 0
    while filer_count < 6:
        x_temp = np.zeros(len(dic_result))
        y_temp = np.zeros(len(dic_result))
        value_temp = np.zeros(len(dic_result))
        for i in range(len(dic_result)):
            x_temp[i] = dic_result[i][0]
            y_temp[i] = dic_result[i][1]
            value_temp[i] = dic_result[i][2]
        min_v = np.min(value_temp)
        max_v = np.max(value_temp)
        for i in range(len(value_temp)):
            value_temp[i] = 1.0 * (value_temp[i] - min_v) / (max_v - min_v)
        dic_result.clear()
        index = 0
        # print(len(dic_result))
        for i in range(len(x_temp)):
            if value_temp[i] > threshold[filer_count]:
                dic_result.update({index: [x_temp[i], y_temp[i], value_temp[i]]})
                if filer_count == 5:
                    plt.scatter(x_temp[i], y_temp[i], color='blue')
                index += 1
        print(len(dic_result))
        if filer_count == 5:
            plt.scatter(5, 4.6, color="green")
            plt.xlabel("h_start")
            plt.ylabel("h_end")
            plt.show()
        if 30 <= len(dic_result) < 60:
            break
        filer_count += 1

    '''
    '''
    index = 0
    while index < 10:
        optimizer_re, optimizer_record = op.optimizer_line_pose(scan_result[0].theta,
                                                                scan_result[len(scan_result) - 1].theta,
                                                                scan_result[0].r,
                                                                scan_result[len(scan_result) - 1].r, scan_result,
                                                                np.pi / 2)
        result.update({index: optimizer_re})
        #print(type(optimizer_re['params']['h_1']))
        index += 1
    '''
    # print(str(result[0]['params']['h_1']) + ',' + str(result[0]['params']['h_2']) + '\n')
    # fd = open(r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_3_2_5_180_2.txt", "w")
    # for i in range(len(result)):
    #    print(i)
    #    fd.writelines(str(result[i]['params']['h_1']) + ',' + str(result[i]['params']['h_2']) + '\n')
    # fd.close()
