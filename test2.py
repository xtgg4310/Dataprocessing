import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import optimizer as op
import point
import simulation as sl

if __name__ == '__main__':
    # data, x, y = txt2matrix.txt_to_matrix_optimization(
    #    r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_-4,2,3_-2,8,-2_20_200_0.8.txt")

    # plt.scatter(x, y, color='blue')
    # plt.scatter(3, -2, color='red')
    # plt.scatter(-3, 2, color='yellow')
    # plt.grid()
    # plt.show()
    # fig = plt.figure()
    # ax1 = Axes3D(fig)
    fig = plt.figure()
    ax1 = Axes3D(fig)

    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    Sonar1 = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-3, 8, 1)
    end = point.point(-4, 9, 1.5)

    theta_target = (end.theta - start.theta) / 3 + start.theta
    theta_target1 = (end.theta - start.theta) * 2 / 3 + start.theta
    height = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    height_end = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    dis1 = [0] * len(height)

    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    scan_result = Sonar.get_result()
    # target = line_target.get_point()
    '''
    re = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target)
    x_r, y_r, z_r = sl.list_position_ndarray(re)
    ax1.scatter3D(Sonar.x, Sonar.y, Sonar.z, color="blue")
    ax1.plot3D(x_t, y_t, z_t, label="4", color="yellow")
    ax1.plot3D(x_r, y_r, z_r, "green")
    result = Sonar.scan_line(line_target, np.pi)

    if len(result) != 0:
        for i in range(len(result)):
            dic = sl.surface_point(result[i][0], result[i][1], sonar=Sonar)
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "orange", alpha=0.01)
    line_target1 = sl.Line(start, end, 1000)
    Sonar1.clear_result()
    Sonar1.scaning_result_simple(line_target1)
    target1 = line_target1.get_point()
    result1 = Sonar1.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target1)
    x_r, y_r, z_r = sl.list_position_ndarray(result1)
    ax1.scatter3D(Sonar1.x, Sonar1.y, Sonar1.z, color="lightgreen")
    ax1.plot3D(x_t, y_t, z_t, label="1", color="purple")
    ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar1.scan_line(line_target1, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1], sonar=Sonar1)
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "red", alpha=0.01)
    plt.show()
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
