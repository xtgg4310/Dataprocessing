import random
import time
import simulation as sl
import point
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import signal_process as sg
import optimizer as op
from matplotlib import cm

if __name__ == '__main__':
    # fig = plt.figure()
    # ax1 = Axes3D(fig)
    # fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-3.5, 18.5, 4.8)
    end = point.point(-4, 19, 5)

    start1 = point.point(-5, 10, -np.sqrt(155))
    end1 = point.point(18, 16, -12)

    start2 = point.point(-4, 8, -np.sqrt(200))
    end2 = point.point(18, 16, -12)

    start3 = point.point(-3, 6, -np.sqrt(235))
    end3 = point.point(18, 16, -12)

    start4 = point.point(-2, 4, -np.sqrt(260))
    end4 = point.point(18, 16, -12)

    dis_array = [start.r, end.r, start1.r, end1.r, start2.r, end2.r, start3.r, end3.r, start4.r, end4.r]
    dis = np.max(dis_array)
    theta_target = (end.theta - start.theta) / 3 + start.theta
    theta_target1 = (end.theta - start.theta) * 2 / 3 + start.theta
    height = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    height_end = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    dis1 = [0] * len(height)

    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    scan_result = Sonar.get_result()

    op.set_global_parameter(scan_result[0].theta,
                            scan_result[len(scan_result) - 1].theta,
                            scan_result[0].r,
                            scan_result[len(scan_result) - 1].r, scan_result,
                            np.pi / 2)

    h_min_1 = op.h_start_min
    h_max_1 = op.h_start_max
    h_min_2 = op.h_end_min
    h_max_2 = op.h_end_max

    # print(x[50], y[50])
    time1 = time.time()
    count = 0
    x = np.linspace(h_min_1, h_max_1, 101)
    y = np.linspace(h_min_2, h_max_2, 101)
    x1, y1 = np.meshgrid(x, y)
    value = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            value[i][j] = op.function_to_target(x1[i][j], y1[i][j])

    value = np.array(value)
    value = (value - np.min(value)) / (np.max(value) - np.min(value))

    dic_result = {}
    result_index = 0

    for i in range(len(x)):
        for j in range(len(y)):
            if value[i][j] > 0.99:
                dic_result.update({result_index: [x1[i][j], y1[i][j], value[i][j]]})
                result_index += 1

    time2 = time.time()
    print(time2 - time1)
    threshold = [0.99, 0.97, 0.95, 0.95, 0.95, 0.6]
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
                if filer_count == 1:
                    plt.scatter(x_temp[i], y_temp[i], color='blue')
                index += 1
        print(len(dic_result))
        if filer_count == 1:
            plt.scatter(4.8, 5, color="green")
            plt.xlabel("h_start")
            plt.ylabel("h_end")
            plt.show()
        if 30 <= len(dic_result) < 100:
            break
        filer_count += 1

        # print(op.function_to_target(0, 4))
    # result = {}
    # index = 0
    # while index < 30:
    #    optimizer_re, optimizer_record = op.optimizer_line_pose(scan_result[0].theta,
    #                                                            scan_result[len(scan_result) - 1].theta,
    #                                                            scan_result[0].r,
    #                                                            scan_result[len(scan_result) - 1].r, scan_result,
    #                                                            np.pi / 2)
    #    result.update({index: optimizer_re})
    #    print(index)
    #    index += 1
    # print(str(result[0]['params']['h_1']) + ',' + str(result[0]['params']['h_2']) + '\n')
    # fd = open(r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_-2,8,2_-3,12,1_2_180_3.txt", "w")
    # for i in range(len(result)):
    #    print(i)
    #    fd.writelines(str(result[i]['params']['h_1']) + ',' + str(result[i]['params']['h_2']) + '\n')
    # fd.close()
    '''
    k = 3
    for i in range(len(height)):
        x, y = sg.x_y_cal(height[i], start.theta, start.r)
        x_1, y_1 = sg.x_y_cal(height_end[k], end.theta, end.r)
        # print(x_1, y_1, x_1 ** 2 + y_1 ** 2 + height[k] ** 2)
        dis1[i] = sg.search_sonar_line_min_dis_theta(x, y, x_1, y_1, height[i], height_end[k], theta_target)
    plt.plot(height, dis1, label=str(height_end[k]))
    plt.show()
    plt.close()
    count = 0
    while count < len(height_end):
        k = count
        for i in range(len(height)):
            x, y = sg.x_y_cal(height[i], start.theta, start.r)
            x_1, y_1 = sg.x_y_cal(height_end[k], end.theta, end.r)
            # print(x_1, y_1, x_1 ** 2 + y_1 ** 2 + height[k] ** 2)
            dis1[i] = sg.search_sonar_line_min_dis_theta(x, y, x_1, y_1, height[i], height_end[k], theta_target)
        plt.plot(height, dis1, label=str(height_end[count]))
        count += 1
    plt.legend()
    plt.show()
    
    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    target = line_target.get_point()
    
    re = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target)
    x_r, y_r, z_r = sl.list_position_ndarray(re)
    ax1.scatter3D(Sonar.x, Sonar.y, Sonar.z, color="blue")
    ax1.plot3D(x_t, y_t, z_t, label="4", color="yellow")
    # ax1.plot3D(x_r, y_r, z_r, "green")
    result = Sonar.scan_line(line_target, np.pi)

    if len(result) != 0:
        for i in range(len(result)):
            dic = sl.surface_point(result[i][0], result[i][1])
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "orange", alpha=0.1)
                
    
    line_target1 = sl.Line(start1, end1, 1000)
    Sonar.clear_result()
    Sonar.scaning_result_simple(line_target1)
    target1 = line_target1.get_point()
    result1 = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target1)
    x_r, y_r, z_r = sl.list_position_ndarray(result1)
    ax1.plot3D(x_t, y_t, z_t, label="1", color="purple")
    # ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target1, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1])
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "red", alpha=0.1)

    line_target2 = sl.Line(start2, end2, 1000)
    Sonar.clear_result()
    Sonar.scaning_result_simple(line_target2)
    target1 = line_target2.get_point()
    result1 = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target1)
    x_r, y_r, z_r = sl.list_position_ndarray(result1)
    ax1.plot3D(x_t, y_t, z_t, label="2", color="blue")
    # ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target2, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1])
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "cyan", alpha=0.1)

    line_target3 = sl.Line(start3, end3, 1000)
    Sonar.clear_result()
    Sonar.scaning_result_simple(line_target3)
    target1 = line_target3.get_point()
    result1 = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target1)
    x_r, y_r, z_r = sl.list_position_ndarray(result1)
    ax1.plot3D(x_t, y_t, z_t, label="3", color="lightgreen")
    #ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target3, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1])
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "lime", alpha=0.1)

    line_target4 = sl.Line(start4, end4, 1000)
    Sonar.clear_result()
    Sonar.scaning_result_simple(line_target4)
    target1 = line_target4.get_point()
    result1 = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target1)
    x_r, y_r, z_r = sl.list_position_ndarray(result1)
    ax1.plot3D(x_t, y_t, z_t, label="0", color="tan")
    #ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target4, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1])
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "grey", alpha=0.01)

    
    cir1 = sl.circle(dis + 1.0, 3600)
    cir_low_re = sl.circle(dis + 1.0, 12)
    x1, y1, z1 = cir1.get_circle_point()
    x2, y2, z2 = cir_low_re.get_circle_point()
    ax1.plot3D(x1, y1, z1, "grey")
    # ax1.plot3D(x1, z1, y1, "grey")
    ax1.plot3D(z1, x1, y1, "grey")
    count = 0
    while count < 2:
        for i in range(12):
            x_re = [Sonar.x, x2[i]]
            y_re = [Sonar.y, y2[i]]
            z_re = [Sonar.z, z2[i]]
            if count == 0:
                ax1.plot3D(x_re, y_re, z_re, "grey", alpha=0.3)
            elif count == 1:
                ax1.plot3D(z_re, x_re, y_re, "grey", alpha=0.3)
            else:
                ax1.plot3D(x_re, z_re, y_re, "grey", alpha=0.3)
        count += 1
    '''

    # ax1.set_xlabel("x")
    # ax1.set_ylabel("y")
    # ax1.set_zlabel("z")
    # Axes3D.set_xlim(ax1, -20, 20)
    # Axes3D.set_ylim(ax1, 2, 25)
    # Axes3D.set_zlim(ax1, -17, 17)
    # plt.legend()
    # plt.show()
