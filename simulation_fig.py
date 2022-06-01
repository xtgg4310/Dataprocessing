import random

import simulation as sl
import point
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import signal_process as sg

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = Axes3D(fig)

    Sonar = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-4, 2, 3)
    end = point.point(-2, 5, -2)

    start1 = point.point(-4, 2, 3.808)
    end1 = point.point(-2, -5, 2.472)

    start2 = point.point(-13, 13, np.sqrt(50))
    end2 = point.point(-2, 6, 12)

    start3 = point.point(-8, 8, np.sqrt(260))
    end3 = point.point(-2, 6, 12)

    start4 = point.point(-11, 11, np.sqrt(146))
    end4 = point.point(-2, 6, 12)
    dis_array = [start.r, end.r, start1.r, end1.r, start2.r, end2.r, start3.r, end3.r, start4.r, end4.r]
    dis = np.max(dis_array)
    theta_target = (end.theta - start.theta) / 3 + start.theta
    theta_target1 = (end.theta - start.theta) * 2 / 3 + start.theta
    height = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    height_end = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    dis1 = [0] * len(height)
    '''
    k = 3
    for i in range(len(height)):
        k = int(random.uniform(0, len(height_end) - 1))
        x, y = sg.x_y_cal(height[i], start.theta, start.r)
        x_1, y_1 = sg.x_y_cal(height_end[k], end.theta, end.r)
        # print(x_1, y_1, x_1 ** 2 + y_1 ** 2 + height[k] ** 2)
        dis1[i] = sg.search_sonar_line_min_dis_theta(x, y, x_1, y_1, height[i], height_end[k], theta_target)
    plt.plot(height, dis1, label=str(height_end[k]))
    plt.show()
    plt.close()
    count = 0
    while count < len(height_end):
        k = len(height_end) - count - 1
        for i in range(len(height)):
            x, y = sg.x_y_cal(height[i], start.theta, start.r)
            x_1, y_1 = sg.x_y_cal(height_end[k], end.theta, end.r)
            # print(x_1, y_1, x_1 ** 2 + y_1 ** 2 + height[k] ** 2)
            dis1[i] = sg.search_sonar_line_min_dis_theta(x, y, x_1, y_1, height[i], height_end[k], theta_target)
        plt.plot(height, dis1, label=str(height_end[count]))
        count += 1
    plt.legend()
    plt.show()
    '''
    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    target = line_target.get_point()

    re = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target)
    x_r, y_r, z_r = sl.list_position_ndarray(re)

    ax1.scatter3D(Sonar.x, Sonar.y, Sonar.z, color="blue")
    ax1.plot3D(x_t, y_t, z_t, label="0_yellow_orange", color="yellow")
    ax1.plot3D(x_r, y_r, z_r, "green")
    result = Sonar.scan_line(line_target, np.pi)
    # for i in range(len(result)):
    #    if len(result[i][0]) != 0:
    # print(result[i][0][0])

    if len(result) != 0:
        for i in range(len(result)):
            dic = sl.surface_point(result[i][0], result[i][1], Sonar)
            for j in range(len(dic)):
                print(result[i][1], len(dic))
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "orange", alpha=0.05)


    '''
    line_target1 = sl.Line(start1, end1, 1000)
    Sonar.clear_result()
    Sonar.scaning_result_simple(line_target1)
    target1 = line_target1.get_point()
    result1 = Sonar.get_result()
    x_t, y_t, z_t = sl.list_position_ndarray(target1)
    x_r, y_r, z_r = sl.list_position_ndarray(result1)
    ax1.plot3D(x_t, y_t, z_t, label="1_purple_red", color="purple")
    ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target1, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1], Sonar)
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
    ax1.plot3D(x_t, y_t, z_t, label="2_blue_cyan", color="blue")
    # ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target2, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1], Sonar)
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
    ax1.plot3D(x_t, y_t, z_t, label="3_lightgreen_lime", color="lightgreen")
    # ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target3, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1], Sonar)
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
    ax1.plot3D(x_t, y_t, z_t, label="4_tan_grey", color="tan")
    # ax1.plot3D(x_r, y_r, z_r, "blue")
    result1 = Sonar.scan_line(line_target4, np.pi)

    if len(result1) != 0:
        for i in range(len(result1)):
            dic = sl.surface_point(result1[i][0], result1[i][1], Sonar)
            for j in range(len(dic)):
                x, y, z = sl.list_position_ndarray(dic[j])
                ax1.plot3D(x, y, z, "grey", alpha=0.2)
    '''
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    # Axes3D.set_xlim(ax1, -20, 20)
    # Axes3D.set_ylim(ax1, 7.0, 13)
    # Axes3D.set_zlim(ax1, -17, 17)
    plt.legend()
    plt.show()

