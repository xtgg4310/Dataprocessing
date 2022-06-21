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
    fig = plt.figure()
    ax1 = Axes3D(fig)
    # fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Sonar = sl.sonar(4, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    Sonar1 = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    Sonar2 = sl.sonar(8, 5, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=np.pi / 2, angle_scan=np.pi / 180,
                      end_angle=np.pi * 3 / 4)
    test_count = 0
    rate = np.zeros(3)
    rate_acc = np.zeros(3)
    test_total_count = 0

    while test_total_count < 100:
        fp = open(
            r"C:\Users\Enigma_2020\Desktop\simulation_result\ground_truth_401_" + str(test_total_count) + ".txt",
            "w")
        test_count = 0
        Sonar.clear_result()
        Sonar1.clear_result()
        x1 = random.uniform(-5, 5)
        y1 = random.uniform(2, 8)
        z1 = random.uniform(0,
                            min(np.sqrt(x1 ** 2 + y1 ** 2) * np.tan(np.pi / 8),
                                np.sqrt((x1 - Sonar1.x) ** 2 + (y1 - Sonar1.y) ** 2) * np.tan(np.pi / 8) + Sonar1.z))
        x2 = random.uniform(x1 - 2, x1 + 2)
        while x1 == x2:
            x2 = random.uniform(x1 - 2, x1 + 2)
        y2 = random.uniform(max(2.0, y1 - (2 - abs(x1 - x2))), min(y1 + (2 - abs(x1 - x2)), 8.0))
        while y1 == y2:
            y2 = random.uniform(max(2.0, y1 - (2 - abs(x1 - x2))), min(y1 + (2 - abs(x1 - x2)), 8.0))
        z2 = random.uniform(max(max(-np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                                    -1 * np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                                        np.pi / 8) + Sonar1.z), z1 - (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))),
                            min(min(np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                                    np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                                        np.pi / 8) + Sonar1.z), z1 + (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))))
        while z1 == z2:
            z2 = random.uniform(max(max(-np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                                        -1 * np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                                            np.pi / 8) + Sonar1.z),
                                    z1 - (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))),
                                min(min(np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                                        np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                                            np.pi / 8) + Sonar1.z),
                                    z1 + (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))))
        height = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        # print(height)
        if height < 1.4 or height > 2:
            continue
        start = point.point(7, 10, 1)
        end = point.point(2, 5, 1)
        height = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        # print(x1, y1, z1)
        # print(x2, y2, z2)
        line_target = sl.Line(start, end, 1000)

        Sonar.scaning_result_simple(line_target)
        Sonar1.scaning_result_simple(line_target)
        Sonar2.scaning_result_simple(line_target)
        scan_result = Sonar.get_result()
        scan_result2 = Sonar1.get_result()
        scan_result3 = Sonar2.get_result()
        #result_line = line_target.get_point()
        #x_t, y_t, z_t = sl.list_position_ndarray(result_line)
        #x_r, y_r, z_r = sl.list_position_ndarray(scan_result3)
        #x_r1, y_r1, z_r1 = sl.list_position_ndarray(scan_result)
        #ax1.plot3D(x_t, y_t, z_t, color="purple")
        #ax1.plot3D(x_r1, y_r1, z_r1, "green")
        #ax1.plot3D(x_r, y_r, z_r, "blue")
        #ax1.scatter3D(8, 5, 1, color="red")
        #ax1.scatter3D(4, 0, 0, color="yellow")
        plt.show()
        op.set_global_parameter(scan_result[0].theta,
                                scan_result[len(scan_result) - 1].theta,
                                scan_result[0].r,
                                scan_result[len(scan_result) - 1].r, scan_result,
                                np.pi * 2 / 3)
        scan_result2[0].sonar_axis_convert(Sonar1)
        scan_result2[len(scan_result2) - 1].sonar_axis_convert(Sonar1)
        # print(scan_result2[0].x, scan_result2[0].y, scan_result2[0].z, scan_result2[0].theta) print(
        # scan_result2[0].sonar_x, scan_result2[0].sonar_y, scan_result2[0].sonar_z, scan_result2[0].sonar_theta)
        op.set_global_parameter2(scan_result2[0].sonar_theta, scan_result2[len(scan_result2) - 1].sonar_theta,
                                 scan_result2[0].sonar_r, scan_result2[len(scan_result2) - 1].sonar_r, scan_result2,
                                 np.pi * 2 / 3, Sonar1, True)
        h_min_1 = max(op.h_min_2[0], op.h_start_min)
        h_max_1 = min(op.h_start_max, op.h_max_2[0])
        h_min_2 = max(op.h_end_min, op.h_min_2[1])
        h_max_2 = min(op.h_end_max, op.h_max_2[1])
        x = np.linspace(h_min_1, h_max_1, 401)
        y = np.linspace(h_min_2, h_max_2, 401)
        x1, y1 = np.meshgrid(x, y)
        value = np.zeros((len(x), len(y)))
        dic_result = {}
        index = 0
        for i in range(len(x)):
            for j in range(len(y)):
                value[i][j] = op.function_to_target_both(x1[i][j], y1[i][j])
                if value[i][j] > -20:
                    dic_result.update({index: [x1[i][j], y1[i][j], value[i][j]]})
                    index += 1
                else:
                    value[i][j] = -21
        # ax1.plot_surface(x1, y1, value)
        # ax1.scatter3D(z1, z2, op.function_to_target_both(z1, z2))
        # for i in range(len(dic_result)):
        #    ax1.scatter3D(dic_result[i][0], dic_result[i][1], dic_result[i][2])
        # plt.show()
        '''
        for i in range(len(dic_result)):
            plt.scatter(dic_result[i][0], dic_result[i][1], color=color[test_count], alpha=0.2)
        for i in range(len(dic_result2)):
            plt.scatter(dic_result2[i][0], dic_result2[i][1], color=color_2[test_count], alpha=0.1)
        if len(cross_set) != 0:
            for i in range(len(cross_set)):
                plt.scatter(cross_set[i][0], cross_set[i][1], color="black", alpha=0.3)
        # if line_target.start.z >= 0:
        plt.scatter(line_target.start.z, line_target.end.z, color="red")

        # else:
        plt.scatter(-1 * line_target.start.z, -1 * line_target.end.z, color=color1[test_count])
        # if line_target.start.z >= 1:
        # plt.scatter(line_target.start.z, line_target.end.z, color=color1[test_count])
        # else:
        plt.scatter(2 - line_target.start.z, 2 - line_target.end.z, color=color1[test_count])
        if len(cross_set) != 0:
            plt.scatter(x_weight, y_weight, color="blue")
            print(test_total_count, x_weight - line_target.start.z, y_weight - line_target.end.z)
        plt.xlabel('h_1')
        plt.ylabel('h_2')
        plt.xlim(min(h_min_1, op.h_min_2[0]), max(h_max_1, op.h_max_2[0]))
        plt.ylim(min(h_min_2, op.h_min_2[1]), max(h_max_2, op.h_max_2[1]))
        plt.show()
        '''
