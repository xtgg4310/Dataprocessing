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
    Sonar1 = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    test_count = 0
    rate = np.zeros(3)
    rate_acc = np.zeros(3)
    test_total_count = 0
    start_time = time.time()
    direction = ["up", "down", "left", "right", "near", "away", "forward", "backward"]
    color = ["yellowgreen", "plum", "lightgreen", "lightgrey", "wheat", "cyan", "salmon", "peachpuff",
             "lightpink", "lightblue"]
    color_2 = ["gold", "darkviolet", "limegreen", "dimgrey", "burlywood", "teal", "tomato", "linen", "deeppink",
               "dodgerblue"]
    color1 = ["yellow", "purple", "green", "black", "tan", "darkcyan", "red", "peru", "deeppink", "blue"]
    distance = 0.3
    error = [0] * 2000
    error_record = np.array([])
    while test_total_count < 2000:
        fp = open(
            r"C:\Users\Enigma_2020\Desktop\simulation_result\ground_truth_top100_401_" + str(test_total_count) + ".txt",
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
        start = point.point(x1, y1, z1)
        end = point.point(x2, y2, z2)
        height = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        # print(x1, y1, z1)
        # print(x2, y2, z2)
        line_target = sl.Line(start, end, 1000)
        index_direction = random.randint(0, len(direction) - 1)

        Sonar.scaning_result_simple(line_target)
        Sonar1.scaning_result_simple(line_target)
        scan_result = Sonar.get_result()
        scan_result2 = Sonar1.get_result()
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
        h_min_1 = op.h_start_min
        h_max_1 = op.h_start_max
        h_min_2 = op.h_end_min
        h_max_2 = op.h_end_max
        x = np.linspace(h_min_1, h_max_1, 401)
        y = np.linspace(h_min_2, h_max_2, 401)
        x_2 = np.linspace(op.h_min_2[0], op.h_max_2[0], 401)
        y_2 = np.linspace(op.h_min_2[1], op.h_max_2[1], 401)
        x1, y1 = np.meshgrid(x, y)
        x2, y2 = np.meshgrid(x_2, y_2)
        value = np.zeros((len(x), len(y)))
        value2 = np.zeros((len(x_2), len(y_2)))
        dic_result = np.array([[]])
        dic_result2 = np.array([[]])
        flag = False

        resolve_1 = 2 * h_max_2 / 401
        resolve_2 = (op.h_max_2[1] - op.h_min_2[1]) / 401
        for i in range(len(x)):
            for j in range(len(y)):
                value[i][j] = op.function_to_target(x1[i][j], y1[i][j])
                if value[i][j] > -20:
                    if not flag:
                        dic_result = np.array([[x1[i][j], y1[i][j], value[i][j]]])
                        flag = True
                    else:
                        dic_result = np.append(dic_result, np.array([[x1[i][j], y1[i][j], value[i][j]]]), axis=0)
                    # print(value[i][j])

        flag = False
        for i in range(len(x_2)):
            for j in range(len(y_2)):
                value2[i][j] = op.function_to_target_sonar2(x2[i][j], y2[i][j])
                # print(value2[i][j])
                if value2[i][j] > -20:
                    if not flag:
                        dic_result2 = np.array([[x2[i][j], y2[i][j], value2[i][j]]])
                        flag = True
                    else:
                        dic_result2 = np.append(dic_result2, np.array([[x2[i][j], y2[i][j], value2[i][j]]]), axis=0)

        dic_result = sorted(dic_result, key=lambda x: x[2], reverse=True)
        dic_result2 = sorted(dic_result2, key=lambda x: x[2], reverse=True)
        set_1 = dic_result[0:199]
        set_2 = dic_result2[0:199]
        # print("groud_truth:", op.function_to_target(line_target.start.z, line_target.end.z))
        # print("groud_truth2:", op.function_to_target_sonar2(line_target.start.z, line_target.end.z))
        # print(max(resolve_1, resolve_2))
        cross_set = sg.point_set_cross(set_1, set_2, max(resolve_1, resolve_2))
        x_weight = -np.inf
        y_weight = -np.inf
        print(test_total_count, len(cross_set), len(set_1), len(set_2))

        if len(cross_set) != 0:
            x_weight, y_weight = sg.weight_recalculate(cross_set)
        error_1 = np.sqrt((x_weight - line_target.start.z) ** 2 + (y_weight - line_target.end.z) ** 2)
        error_2 = np.sqrt(
            (x_weight - (-1 * line_target.start.z)) ** 2 + (y_weight - (-1 * line_target.start.z)) ** 2)
        error_3 = np.sqrt((x_weight - (2 - line_target.start.z)) ** 2 + (y_weight - (2 - line_target.start.z)) ** 2)
        error_temp = min(error_1, error_2, error_3)
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
        fp.writelines(str(error_temp) + "\n")

        # line_target.move_line(direction[index_direction], distance)
        error[test_total_count] = error_temp
        test_total_count += 1
        fp.close()
        if test_total_count == 0:
            error_record = np.array([error_temp])
        else:
            error_record = np.append(error_record, error_temp)
        if test_total_count % 100 == 0:
            print(test_total_count, ":", np.mean(error), np.var(error), np.max(error), np.min(error))
    error = np.array(error)
    print(np.mean(error), np.var(error), np.max(error), np.min(error))
