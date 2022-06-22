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
    Sonar1 = sl.sonar(10, 5, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=np.pi / 2, angle_scan=np.pi / 180,
                      end_angle=np.pi * 3 / 4)
    Sonar2 = sl.sonar(-4, 0, 1.5, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
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

    while test_total_count < 100:
        fp = open(
            r"C:\Users\Enigma_2020\Desktop\simulation_fig_vertical_sonar\ground_truth_10_5_1_4_0_1.5_201_dis_top200_" + str(
                test_total_count) + ".txt",
            "w")
        test_count = 0
        Sonar.clear_result()
        Sonar1.clear_result()
        Sonar2.clear_result()

        x1 = random.uniform(-8, 8)
        y1 = random.uniform(2, 8)
        z1 = random.uniform(0,
                            min(np.sqrt((x1 - Sonar2.x) ** 2 + (y1 - Sonar2.y) ** 2) * np.tan(np.pi / 8) + Sonar2.z,
                                np.sqrt(x1 ** 2 + y1 ** 2) * np.tan(np.pi / 8),
                                np.sqrt((x1 - Sonar1.x) ** 2 + (y1 - Sonar1.y) ** 2) * np.tan(np.pi / 8) + Sonar1.z))
        x2 = random.uniform(x1 - 2, x1 + 2)
        while x1 == x2:
            x2 = random.uniform(x1 - 2, x1 + 2)
        y2 = random.uniform(max(2.0, y1 - (2 - abs(x1 - x2))), min(y1 + (2 - abs(x1 - x2)), 8.0))
        while y1 == y2:
            y2 = random.uniform(max(2.0, y1 - (2 - abs(x1 - x2))), min(y1 + (2 - abs(x1 - x2)), 8.0))
        z2 = random.uniform(
            max(max(-1 * np.sqrt((x2 - Sonar2.x) ** 2 + (y2 - Sonar2.y) ** 2) * np.tan(np.pi / 8) + Sonar2.z,
                    -1 * np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                    -1 * np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                        np.pi / 8) + Sonar1.z), z1 - (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))),
            min(min(np.sqrt((x1 - Sonar2.x) ** 2 + (y1 - Sonar2.y) ** 2) * np.tan(np.pi / 8) + Sonar2.z,
                    np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                    np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                        np.pi / 8) + Sonar1.z), z1 + (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))))
        while z1 == z2:
            z2 = random.uniform(
                max(max(-1 * np.sqrt((x2 - Sonar2.x) ** 2 + (y2 - Sonar2.y) ** 2) * np.tan(np.pi / 8) + Sonar2.z,
                        -1 * np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                        -1 * np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                            np.pi / 8) + Sonar1.z),
                    z1 - (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))),
                min(min(np.sqrt((x1 - Sonar2.x) ** 2 + (y1 - Sonar2.y) ** 2) * np.tan(np.pi / 8) + Sonar2.z,
                        np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                        np.sqrt((x2 - Sonar1.x) ** 2 + (y2 - Sonar1.y) ** 2) * np.tan(
                            np.pi / 8) + Sonar1.z),
                    z1 + (2 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))))
        height = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        # print(height)
        if height < 1.4 or height > 2:
            continue
        start = point.point(x1, y1, z1)
        end = point.point(x2, y2, z2)
        print(x1, y1, z1)
        print(x2, y2, z2)
        line_target = sl.Line(start, end, 1000)
        index_direction = random.randint(0, len(direction) - 1)
        while test_count < 1:
            print(test_total_count)
            Sonar.scaning_result_simple(line_target)
            Sonar1.scaning_result_simple(line_target)
            Sonar2.scaning_result_simple(line_target)
            scan_result = Sonar.get_result()
            scan_result2 = Sonar1.get_result()
            scan_result3 = Sonar2.get_result()
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
            scan_result3[0].sonar_axis_convert(Sonar2)
            scan_result3[len(scan_result3) - 1].sonar_axis_convert(Sonar2)
            op.set_global_parameter3(scan_result3[0].sonar_theta, scan_result3[len(scan_result3) - 1].sonar_theta,
                                     scan_result3[0].sonar_r, scan_result3[len(scan_result3) - 1].sonar_r, scan_result3,
                                     np.pi * 2 / 3, Sonar2, True)
            h_min_1 = max(op.h_start_min, op.h_min_2[0], op.h_min_3[0])
            h_max_1 = min(op.h_start_max, op.h_max_2[0], op.h_max_3[0])
            h_min_2 = max(op.h_end_min, op.h_min_2[1], op.h_min_3[1])
            h_max_2 = min(op.h_end_max, op.h_max_2[1], op.h_max_3[1])
            x = np.linspace(h_min_1, h_max_1, 501)
            y = np.linspace(h_min_2, h_max_2, 501)
            x1, y1 = np.meshgrid(x, y)
            value = np.zeros((len(x), len(y)))
            dic_result = []
            flag = False
            for i in range(len(x)):
                for j in range(len(y)):
                    value[i][j] = op.function_to_target_3_sonar_dis(x1[i][j], y1[i][j])
                    if flag:
                        dic_result = np.append(dic_result, np.array([[x1[i][j], y1[i][j], value[i][j]]]), axis=0)
                    else:
                        flag = True
                        dic_result = np.array([[x1[i][j], y1[i][j], value[i][j]]])
            top_200_result = sorted(dic_result, key=lambda x: x[2], reverse=True)[0:200]
            print(top_200_result)

            for i in range(200):
                plt.scatter(top_200_result[i][0], top_200_result[i][1], color=color[test_count], alpha=0.5)

            plt.scatter(line_target.start.z, line_target.end.z, color="red")

            plt.scatter(-1 * line_target.start.z, -1 * line_target.end.z, color=color1[test_count])

            plt.scatter(2 - line_target.start.z, 2 - line_target.end.z, color=color1[test_count])

            plt.xlabel('h_1')
            plt.ylabel('h_2')
            plt.xlim(min(h_min_1, op.h_min_2[0],op.h_min_3[0]), max(h_max_1, op.h_max_2[0],op.h_max_3[0]))
            plt.ylim(min(h_min_2, op.h_min_2[1],op.h_min_3[1]), max(h_max_2, op.h_max_2[1],op.h_max_3[1]))
            # plt.legend()
            # plt.show()
            plt.savefig(
                r"C:\Users\Enigma_2020\Desktop\simulation_fig_vertical_sonar\fig_area_10_5_1_4_0_1.5_201_dis_top200_" + str(
                    test_total_count) + ".jpg")
            plt.close()
            fp.writelines(
                str(line_target.start.x) + "," + str(line_target.start.y) + "," + str(
                    line_target.start.z) + "," + str(
                    line_target.end.x) + "," + str(line_target.end.y) + "," +
                str(line_target.end.z) + "|||||"  # + str(x_weight - line_target.start.z) + "," + str(
                # y_weight - line_target.end.z) + "|||||" + str(x_weight - (-1 * line_target.start.z)) + "," + str(
                # y_weight - (-1 * line_target.start.z)) + "|||||" + str(
                # x_weight - (2 - line_target.start.z)) + "," + str(
                # y_weight - (2 - line_target.start.z))
                + "\n")
            fp.close()
            # line_target.move_line(direction[index_direction], distance)
            test_count += 1
        test_total_count += 1
