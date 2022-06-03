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
    test_count = 0
    rate = np.zeros(3)
    rate_acc = np.zeros(3)
    test_total_count = 0
    start_time = time.time()
    direction = ["up", "down", "left", "right", "near", "away", "forward", "backward"]
    color = ["yellowgreen", "plum", "lightgreen", "lightgrey", "wheat", "cyan", "salmon", "peachpuff",
             "lightpink", "lightblue"]
    color1 = ["yellow", "purple", "green", "black", "tan", "darkcyan", "red", "peru", "deeppink", "blue"]
    distance = 0.3

    while test_total_count < 20:
        fp = open(r"C:\Users\Enigma_2020\Desktop\simulation_fig\ground_truth_" + str(test_total_count) + ".txt", "w")
        test_count = 0
        Sonar.clear_result()
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)
        while x1 == x2:
            x2 = random.uniform(-5, 5)
        y1 = random.uniform(2, 8)
        y2 = random.uniform(2, 8)
        while y1 == y2:
            y2 = random.uniform(2, 8)
        z1 = random.uniform(0,
                            np.sqrt(x1 ** 2 + y1 ** 2) * np.tan(np.pi / 8))
        z2 = random.uniform(-np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                            np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8))
        while z1 == z2:
            z2 = random.uniform(-np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8),
                                np.sqrt(x2 ** 2 + y2 ** 2) * np.tan(np.pi / 8))
        start = point.point(x1, y1, z1)
        end = point.point(x2, y2, z2)
        line_target = sl.Line(start, end, 1000)
        index_direction = random.randint(0, len(direction) - 1)
        while test_count < 10:
            print(test_total_count, test_count)
            Sonar.scaning_result_simple(line_target)
            scan_result = Sonar.get_result()
            op.set_global_parameter(scan_result[0].theta,
                                    scan_result[len(scan_result) - 1].theta,
                                    scan_result[0].r,
                                    scan_result[len(scan_result) - 1].r, scan_result,
                                    np.pi * 2 / 3)

            h_min_1 = op.h_start_min
            h_max_1 = op.h_start_max
            h_min_2 = op.h_end_min
            h_max_2 = op.h_end_max
            x = np.linspace(h_min_1, h_max_1, 401)
            y = np.linspace(h_min_2, h_max_2, 401)
            x1, y1 = np.meshgrid(x, y)
            value = np.zeros((len(x), len(y)))
            dic_result = {}
            index = 0
            for i in range(len(x)):
                for j in range(len(y)):
                    value[i][j] = op.function_to_target(x1[i][j], y1[i][j])
                    print(value[i][j])
                    if value[i][j] > -10:
                        dic_result.update({index: [x1[i][j], y1[i][j]]})

            for i in range(len(dic_result)):
                plt.scatter(dic_result[i][0], dic_result[i][1], color=color[test_count])

            if line_target.start.z >= 0:
                plt.scatter(line_target.start.z, line_target.end.z, color=color1[test_count],
                            label=str(test_count) + "_truth")
            else:
                plt.scatter(-1 * line_target.start.z, -1 * line_target.end.z, color=color1[test_count],
                            label=str(test_count) + "_truth")
            plt.xlabel('h_1')
            plt.ylabel('h_2')
            plt.xlim(h_min_1, h_max_1)
            plt.ylim(h_min_2, h_max_2)
            plt.legend()
            plt.savefig(
                r"C:\Users\Enigma_2020\Desktop\simulation_fig\fig_area" + direction[index_direction] + "_" + str(
                    test_total_count) + "_" + str(test_count) + ".jpg")
            fp.writelines(
                str(line_target.start.x) + "," + str(line_target.start.y) + "," + str(
                    line_target.start.z) + "," + str(
                    line_target.end.x) + "," + str(line_target.end.y) + "," +
                str(line_target.end.z) + "\n")

            line_target.move_line(direction[index_direction], distance)
            test_count += 1
        plt.xlabel('h_1')
        plt.ylabel('h_2')
        plt.legend()
        plt.savefig(
            r"C:\Users\Enigma_2020\Desktop\simulation_fig\fig_" + direction[index_direction] + "_" + str(
                test_total_count) + ".jpg")
        plt.close()
        test_total_count += 1
        fp.close()
