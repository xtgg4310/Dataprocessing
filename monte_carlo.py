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
    fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    test_count = 0
    rate = np.zeros(3)
    rate_acc = np.zeros(3)
    correct = 0
    correct_acc = 0
    test_total_count = 0
    start_time = time.time()
    while test_total_count < 3:
        test_count = 0
        correct = 0
        while test_count < 3000:
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

            # print(x[50], y[50])
            value_groundtruth = op.function_to_target(z1, z2)
            x = np.linspace(h_min_1, h_max_1, 301)
            y = np.linspace(h_min_2, h_max_2, 301)
            x.sort()
            y.sort()
            x1, y1 = np.meshgrid(x, y)
            value = np.zeros((len(x), len(y)))
            flag = False
            flag_accuracy = False
            for i in range(len(x)):
                for j in range(len(y)):
                    value[i][j] = op.function_to_target(x1[i][j], y1[i][j])
            index_max = np.argmax(value)
            x_max = int(np.floor(index_max / 301))
            y_max = index_max % 301
            if z1 - 0.1 < x1[x_max][y_max] < z1 + 0.1 and z2 - 0.1 < y1[x_max][y_max] < z2 + 0.1:
                flag = True
            if z1 - 0.01 < x1[x_max][y_max] < z1 + 0.01 and z2 - 0.01 < y1[x_max][y_max] < z2 + 0.01:
                flag_accuracy = True
            if flag:
                correct += 1
            if flag_accuracy:
                correct_acc += 1
            test_count += 1
            print(test_count, correct, correct_acc, x1[x_max][y_max], y1[x_max][y_max], value[x_max][y_max], z1, z2,
                  value_groundtruth)
        rate[test_total_count] = correct / 3000.0
        rate_acc[test_total_count] = correct_acc / 3000.0
        print(time.time() - start_time, test_total_count, rate[test_total_count], rate_acc[test_total_count])
        test_total_count += 1

    print(rate)
