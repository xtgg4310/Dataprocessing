import random

import simulation as sl
import point
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import signal_process as sg
import optimizer as op
import txt2matrix

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
    x = random.sample(range(0, 1000), 100)
    x.sort()
    print(x)
    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    Sonar1 = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-3, 8, 1.7)
    end = point.point(-4, 12, 2)

    theta_target = (end.theta - start.theta) / 3 + start.theta
    theta_target1 = (end.theta - start.theta) * 2 / 3 + start.theta
    height = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    height_end = [-1.2, -0.8, -0.3, 0, 0.2, 0.4, 0.6, 0.8, 1]
    dis1 = [0] * len(height)

    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    scan_result = Sonar.get_result()
    result = {}
    index = 0
    while index < 10:
        optimizer_re, optimizer_record = op.optimizer_line_pose(scan_result[0].theta,
                                                                scan_result[len(scan_result) - 1].theta,
                                                                scan_result[0].r,
                                                                scan_result[len(scan_result) - 1].r, scan_result,
                                                                np.pi / 2)
        result.update({index: optimizer_re})
        print(index)
        index += 1

    # print(str(result[0]['params']['h_1']) + ',' + str(result[0]['params']['h_2']) + '\n')
    # fd = open(r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_3_2_5_180_2.txt", "w")
    # for i in range(len(result)):
    #    print(i)
    #    fd.writelines(str(result[i]['params']['h_1']) + ',' + str(result[i]['params']['h_2']) + '\n')
    # fd.close()
