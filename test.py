import time
import numpy as np
from matplotlib import pyplot as plt
import optimizer as op
import point
import simulation as sl
import signal_process as sg

if __name__ == '__main__':
    # fig = plt.figure()
    # ax1 = Axes3D(fig)
    # fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-3, 8, 4)
    end = point.point(-4, 12, 3)

    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    scan_result = Sonar.get_result()
    print(scan_result[0].r, scan_result[1001].r)
    # print()
    '''
    op.set_global_parameter(scan_result[0].theta,
                            scan_result[len(scan_result) - 1].theta,
                            scan_result[0].r,
                            scan_result[len(scan_result) - 1].r, scan_result,
                            np.pi / 2)
    
    direction = ["up", "down", "left", "right", "near", "away", "forward", "backward"]
    h_min_1 = op.h_start_min
    h_max_1 = op.h_start_max
    h_min_2 = op.h_end_min
    h_max_2 = op.h_end_max

    # print(x[50], y[50])
    time1 = time.time()
    count = 0
    x = np.linspace(h_min_1, h_max_1, 301)
    y = np.linspace(h_min_2, h_max_2, 301)
    x1, y1 = np.meshgrid(x, y)
    value = np.zeros((len(x), len(y)))
    
    for i in range(len(x)):
        for j in range(len(y)):
            value[i][j] = op.function_to_target(x1[i][j], y1[i][j])

    value = np.array(value)
    value = (value - np.min(value)) / (np.max(value) - np.min(value))
    index_max = np.argmax(value)
    x_max = int(np.floor(index_max / 301))
    y_max = index_max % 301
    print(np.max(value), value[x_max][y_max])
    dic_result = {}
    result_index = 0

    for i in range(len(x)):
        for j in range(len(y)):
            if value[i][j] > 0.99:
                dic_result.update({result_index: [x1[i][j], y1[i][j], value[i][j]]})
                result_index += 1

    time2 = time.time()
    '''
    result = {}
    index = 0
    while index < 10:
        optimizer_re, optimizer_record = op.optimizer_line_pose(scan_result[0].theta,
                                                                scan_result[len(scan_result) - 1].theta,
                                                                scan_result[0].r,
                                                                scan_result[len(scan_result) - 1].r, scan_result,
                                                                np.pi / 2)
        result.update({index: optimizer_re})
        # print(type(optimizer_re['params']['h_1']))
        index += 1

    # print(str(result[0]['params']['h_1']) + ',' + str(result[0]['params']['h_2']) + '\n')
    # fd = open(r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_-2,8,2_-3,12,1_2_180_3.txt", "w")
    # for i in range(len(result)):
    #   print(i)
    #   fd.writelines(str(result[i]['params']['h_1']) + ',' + str(result[i]['params']['h_2']) + '\n')
    # fd.close()
