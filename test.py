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
    start = point.point(-3.5, 18.5, 5)
    end = point.point(-4, 18, 4.3)

    line_target = sl.Line(start, end, 1000)
    Sonar.scaning_result_simple(line_target)
    scan_result = Sonar.get_result()

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
    x = np.linspace(h_min_1, h_max_1, 101)
    y = np.linspace(h_min_2, h_max_2, 101)
    x1, y1 = np.meshgrid(x, y)
    value = np.zeros((len(x), len(y)))
    
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
                if filer_count == 1:
                    plt.scatter(x_temp[i], y_temp[i], color='blue')
                index += 1
        print(len(dic_result))
        if filer_count == 1:
            plt.scatter(5, 4.6, color="green")
            plt.xlabel("h_start")
            plt.ylabel("h_end")
            plt.show()
        if 30 <= len(dic_result) < 60:
            break
        filer_count += 1


    '''
    result = {}
    index = 0
    while index < 30:
        best_x, best_y = op.optimizer_genetic_line_pose(scan_result[0].theta,
                                                        scan_result[len(scan_result) - 1].theta,
                                                        scan_result[0].r,
                                                        scan_result[len(scan_result) - 1].r, scan_result,
                                                        np.pi / 2)
        print(best_x, best_y)
        index += 1

    # print(str(result[0]['params']['h_1']) + ',' + str(result[0]['params']['h_2']) + '\n')
    # fd = open(r"C:\Users\Enigma_2020\Desktop\sonar_simulation\record_test_-2,8,2_-3,12,1_2_180_3.txt", "w")
    # for i in range(len(result)):
    #   print(i)
    #   fd.writelines(str(result[i]['params']['h_1']) + ',' + str(result[i]['params']['h_2']) + '\n')
    # fd.close()
   
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
    '''
