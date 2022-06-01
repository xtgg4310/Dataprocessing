import time
import numpy as np
from matplotlib import pyplot as plt
import optimizer as op
import point
import simulation as sl

if __name__ == '__main__':
    Sonar = sl.sonar(0, 0, 0, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
    start = point.point(-3.5, 18.5, 5)
    end = point.point(-4, 18, 4.6)
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

    time1 = time.time()
    count = 0
    x = np.linspace(h_min_1, h_max_1, 101)
    y = np.linspace(h_min_2, h_max_2, 101)
    x1, y1 = np.meshgrid(x, y)
    value_dic = {}
    for i in range(len(x)):
        for j in range(len(y)):
            value_dic.update({i * 101 + j: op.function_to_target(x1[i][j], y1[i][j])})
    sort_value = sorted(value_dic.items(), key=lambda i: i[1], reverse=True)
    count = 0
    for key in sort_value:
        if count == 20:
            break
        else:
            print(x1[int(np.floor(key[0] / 101))][key[0] % 101], y1[int(np.floor(key[0] / 101))][key[0] % 101], key[1])
            count += 1