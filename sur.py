import numpy as np
from scipy.interpolate import pchip_interpolate as pch


def search_mindis_section(index_min, index_max):
    i_min = 0
    i_max = 0
    result = {}
    index = 0
    while i_min < len(index_min) and i_max < len(index_max):
        if index_min[i_min] < index_max[i_max]:
            i_min += 1
        elif index_min[i_min] > index_max[i_max]:
            i_max += 1
        else:
            result.update({index: i_min})
    return result


def get_pointfromsurface(scan_rotate_result):
    index_min = np.ndarray([])
    index_max = np.ndarray([])
    temp_min = 0
    temp_max = 0
    relate_x_y = np.ndarray([])
    for i in range(1, len(scan_rotate_result)):
        if scan_rotate_result[i][0][0] < scan_rotate_result[temp_min][0][0]:
            temp_min = i
        if scan_rotate_result[i][0][len(scan_rotate_result[i][0]) - 1] < scan_rotate_result[temp_max][0][
            len(scan_rotate_result[temp_max][0]) - 1]:
            temp_max = i
        if scan_rotate_result[i][0][0] == scan_rotate_result[temp_min][0][0]:
            index_min = np.append(index_min, i)
        if scan_rotate_result[i][0][len(scan_rotate_result[i][0]) - 1] == scan_rotate_result[temp_max][0][
            len(scan_rotate_result[temp_max][0]) - 1]:
            index_max = np.append(index_max, i)
    result = search_mindis_section(index_min, index_max)
    point_set_temp = {}
    point_index = 0
    if result[0] == 0 or result[0] == len(scan_rotate_result):
        aaa = 0
    else:
        for i in range(result[0]):
            point_set_temp.update({point_index: [scan_rotate_result[i][0][0],
                                                 scan_rotate_result[i][1][len(scan_rotate_result[i][1]) - 1]]})
            point_index += 1

        for i in range(result[len(result) - 1], len(scan_rotate_result)):
            point_set_temp.update({point_index: [scan_rotate_result[i][0][0],
                                                 scan_rotate_result[i][1][0]]})
            point_index += 1
