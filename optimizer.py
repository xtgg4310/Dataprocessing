from bayes_opt import BayesianOptimization
import point
import signal_process as sg
import numpy as np
import simulation as sl
from sko.PSO import PSO
from sko.SA import SA
from sko.GA import GA
import torch
import copy

theta_sonar1 = []
r_1 = 0
r_2 = 0
target_dis_sonar1 = []
h_start_min = 0
h_start_max = 0
h_end_min = 0
h_end_max = 0
pbounds = {}
start_theta_re = 0
end_theta_re = 0
lb = [0, 0]
ls = [0, 0]
theta_sonar2 = []
r = []
target_dis = []
h_min_2 = []
h_max_2 = []
start_theta_re2 = 0
end_theta_re2 = 0
sonar2 = sl.sonar(1, 0, 1, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
k_sonar2 = []
k_sonar1 = []
diff_k = []
diff_k_2 = []
theta_sonar3 = []
r_3 = []
target_dis_3 = []
h_min_3 = []
h_max_3 = []
start_theta_re3 = 0
end_theta_re3 = 0
sonar3 = sl.sonar(4, 0, 1.5, np.pi * 180 / 180, np.pi / 180, 40, current_angle=0)
k_sonar3 = []
diff_k_3 = []
theta_sonar_ = []
r_ = []
target_dis_ = []
h_min_ = []
h_max_ = []
start_theta_re_ = []
end_theta_re_ = []
sonar_ = {}
k_sonar_ = []
diff_k_ = []


def set_global_parameter(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar=None, Flag=False):
    global theta_sonar1
    global target_dis_sonar1
    global r_1
    global r_2
    global h_start_min
    global h_start_max
    global h_end_min
    global h_end_max
    global pbounds
    global start_theta_re
    global end_theta_re
    global lb
    global ls
    global k_sonar1
    theta_sonar1 = np.zeros(9)
    target_dis_sonar1 = np.zeros(9)
    k_sonar1 = np.zeros(8)
    if Flag:
        for i in range(len(scan)):
            scan[i].sonar_axis_convert(sonar)
    start_theta_re = start_theta
    end_theta_re = end_theta
    theta_sonar1[0] = (end_theta - start_theta) * 20 / 180 + start_theta
    theta_sonar1[1] = (end_theta - start_theta) * 40 / 180 + start_theta
    theta_sonar1[2] = (end_theta - start_theta) * 50 / 180 + start_theta
    theta_sonar1[3] = (end_theta - start_theta) * 70 / 180 + start_theta
    theta_sonar1[4] = (end_theta - start_theta) * 90 / 180 + start_theta
    theta_sonar1[5] = (end_theta - start_theta) * 100 / 180 + start_theta
    theta_sonar1[6] = (end_theta - start_theta) * 120 / 180 + start_theta
    theta_sonar1[7] = (end_theta - start_theta) * 140 / 180 + start_theta
    theta_sonar1[8] = (end_theta - start_theta) * 160 / 180 + start_theta
    r_1 = dis1
    r_2 = dis2
    h_start_min = -r_1 * np.sin(vertical_angle / 2)
    h_end_min = -r_2 * np.sin(vertical_angle / 2)
    h_start_max = r_1 * np.sin(vertical_angle / 2)
    h_end_max = r_2 * np.sin(vertical_angle / 2)
    lb = [h_start_min, h_end_min]
    ls = [h_start_max, h_end_max]
    pbounds = {'h_1': (h_start_min, h_start_max), "h_2": (h_end_min, h_end_max)}
    for i in range(9):
        target_dis_sonar1[i], theta_sonar1[i] = sg.search_theta_ground_truth(scan, theta_sonar1[i])
    for i in range(8):
        k_sonar1[i] = (target_dis_sonar1[i + 1] - target_dis_sonar1[i]) / (theta_sonar1[i + 1] - theta_sonar1[i])


def set_global_parameter2(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, Flag=False):
    global theta_sonar2
    global r
    global target_dis
    global h_min_2
    global h_max_2
    global start_theta_re2
    global end_theta_re2
    global sonar2
    global k_sonar2
    start_theta_re2 = start_theta
    end_theta_re2 = end_theta
    theta_sonar2 = np.zeros(9)
    r = np.zeros(2)
    target_dis = np.zeros(9)
    k_sonar2 = np.zeros(8)
    h_min_2 = np.zeros(2)
    h_max_2 = np.zeros(2)
    sonar2 = copy.deepcopy(sonar)
    # (sonar2.x, sonar2.y, sonar2.z)
    if Flag:
        for i in range(len(scan)):
            scan[i].sonar_axis_convert(sonar)
    theta_sonar2[0] = (end_theta - start_theta) * 20 / 180 + start_theta
    theta_sonar2[1] = (end_theta - start_theta) * 40 / 180 + start_theta
    theta_sonar2[2] = (end_theta - start_theta) * 50 / 180 + start_theta
    theta_sonar2[3] = (end_theta - start_theta) * 70 / 180 + start_theta
    theta_sonar2[4] = (end_theta - start_theta) * 90 / 180 + start_theta
    theta_sonar2[5] = (end_theta - start_theta) * 100 / 180 + start_theta
    theta_sonar2[6] = (end_theta - start_theta) * 120 / 180 + start_theta
    theta_sonar2[7] = (end_theta - start_theta) * 140 / 180 + start_theta
    theta_sonar2[8] = (end_theta - start_theta) * 160 / 180 + start_theta
    r[0] = dis1
    r[1] = dis2
    h_min_2[0] = -r[0] * np.sin(vertical_angle / 2) + sonar2.z
    h_min_2[1] = -r[1] * np.sin(vertical_angle / 2) + sonar2.z
    h_max_2[0] = r[0] * np.sin(vertical_angle / 2) + sonar2.z
    h_max_2[1] = r[1] * np.sin(vertical_angle / 2) + sonar2.z
    for i in range(9):
        target_dis[i], theta_sonar2[i] = sg.search_theta_ground_truth(scan, theta_sonar2[i], sonar2, True)
    for i in range(1, 9):
        k_sonar2[i - 1] = (target_dis[i] - target_dis[i - 1]) / (theta_sonar2[i] - theta_sonar2[i - 1])


def set_global_parameter3(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, Flag=False):
    global theta_sonar3
    global r_3
    global target_dis_3
    global h_min_3
    global h_max_3
    global start_theta_re3
    global end_theta_re3
    global sonar3
    global k_sonar3

    start_theta_re3 = start_theta
    end_theta_re3 = end_theta
    theta_sonar3 = np.zeros(9)
    r_3 = np.zeros(2)
    target_dis_3 = np.zeros(9)
    k_sonar3 = np.zeros(8)
    h_min_3 = np.zeros(2)
    h_max_3 = np.zeros(2)
    sonar3 = copy.deepcopy(sonar)
    # print(sonar.x, sonar.y, sonar.z)
    # print(sonar3.x, sonar3.y, sonar3.z)
    if Flag:
        for i in range(len(scan)):
            scan[i].sonar_axis_convert(sonar)
    theta_sonar3[0] = (end_theta - start_theta) * 20 / 180 + start_theta
    theta_sonar3[1] = (end_theta - start_theta) * 40 / 180 + start_theta
    theta_sonar3[2] = (end_theta - start_theta) * 50 / 180 + start_theta
    theta_sonar3[3] = (end_theta - start_theta) * 70 / 180 + start_theta
    theta_sonar3[4] = (end_theta - start_theta) * 90 / 180 + start_theta
    theta_sonar3[5] = (end_theta - start_theta) * 100 / 180 + start_theta
    theta_sonar3[6] = (end_theta - start_theta) * 120 / 180 + start_theta
    theta_sonar3[7] = (end_theta - start_theta) * 140 / 180 + start_theta
    theta_sonar3[8] = (end_theta - start_theta) * 160 / 180 + start_theta
    r_3[0] = dis1
    r_3[1] = dis2
    h_min_3[0] = -r_3[0] * np.sin(vertical_angle / 2) + sonar3.z
    h_min_3[1] = -r_3[1] * np.sin(vertical_angle / 2) + sonar3.z
    h_max_3[0] = r_3[0] * np.sin(vertical_angle / 2) + sonar3.z
    h_max_3[1] = r_3[1] * np.sin(vertical_angle / 2) + sonar3.z
    for i in range(9):
        target_dis_3[i], theta_sonar3[i] = sg.search_theta_ground_truth(scan, theta_sonar3[i], sonar3, True)
    for i in range(1, 9):
        k_sonar3[i - 1] = (target_dis_3[i] - target_dis_3[i - 1]) / (theta_sonar3[i] - theta_sonar3[i - 1])


def add_parameters(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, Flag=False):
    global theta_sonar_
    global r_
    global target_dis_
    global h_min_
    global h_max_
    global start_theta_re_
    global end_theta_re_
    global sonar_
    global k_sonar_
    start_theta_temp = start_theta
    end_theta_temp = end_theta
    theta_sonar_temp = np.zeros(9)
    r_temp = np.zeros(2)
    target_dis_temp = np.zeros(9)
    k_sonar_temp = np.zeros(8)
    h_min_temp = np.zeros(2)
    h_max_temp = np.zeros(2)
    sonar_temp = copy.deepcopy(sonar)
    # print(sonar.x, sonar.y, sonar.z)
    # print(sonar3.x, sonar3.y, sonar3.z)
    if Flag:
        for i in range(len(scan)):
            scan[i].sonar_axis_convert(sonar)
    theta_sonar_temp[0] = (end_theta - start_theta) * 20 / 180 + start_theta
    theta_sonar_temp[1] = (end_theta - start_theta) * 40 / 180 + start_theta
    theta_sonar_temp[2] = (end_theta - start_theta) * 50 / 180 + start_theta
    theta_sonar_temp[3] = (end_theta - start_theta) * 70 / 180 + start_theta
    theta_sonar_temp[4] = (end_theta - start_theta) * 90 / 180 + start_theta
    theta_sonar_temp[5] = (end_theta - start_theta) * 100 / 180 + start_theta
    theta_sonar_temp[6] = (end_theta - start_theta) * 120 / 180 + start_theta
    theta_sonar_temp[7] = (end_theta - start_theta) * 140 / 180 + start_theta
    theta_sonar_temp[8] = (end_theta - start_theta) * 160 / 180 + start_theta
    r_temp[0] = dis1
    r_temp[1] = dis2
    h_min_temp[0] = -r_temp[0] * np.sin(vertical_angle / 2) + sonar_temp.z
    h_min_temp[1] = -r_temp[1] * np.sin(vertical_angle / 2) + sonar_temp.z
    h_max_temp[0] = r_temp[0] * np.sin(vertical_angle / 2) + sonar_temp.z
    h_max_temp[1] = r_temp[1] * np.sin(vertical_angle / 2) + sonar_temp.z
    for i in range(9):
        target_dis_temp[i], theta_sonar_temp[i] = sg.search_theta_ground_truth(scan, theta_sonar_temp[i], sonar_temp,
                                                                               True)
    for i in range(1, 9):
        k_sonar_temp[i - 1] = (target_dis_temp[i] - target_dis_temp[i - 1]) / (theta_sonar3[i] - theta_sonar3[i - 1])
    if not theta_sonar_:
        theta_sonar_ = np.array([theta_sonar_temp])
        start_theta_re_ = np.array(start_theta_temp)
        end_theta_re_ = np.array(end_theta_temp)
        r_ = np.array([r_temp])
        target_dis_ = np.array([target_dis_temp])
        h_min_ = np.array([h_min_temp])
        h_max_ = np.array([h_max_temp])
        k_sonar_ = np.array([k_sonar_temp])
        sonar_ = {}
        sonar_.update({0: sonar_temp})
    else:
        theta_sonar_ = np.append(theta_sonar_, np.array([theta_sonar_temp]), axis=0)
        start_theta_re_ = np.append(start_theta_re_, np.array(start_theta_temp))
        end_theta_re_ = np.append(end_theta_re_, np.array(end_theta_temp))
        r_ = np.append(r_, np.array([r_temp]), axis=0)
        target_dis_ = np.append(target_dis_, np.array([target_dis_temp]), axis=0)
        h_min_ = np.append(h_min_, np.array([h_min_temp]), axis=0)
        h_max_ = np.append(h_max_, np.array([h_max_temp]), axis=0)
        k_sonar_ = np.append(k_sonar_, np.array([k_sonar_temp]), axis=0)
        sonar_.update({len(sonar_): sonar_temp})


def function_to_target_sonar2(h_1, h_2):
    global theta_sonar2
    global r
    global target_dis
    global start_theta_re2
    global end_theta_re2
    global sonar2
    global diff_k_2
    h_1 = h_1 - sonar2.z
    h_2 = h_2 - sonar2.z
    temp_record_2 = np.zeros(9)
    diff_k_2 = np.zeros(8)
    x = np.zeros(2)
    y = np.zeros(2)
    if h_1 > r[0] or h_2 > r[1]:
        return -np.inf
    x[0], y[0] = sg.x_y_cal(h_1, start_theta_re2, r[0])
    x[1], y[1] = sg.x_y_cal(h_2, end_theta_re2, r[1])
    diff_sonar2 = np.zeros(9)
    for i in range(9):
        temp_record_2[i] = sg.search_sonar_line_min_dis_theta(x[0], y[0], x[1], y[1], h_1, h_2, theta_sonar2[i])
        diff_sonar2[i] = abs(temp_record_2[i] - target_dis[i]) * 100
    for i in range(8):
        diff_k_2[i] = abs(
            (temp_record_2[i + 1] - temp_record_2[i]) / (theta_sonar2[i + 1] - theta_sonar2[i]) - k_sonar2[i]) * 100
    return -1 * np.sum(diff_sonar2) ** 2


def reset_parameters():
    global theta_sonar_
    global r_
    global target_dis_
    global h_min_
    global h_max_
    global start_theta_re_
    global end_theta_re_
    global sonar_
    global k_sonar_
    global diff_k_
    theta_sonar_ = []
    r_ = []
    target_dis_ = []
    h_min_ = []
    h_max_ = []
    start_theta_re_ = []
    end_theta_re_ = []
    sonar_ = {}
    k_sonar_ = []
    diff_k_ = []


def function_to_target(h_1, h_2):
    global r_1
    global r_2
    global start_theta_re
    global end_theta_re
    global theta_sonar1
    global target_dis_sonar1
    global k_sonar1
    global diff_k
    dis1_record = np.zeros(9)
    temp_record1 = np.zeros(9)
    diff_k = np.zeros(8)
    if h_1 > r_1 or h_2 > r_2:
        return -np.inf
    x_1, y_1 = sg.x_y_cal(h_1, start_theta_re, r_1)
    x_2, y_2 = sg.x_y_cal(h_2, end_theta_re, r_2)
    for i in range(9):
        temp_record1[i] = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_sonar1[i])
        dis1_record[i] = abs(temp_record1[i] - target_dis_sonar1[i]) * 100
    for i in range(8):
        diff_k[i] = abs(
            (temp_record1[i + 1] - temp_record1[i]) / (theta_sonar1[i + 1] - theta_sonar1[i]) - k_sonar1[i]) * 100
    return -1 * np.sum(dis1_record) ** 2


def function_to_target_sonar3(h_1, h_2):
    global theta_sonar3
    global r_3
    global target_dis_3
    global start_theta_re3
    global end_theta_re3
    global sonar3
    global diff_k_3
    h_1 = h_1 - sonar3.z
    h_2 = h_2 - sonar3.z
    temp_record_3 = np.zeros(9)
    diff_k_3 = np.zeros(8)
    x = np.zeros(2)
    y = np.zeros(2)
    if h_1 > r_3[0] or h_2 > r_3[1]:
        return -np.inf
    x[0], y[0] = sg.x_y_cal(h_1, start_theta_re3, r_3[0])
    x[1], y[1] = sg.x_y_cal(h_2, end_theta_re3, r_3[1])
    diff_sonar3 = np.zeros(9)
    for i in range(9):
        temp_record_3[i] = sg.search_sonar_line_min_dis_theta(x[0], y[0], x[1], y[1], h_1, h_2, theta_sonar3[i])
        diff_sonar3[i] = abs(temp_record_3[i] - target_dis_3[i]) * 100
    for i in range(8):
        diff_k_3[i] = abs(
            (temp_record_3[i + 1] - temp_record_3[i]) / (theta_sonar3[i + 1] - theta_sonar3[i]) - k_sonar3[i]) * 100
    return -1 * np.sum(diff_sonar3) ** 2


def function_to_target_several(h_1, h_2, n):
    global theta_sonar_
    global r_
    global target_dis_
    global start_theta_re_
    global end_theta_re_
    global sonar_
    global k_sonar_
    global diff_k_
    theta_sonar_temp = theta_sonar_[n]
    r_temp = r_[n]
    target_dis_temp = target_dis_[n]
    start_theta_temp = start_theta_re_[n]
    end_theta_temp = end_theta_re_[n]
    sonar_temp = sonar_[n]
    h_1 = h_1 - sonar_temp.z
    h_2 = h_2 - sonar_temp.z
    k_sonar_temp = k_sonar_[n]
    temp_record_temp = np.zeros(9)
    diff_k_temp = np.zeros(8)
    x = np.zeros(2)
    y = np.zeros(2)
    if h_1 > r_temp[0] or h_2 > r_temp[1]:
        return -np.inf
    x[0], y[0] = sg.x_y_cal(h_1, start_theta_temp, r_temp[0])
    x[1], y[1] = sg.x_y_cal(h_2, end_theta_temp, r_temp[1])
    diff_sonar_temp = np.zeros(9)
    for i in range(9):
        temp_record_temp[i] = sg.search_sonar_line_min_dis_theta(x[0], y[0], x[1], y[1], h_1, h_2, theta_sonar_temp[i])
        diff_sonar_temp[i] = abs(temp_record_temp[i] - target_dis_temp[i]) * 100
    for i in range(8):
        diff_k_temp[i] = abs(
            (temp_record_temp[i + 1] - temp_record_temp[i]) / (theta_sonar_temp[i + 1] - theta_sonar_temp[i]) -
            k_sonar_temp[i]) * 100
    if not diff_k_:
        diff_k_ = np.array([diff_k_temp])
    else:
        diff_k_ = np.append(diff_k_, np.array([diff_k_temp]), axis=0)
    return -1 * np.sum(diff_sonar_temp) ** 2


def function_to_target_for_sko(h):
    h_1, h_2 = h
    return -1 * function_to_target(h_1, h_2)


def function_to_target_sonar_k(h_1, h_2):
    global diff_k
    loss = function_to_target(h_1, h_2)
    return loss - np.sum(diff_k)


def function_to_target_sonar2_k(h_1, h_2):
    global diff_k_2
    loss = function_to_target_sonar2(h_1, h_2)
    return loss - np.sum(diff_k_2)


def function_to_target_sonar3_k(h_1, h_2):
    global diff_k_3
    loss = function_to_target_sonar3(h_1, h_2)
    return loss - np.sum(diff_k_3)


def function_to_target_sonar_more(h_1, h_2, k):
    count = 0
    loss_sum = 0
    while count < k:
        loss_sum += function_to_target_several(h_1, h_2, count)
        count += 1
    return loss_sum + function_to_target_3_sonar_dis(h_1, h_2)


def function_to_target_both_dis(h_1, h_2):
    return function_to_target(h_1, h_2) + function_to_target_sonar2(h_1, h_2)


def function_to_target_3_sonar_dis(h_1, h_2):
    return function_to_target(h_1, h_2) + function_to_target_sonar2(h_1, h_2) + function_to_target_sonar3(h_1, h_2)


def function_to_target_3_both(h_1, h_2):
    return function_to_target_sonar_k(h_1, h_2) + function_to_target_sonar2_k(h_1, h_2) + function_to_target_sonar3_k(
        h_1, h_2)


def function_to_target_both(h_1, h_2):
    global diff_k
    global diff_k_2
    loss1 = function_to_target(h_1, h_2)
    loss2 = function_to_target_sonar2(h_1, h_2)
    return loss1 + loss2 - np.sum(diff_k) - np.sum(diff_k_2)


def optimizer_line_pose(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar=None, flag=False):
    global pbounds
    set_global_parameter(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, flag)
    # print(pbounds)
    optimizer = BayesianOptimization(
        f=function_to_target,
        pbounds=pbounds,
        random_state=None
    )
    optimizer.maximize(
        init_points=20,
        n_iter=200,
        # acq='ucb',
        kappa=1,
    )
    return optimizer.max, optimizer.res


def optimizer_particle_swarm_line_pose(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar=None,
                                       flag=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_global_parameter(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, flag)
    optimizer_pso = PSO(func=function_to_target_for_sko, n_dim=2, pop=200, max_iter=1000, lb=lb, ub=ls, w=0.8, c1=0.8,
                        c2=0.8)

    optimizer_pso.run()
    return optimizer_pso.best_x, optimizer_pso.best_y


def optimizer_annel_line_pose(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar=None,
                              flag=False):
    set_global_parameter(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, flag)
    optimizer_annel = SA(func=function_to_target_for_sko, n_dim=2, x0=[h_start_min, h_end_min],
                         T_max=np.max(h_start_max, h_end_max), T_min=np.min(h_start_min, h_end_min), L=300,
                         max_stay_counter=150)
    best_x, best_y = optimizer_annel.run()
    return best_x, best_y


def optimizer_genetic_line_pose(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar=None, flag=False):
    set_global_parameter(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar, flag)
    optimizer_GA = GA(func=function_to_target_for_sko, n_dim=2, size_pop=200, max_iter=2000, prob_mut=0.01, lb=lb,
                      ub=ls,
                      precision=1e-7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer_GA.to(device)
    best_x, best_y = optimizer_GA.run()
    return best_x, best_y


def height_real_position(point, sonar):
    x, y, z = sonar.get_sonar_position()
    real_x = point.x
    real_y = point.y
    real_z = z + point.z
    return real_x, real_y, real_z
