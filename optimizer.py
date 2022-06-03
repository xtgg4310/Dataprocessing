from bayes_opt import BayesianOptimization
import point
import signal_process as sg
import numpy as np
import simulation as sl
from sko.PSO import PSO
from sko.SA import SA
from sko.GA import GA
import torch
theta_1 = 0
theta_2 = 0
theta_3 = 0
theta_4 = 0
theta_5 = 0
theta_6 = 0
theta_7 = 0
theta_8 = 0
theta_9 = 0
theta2_1 = 0
theta2_2 = 0
theta2_3 = 0
theta2_4 = 0
theta2_5 = 0

r_1 = 0
r_2 = 0
target_dis_1 = 0
target_dis_2 = 0
target_dis_3 = 0
target_dis_4 = 0
target_dis_5 = 0
target_dis_6 = 0
target_dis_7 = 0
target_dis_8 = 0
target_dis_9 = 0
target2_dis_1 = 0
target2_dis_2 = 0
target2_dis_3 = 0
target2_dis_4 = 0
target2_dis_5 = 0
h_start_min = 0
h_start_max = 0
h_end_min = 0
h_end_max = 0
pbounds = {}
start_theta_re = 0
end_theta_re = 0
lb = [0, 0]
ls = [0, 0]


def set_global_parameter(start_theta, end_theta, dis1, dis2, scan, vertical_angle, sonar=None, Flag=False):
    global theta_1
    global theta_2
    global theta_3
    global theta_4
    global theta_5
    global theta_6
    global theta_7
    global theta_8
    global theta_9
    global theta2_1
    global theta2_2
    global theta2_3
    global theta2_4
    global theta2_5
    global r_1
    global r_2
    global target_dis_1
    global target_dis_2
    global target_dis_3
    global target_dis_4
    global target_dis_5
    global target_dis_6
    global target_dis_7
    global target_dis_8
    global target_dis_9
    global target2_dis_1
    global target2_dis_2
    global target2_dis_3
    global target2_dis_4
    global target2_dis_5
    global h_start_min
    global h_start_max
    global h_end_min
    global h_end_max
    global pbounds
    global start_theta_re
    global end_theta_re
    global lb
    global ls
    if Flag:
        for i in range(len(scan)):
            scan[i].sonar_axis_convert(sonar)
    start_theta_re = start_theta
    end_theta_re = end_theta
    theta_1 = (end_theta - start_theta) * 20 / 180 + start_theta
    theta_2 = (end_theta - start_theta) * 40 / 180 + start_theta
    theta_3 = (end_theta - start_theta) * 50 / 180 + start_theta
    theta_4 = (end_theta - start_theta) * 70 / 180 + start_theta
    theta_5 = (end_theta - start_theta) * 90 / 180 + start_theta
    theta_6 = (end_theta - start_theta) * 110 / 180 + start_theta
    theta_7 = (end_theta - start_theta) * 130 / 180 + start_theta
    theta_8 = (end_theta - start_theta) * 150 / 180 + start_theta
    theta_9 = (end_theta - start_theta) * 170 / 180 + start_theta
    theta2_1 = (end_theta - start_theta) * 35 / 180 + start_theta
    theta2_2 = (end_theta - start_theta) * 60 / 180 + start_theta
    theta2_3 = (end_theta - start_theta) * 90 / 180 + start_theta
    theta2_4 = (end_theta - start_theta) * 120 / 180 + start_theta
    theta2_5 = (end_theta - start_theta) * 145 / 180 + start_theta
    r_1 = dis1
    r_2 = dis2
    h_start_min = 0
    h_end_min = -r_2 * np.sin(vertical_angle / 2)
    h_start_max = r_1 * np.sin(vertical_angle / 2)
    h_end_max = r_2 * np.sin(vertical_angle / 2)
    lb = [h_start_min, h_end_min]
    ls = [h_start_max, h_end_max]
    pbounds = {'h_1': (h_start_min, h_start_max), "h_2": (h_end_min, h_end_max)}
    target_dis_1, theta_1 = sg.search_theta_ground_truth(scan, theta_1)
    target_dis_2, theta_2 = sg.search_theta_ground_truth(scan, theta_2)
    target_dis_3, theta_3 = sg.search_theta_ground_truth(scan, theta_3)
    target_dis_4, theta_4 = sg.search_theta_ground_truth(scan, theta_4)
    target_dis_5, theta_5 = sg.search_theta_ground_truth(scan, theta_5)
    target_dis_6, theta_6 = sg.search_theta_ground_truth(scan, theta_6)
    target_dis_7, theta_7 = sg.search_theta_ground_truth(scan, theta_7)
    target_dis_8, theta_8 = sg.search_theta_ground_truth(scan, theta_8)
    target_dis_9, theta_9 = sg.search_theta_ground_truth(scan, theta_9)
    target2_dis_1, theta2_1 = sg.search_theta_ground_truth(scan, theta2_1)
    target2_dis_2, theta2_2 = sg.search_theta_ground_truth(scan, theta2_2)
    target2_dis_3, theta2_3 = sg.search_theta_ground_truth(scan, theta2_3)
    target2_dis_4, theta2_4 = sg.search_theta_ground_truth(scan, theta2_4)
    target2_dis_5, theta2_5 = sg.search_theta_ground_truth(scan, theta2_5)


def function_to_target(h_1, h_2):
    global theta_1
    global theta_2
    global theta_3
    global theta_4
    global theta_5
    global theta_6
    global theta_7
    global theta_8
    global theta_9
    global r_1
    global r_2
    global target_dis_1
    global target_dis_2
    global target_dis_3
    global target_dis_4
    global target_dis_5
    global target_dis_6
    global target_dis_7
    global start_theta_re
    global end_theta_re
    x_1, y_1 = sg.x_y_cal(h_1, start_theta_re, r_1)
    x_2, y_2 = sg.x_y_cal(h_2, end_theta_re, r_2)
    dis_1 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_1)
    dis_2 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_2)
    dis_3 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_3)
    dis_4 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_4)
    dis_5 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_5)
    dis_6 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_6)
    dis_7 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_7)
    dis_8 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_8)
    dis_9 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_9)
    return -1 * ((abs(dis_1 - target_dis_1) + abs(
        dis_2 - target_dis_2) + abs(dis_3 - target_dis_3) + abs(dis_4 - target_dis_4) + abs(
        dis_5 - target_dis_5) + abs(dis_6 - target_dis_6) + abs(
        dis_7 - target_dis_7) + abs(dis_8 - target_dis_8) + abs(dis_9 - target_dis_9)) * 100)


def function_to_target_2(h_1, h_2):
    global theta_1
    global theta_7
    global theta_3
    global theta_6
    global theta_5
    global r_1
    global r_2
    global target_dis_1
    global target_dis_7
    global target_dis_3
    global target_dis_6
    global target_dis_5
    global start_theta_re
    global end_theta_re
    x_1, y_1 = sg.x_y_cal(h_1, start_theta_re, r_1)
    x_2, y_2 = sg.x_y_cal(h_2, end_theta_re, r_2)
    dis_1 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_1)
    dis_7 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_7)
    dis_3 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_3)
    dis_6 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_6)
    dis_5 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta_5)
    return -1 * ((abs(dis_1 - target_dis_1) + abs(
        dis_7 - target_dis_7) + abs(dis_3 - target_dis_3) + abs(dis_6 - target_dis_6) + abs(
        dis_5 - target_dis_5)) * 100) ** 2


def function_to_target_3(h_1, h_2):
    global theta2_1
    global theta2_2
    global theta2_3
    global theta2_4
    global theta2_5
    global r_1
    global r_2
    global target2_dis_1
    global target2_dis_2
    global target2_dis_3
    global target2_dis_4
    global target2_dis_5
    global start_theta_re
    global end_theta_re
    x_1, y_1 = sg.x_y_cal(h_1, start_theta_re, r_1)
    x_2, y_2 = sg.x_y_cal(h_2, end_theta_re, r_2)
    dis_1 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta2_1)
    dis_2 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta2_2)
    dis_3 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta2_3)
    dis_4 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta2_4)
    dis_5 = sg.search_sonar_line_min_dis_theta(x_1, y_1, x_2, y_2, h_1, h_2, theta2_5)
    return -1 * ((abs(dis_1 - target2_dis_1)
                  # + abs(dis_2 - target2_dis_2)
                  + abs(dis_3 - target2_dis_3)
                  # + abs(dis_4 - target2_dis_4)
                  + abs(dis_5 - target2_dis_5)) * 100) ** 2


def function_to_target_for_sko(h):
    h_1, h_2 = h
    return -1*function_to_target(h_1, h_2)


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
        init_points=10,
        n_iter=200,
        kappa=1,
    )
    print(function_to_target(1, 1.5))
    print(function_to_target(optimizer.max['params']['h_1'], optimizer.max['params']['h_2']))
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
    optimizer_GA = GA(func=function_to_target_for_sko, n_dim=2, size_pop=200, max_iter=2000, prob_mut=0.01, lb=lb, ub=ls,
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
