import math
import numpy as np
import point


class circle:
    def __init__(self, d, num):
        self.d = d
        self.point_num = num
        self.x = np.array([0.0] * self.point_num)
        self.y = np.array([0.0] * self.point_num)
        self.z = np.array([0.0] * self.point_num)
        self.generate_circle_point()

    def generate_circle_point(self):
        interval_theta = 2.0 * np.pi / self.point_num
        for i in range(self.point_num):
            self.x[i] = self.d * np.cos(interval_theta * i)
            self.y[i] = self.d * np.sin(interval_theta * i)
            self.z[i] = 0
            print(self.x[i], self.y[i])

    def get_parameters(self):
        return self.d

    def get_circle_point(self):
        return self.x, self.y, self.z


class Line:
    def __init__(self, start_1, end_1, resolution):
        self.start = point.point(start_1.x, start_1.y, start_1.z)
        self.end = point.point(end_1.x, end_1.y, end_1.z)
        self.resolution = resolution
        self.length = math.sqrt(
            (self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2 + (self.end.z - self.start.z) ** 2)
        self.point_dic = {}
        self.min_dis_origin = np.inf
        self.min_dis_origin_theta = 0
        self.line_point()
        self.theta_line = math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
        self.fai_line = math.asin((self.end.z - self.start.z) / self.length)

    def get_point(self):
        return self.point_dic

    def get_position(self):
        return self.start, self.end

    def get_length(self):
        return self.length

    def clear_line(self):
        self.point_dic.clear()
        self.min_dis_origin = np.inf
        self.min_dis_origin_theta = 0

    def line_point(self):
        inter_x = (self.end.x - self.start.x) / (self.resolution + 1)
        inter_y = (self.end.y - self.start.y) / (self.resolution + 1)
        inter_z = (self.end.z - self.start.z) / (self.resolution + 1)
        self.point_dic.update({0: self.start})
        for i in range(1, self.resolution + 1):
            temp = point.point(self.start.x + inter_x * i, self.start.y + inter_y * i, self.start.z + inter_z * i)
            self.point_dic.update({i: temp})
            last_dis = self.min_dis_origin
            self.min_dis_origin = min(self.min_dis_origin, temp.r)
            if last_dis != self.min_dis_origin:
                self.min_dis_origin_theta = temp.theta
        self.point_dic.update({self.resolution + 1: self.end})

    def move_line(self, direction, distance, x_s=0, y_s=0, z_s=0, x_e=0, y_e=0, z_e=0):
        print(self.start.x, self.start.y, self.start.z, self.end.x, self.end.y, self.end.z)
        if direction == 'up':
            self.start.change_xyz(0, 0, distance)
            self.end.change_xyz(0, 0, distance)
        elif direction == 'down':
            self.start.change_xyz(0, 0, -1 * distance)
            self.end.change_xyz(0, 0, -1 * distance)
        elif direction == 'left':
            self.start.change_xyz(-1 * distance, 0, 0)
            self.end.change_xyz(-1 * distance, 0, 0)
        elif direction == 'right':
            self.start.change_xyz(distance, 0, 0)
            self.end.change_xyz(distance, 0, 0)
        elif direction == 'near':
            self.start.change_xyz(0, -1 * distance, 0)
            self.end.change_xyz(0, -1 * distance, 0)
        elif direction == 'away':
            self.start.change_xyz(0, distance, 0)
            self.end.change_xyz(0, distance, 0)
        else:
            z_change = math.sin(self.fai_line) * distance
            x_change = math.cos(self.theta_line) * math.cos(self.fai_line) * distance
            y_change = math.cos(self.fai_line) * distance * math.sin(self.theta_line)
            if direction == 'forward':
                self.start.change_xyz(x_change, y_change, z_change)
                self.end.change_xyz(x_change, y_change, z_change)
                print(z_change, x_change, y_change)
            elif direction == 'backward':
                self.start.change_xyz(-1 * x_change, -1 * y_change, -1 * z_change)
                self.end.change_xyz(-1 * x_change, -1 * y_change, -1 * z_change)
            else:
                self.start.change_xyz(x_s, y_s, z_s)
                self.end.change_xyz(x_e, y_e, z_e)
                # print(z_change, x_change, y_change)
        # print(self.start.x, self.start.y, self.start.z, self.end.x, self.end.y, self.end.z)
        self.clear_line()
        self.length = math.sqrt(
            (self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2 + (self.end.z - self.start.z) ** 2)
        self.line_point()
        self.theta_line = math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
        self.fai_line = math.asin((self.end.z - self.start.z) / self.length)


class sonar:

    def __init__(self, x, y, z, vertical=np.pi * 50 / 180, speed=np.pi / 360, range=40, current_angle=0,
                 angle_scan=np.pi / 180, end_angle=np.pi):
        self.x = x
        self.y = y
        self.z = z
        self.vertical_angel = vertical
        self.speed = speed
        self.range = range
        self.scan_result = {}
        self.scan_rotate_result = {}
        self.current_angle = current_angle
        self.beam_angle = angle_scan
        self.next_angle = angle_scan + self.current_angle
        self.resolution = self.range * 0.08
        self.end_angle = end_angle

    def get_result(self):
        return self.scan_result

    def get_sonar_position(self):
        return self.x, self.y, self.z

    def target2result(self, target):  # 绝对坐标
        dis = math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2 + (target.z - self.z) ** 2)
        if target.x != self.x:
            k = np.abs((target.y - self.y) / (target.x - self.x))
            if target.x > self.x:
                flag_x = 1
            else:
                flag_x = -1
            if target.y > self.y:
                flag_y = 1
            else:
                flag_y = -1
            temp = point.point(math.sqrt(1 / (1 + k ** 2)) * dis * flag_x + self.x,
                               math.sqrt(1 / (1 + k ** 2)) * k * dis * flag_y + self.y, self.z)
        else:
            if target.y > self.y:
                flag_y = 1
            else:
                flag_y = -1
            temp = point.point(self.x, dis * flag_y + self.y, self.z)
        return temp, dis

    def clear_result(self):
        self.scan_result = {}
        self.scan_rotate_result = {}

    def scaning_result_simple(self, line):
        target = line.get_point()
        length = len(target)
        for i in range(length):
            temp, _ = self.target2result(target[i])
            self.scan_result.update({i: temp})

    def scan_line_one_time(self, line):
        if self.current_angle > self.end_angle:
            return self.current_angle, self.current_angle, None, None
        result = np.array([])
        point_set = line.get_point()
        theta_record = np.array([])

        for i in range(len(point_set)):
            # print(point_set)
            if ((self.current_angle <= real_theta2relate_theta(point_set[i], self.x, self.y) <= self.next_angle) or (
                    self.current_angle <= (
                    2 * np.pi + real_theta2relate_theta(point_set[i], self.x, self.y)) <= self.next_angle)) \
                    and -1 * self.vertical_angel / 2 <= real_fai2relate_fai(point_set[i], self.x, self.y,
                                                                            self.z) <= self.vertical_angel / 2 and \
                    point_set[i].r <= self.range:
                _, temp = self.target2result(point_set[i])
                result = np.append(result, temp)
            else:
                continue
        theta_result = np.array([self.current_angle, self.next_angle])
        if len(result) == 0:
            result = []
        else:
            theta_record.sort()
            result.sort()
        if self.next_angle + self.speed <= self.end_angle:
            temp_next = self.next_angle + self.speed
        else:
            temp_next = self.end_angle
        return self.current_angle + self.speed, temp_next, result, theta_result

    def get_scan_result(self):
        return self.scan_rotate_result, self.scan_result

    def scan_line(self, line):
        index = 0
        while self.current_angle <= self.end_angle:
            self.current_angle, self.next_angle, re1, theta_record = self.scan_line_one_time(line)
            if re1 == np.array([]):
                continue
            self.scan_rotate_result.update({index: [re1, theta_record]})
            index += 1
        self.current_angle = 0
        self.next_angle = self.current_angle + self.beam_angle
        return self.scan_rotate_result


def list_position_ndarray(target):
    x_t = [0] * len(target)
    y_t = [0] * len(target)
    z_t = [0] * len(target)
    for i in range(len(target)):
        x_t[i] = target[i].x
        y_t[i] = target[i].y
        z_t[i] = target[i].z
    x_t = np.array(x_t)
    y_t = np.array(y_t)
    z_t = np.array(z_t)
    return x_t, y_t, z_t


def sonar_line_edge(Sonar, target):
    x_start_line = np.array([Sonar.x, target[0].x])
    y_start_line = np.array([Sonar.y, target[0].y])
    z_start_line = np.array([Sonar.z, target[0].z])
    x_end_line = np.array([Sonar.x, target[len(target) - 1].x])
    y_end_line = np.array([Sonar.y, target[len(target) - 1].y])
    z_end_line = np.array([Sonar.z, target[len(target) - 1].z])
    return x_start_line, y_start_line, z_start_line, x_end_line, y_end_line, z_end_line


def surface_point(dis, theta, sonar, resolution=False):
    dic = {}
    dic_index = 0
    begin_theta = theta[0]
    end_theta = theta[1]
    number = int(abs(end_theta - begin_theta) * 500)
    x, y, z = sonar.get_sonar_position()
    if number == 0:
        return dic
    interval_theta = (end_theta - begin_theta) / number
    # print(theta)
    if not resolution:
        for i in range(len(dis)):
            temp = {}
            temp_index = 0
            for j in range(number):
                x_index = dis[i] * np.cos(begin_theta + j * interval_theta) + x
                y_index = dis[i] * np.sin(begin_theta + j * interval_theta) + y
                add_point = point.point(x_index, y_index, z)
                temp.update({temp_index: add_point})
                temp_index += 1
            dic.update({dic_index: temp})
            dic_index += 1
    else:
        a = 0
    return dic


def real_theta2relate_theta(point, x, y):
    relate_x = point.x - x
    relate_y = point.y - y
    theta = math.atan2(relate_y, relate_x)
    return theta


def real_fai2relate_fai(point, x, y, z):
    dis_relate = np.sqrt((point.x - x) ** 2 + (point.y - y) ** 2 + (point.z - z) ** 2)
    relate_z = point.z - z
    fai = math.asin(relate_z / dis_relate)
    return fai


def move_line(start, end, line, direction, distance, x_s=0, y_s=0, z_s=0, x_e=0, y_e=0, z_e=0):
    # print(start.x, start.y, start.z, end.x, end.y, end.z)
    if direction == 'up':
        start.change_xyz(0, 0, distance)
        end.change_xyz(0, 0, distance)
    elif direction == 'down':
        start.change_xyz(0, 0, -1 * distance)
        end.change_xyz(0, 0, -1 * distance)
    elif direction == 'left':
        start.change_xyz(-1 * distance, 0, 0)
        end.change_xyz(-1 * distance, 0, 0)
    elif direction == 'right':
        start.change_xyz(distance, 0, 0)
        end.change_xyz(distance, 0, 0)
    elif direction == 'near':
        start.change_xyz(0, -1 * distance, 0)
        end.change_xyz(0, -1 * distance, 0)
    elif direction == 'away':
        start.change_xyz(0, distance, 0)
        end.change_xyz(0, distance, 0)
    else:
        z_change = math.sin(line.fai_line) * distance
        x_change = math.cos(line.theta_line) * math.cos(line.fai_line) * distance
        y_change = math.cos(line.fai_line) * distance * math.sin(line.theta_line)
        if direction == 'forward':
            start.change_xyz(x_change, y_change, z_change)
            end.change_xyz(x_change, y_change, z_change)
            print(z_change, x_change, y_change)
        elif direction == 'backward':
            start.change_xyz(-1 * x_change, -1 * y_change, -1 * z_change)
            end.change_xyz(-1 * x_change, -1 * y_change, -1 * z_change)
        else:
            start.change_xyz(x_s, y_s, z_s)
            end.change_xyz(x_e, y_e, z_e)
            # print(z_change, x_change, y_change)
        # print(self.start.x, self.start.y, self.start.z, self.end.x, self.end.y, self.end.z)
