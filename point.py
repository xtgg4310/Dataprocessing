import math
import numpy as np


class point:
    def __init__(self, x=0.0, y=0.0, z=0.0, sonar_x=0, sonar_y=0, sonar_z=0):
        self.x = x
        self.y = y
        self.z = z
        self.r = 0
        self.theta = 0
        self.fai = 0
        self.convert_xyz_r()
        self.sonar_x = sonar_x
        self.sonar_y = sonar_y
        self.sonar_z = sonar_z
        self.sonar_r = np.sqrt(self.sonar_x ** 2 + self.sonar_y ** 2 + self.sonar_z ** 2)
        self.sonar_theta = math.atan2(self.sonar_y, self.sonar_x)
        #self.sonar_fai = math.asin(self.sonar_z / self.sonar_r)

    def show_xyz_position(self):
        print(self.x, self.y, self.z)

    def show_r_position(self):
        print(self.r, self.theta, self.fai)

    def get_xyz_position(self):
        return self.x, self.y, self.z

    def get_r_position(self):
        return self.r, self.theta, self.fai

    def sonar_axis_convert(self, sonar):
        self.sonar_x = self.x - sonar.x
        self.sonar_y = self.y - sonar.y
        self.sonar_z = self.z - sonar.z
        self.sonar_r = np.sqrt(self.sonar_x ** 2 + self.sonar_y ** 2 + self.sonar_z ** 2)
        self.sonar_theta = math.atan2(self.sonar_y, self.sonar_x)
        #self.sonar_fai = math.asin(self.sonar_z / self.sonar_r)

    def convert_xyz_r(self):
        self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.theta = math.atan2(self.y, self.x)
        self.fai = math.asin(self.z / self.r)

    def change_xyz(self, x, y, z):
        self.x = self.x + x
        self.y = self.y + y
        self.z = self.z + z
        self.convert_xyz_r()
