import math
import itertools
import pickle
import numpy as np
from map_plan import MapPlan
from image_tree import ImageTree
from walls_representation import Point


class MapBuilder:
    cell_size = 1  # in meters
    P0 = -53  # dBm
    reflection_coef = -12.11 * 0.7  # dBm
    attenuation_exponent = 5.0  # attenuation exponent
    EPS = 1E-9

    def __init__(self, file_name):
        self.file_name = file_name
        data = self.read_data()
        self.map_plan = MapPlan(data)
        self.image_tree = ImageTree(self.map_plan.AP, self.map_plan.walls)
        self.m = int(round(math.fabs(self.map_plan.boundaries[1].p1.x - self.map_plan.boundaries[1].p3.x)))
        self.n = int(round(math.fabs(self.map_plan.boundaries[0].p1.y - self.map_plan.boundaries[0].p3.y)))
        self.signal_strength_matrix = np.zeros((self.m, self.n)).astype(np.int32)  # TODO
        print()

    def read_data(self):
        with open(self.file_name, 'r') as file:
            return file.readlines()

    @staticmethod
    def calculate_traversed_distance(a, b):
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)

    @staticmethod
    def get_direction_vector(a, b):
        return [b.x - a.x, b.y - a.y, b.z - a.z]

    @staticmethod
    def check_if_point_belongs_to_line_segment(a, b, p):
        return ((a.x - p.x) * (b.x - p.x) + (a.y - p.y) * (b.y - p.y) + (a.z - p.z) * (b.z - p.z)) <= 0

    @staticmethod
    def scalar_product(point, dir_vector):
        return point.x * dir_vector[0] + point.y * dir_vector[1] + point.z * dir_vector[2]

    def check_if_point_belongs_to_wall(self, p, wall):
        if self.scalar_product(wall.p1, self.get_direction_vector(wall.p1, wall.p2)) <= \
                self.scalar_product(p, self.get_direction_vector(wall.p1, wall.p2)) <= \
                self.scalar_product(wall.p2, self.get_direction_vector(wall.p1, wall.p2)) and \
                                self.scalar_product(wall.p1, self.get_direction_vector(wall.p1, wall.p4)) <= \
                                self.scalar_product(p, self.get_direction_vector(wall.p1, wall.p4)) <= \
                        self.scalar_product(wall.p4, self.get_direction_vector(wall.p1, wall.p4)):
            return True
        return False

    # with a line segment and a plane
    def get_intersection_point(self, p1, p2, wall):
        intersect_epsilon = 0.1
        a = wall.plane_equation[0]
        b = wall.plane_equation[1]
        c = wall.plane_equation[2]
        direction_vector = self.get_direction_vector(p1, p2)

        num = (wall.p1.x - p1.x) * a + (wall.p1.y - p1.y) * b + (wall.p1.z - p1.z) * c
        denom = direction_vector[0] * a + direction_vector[1] * b + direction_vector[2] * c

        if denom == 0:
            return -1
        t = num / denom
        if t > intersect_epsilon:
            p = Point(0, 0, 0)
            p.x = p1.x + direction_vector[0] * t
            p.y = p1.y + direction_vector[1] * t
            p.z = p1.z + direction_vector[2] * t

            if math.fabs(p.x) < self.EPS: p.x = 0
            if math.fabs(p.y) < self.EPS: p.y = 0
            if math.fabs(p.z) < self.EPS: p.z = 0
            if p1.is_equal(p) or p2.is_equal(p):
                return -1
            if self.check_if_point_belongs_to_line_segment(p1, p2, p):
                if self.check_if_point_belongs_to_wall(p, wall):
                    return p
        return -1

    def calculate_reflection_paths(self, last_layer, reflection_number, Tx, Rx):
        paths = []
        ray_distance_threshold = 1000  # meters

        if reflection_number == 0:
            traversed_distance = self.calculate_traversed_distance(Tx, Rx) * self.cell_size
            # if traversed distance is too long - skip this path
            if traversed_distance > ray_distance_threshold:
                return paths
            # else find this path if exists
            paths = [[Rx, Tx, traversed_distance]]  # path[-1] contains traversed distance
        else:
            for i in last_layer:
                image_point = self.image_tree.image_tree.tree[i].data
                traversed_distance = self.calculate_traversed_distance(Tx, image_point) * self.cell_size
                # if traversed distance is too long - skip this path
                if traversed_distance > ray_distance_threshold:
                    continue

                # else find this path if exists
                path = [traversed_distance]  # path[0] contains traversed distance
                is_correct = True
                path.append(Tx)
                wall = image_point.assigned_wall
                intersection_point = self.get_intersection_point(Tx, image_point, wall)
                if intersection_point != -1:
                    path.append(intersection_point)
                else:
                    path.clear()
                    continue

                # this will execute for 2+ reflections
                image_point_ = self.image_tree.image_tree.tree[i]
                for j in range(2, reflection_number + 1):
                    image_point = image_point_.parent.data
                    image_point_ = image_point_.parent
                    wall = image_point.assigned_wall
                    intersection_point = self.get_intersection_point(path[-1], image_point, wall)
                    if intersection_point != -1:
                        is_correct = True
                        path.append(intersection_point)
                    else:
                        path.clear()
                        is_correct = False
                        break
                if is_correct:
                    path.append(Rx)
                    path.reverse()  # path[-1] contains traversed distance
                    paths.append(path)

        return paths

    def get_all_paths(self, Tx, Rx):
        reflection_number = 0
        paths = [self.calculate_reflection_paths(0, reflection_number, Tx, Rx)]
        reflection_number = 1
        paths.append(self.calculate_reflection_paths(self.image_tree.image_tree.get_children_indices(0),
                                                     reflection_number, Tx, Rx))

        return list(itertools.chain.from_iterable(paths))

    # for one position
    def get_signal_strength(self, Tx, Rx):
        if Tx.is_equal(Rx):
            return 0
        paths = self.get_all_paths(Tx, Rx)

        # calculate signal strength for each path, then sum
        signal_strength = 0

        for path in paths:
            num_of_reflections = len(path) - 3  # len(path) minus start point, end point, path length stored in path[-1]
            traversed_distance = path[-1]
            signal_strength += 10 ** ((self.P0 - 10 * self.attenuation_exponent * math.log10(traversed_distance) -
                                       self.reflection_coef * num_of_reflections) / 10)  # sum in mW
        try:
            signal_strength_dBm = 10 * math.log10(signal_strength)  # to dBm
        except ValueError:
            return 0
        return int(round(signal_strength_dBm))

    def build_coverage_map(self, z=1, serialize=True):
        for x in range(0, self.m):
            for y in range(0, self.n):
                self.signal_strength_matrix[x][y] = self.get_signal_strength(Point(x, y, z), self.map_plan.AP)
            print(x)

        if serialize:
            self.serialize(self.signal_strength_matrix, "data")

    @staticmethod
    def serialize(data, filename="data"):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
