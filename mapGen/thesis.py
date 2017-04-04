# http://stackoverflow.com/questions/15797920/how-to-convert-wifi-signal-strength-from-quality-percent-to-rssi-dbm
# dBm to percentage
import timeit

import math
import itertools
import numpy as np

import plotly
import plotly.graph_objs as go

import pickle

EPS = 1E-9
P0 = -35  # dBm
reflection_coef = -12.11  # dBm
transmission_coef = -0.4937  # dBm
n = 3.0  # attenuation exponent (dBm)


# reflection_coef = 0.25
# transmission_coef = 0.5
# Tx - user, Rx - AP

# Define classes for Point, Wall, Room and Building instances
class Point:
    x = 0
    y = 0
    z = 0

    assigned_wall = None  # to which wall image point is assigned
    assigned_room = None  # assigned room to AP

    def __init__(self, x_, y_, z_):
        self.x = x_
        self.y = y_
        self.z = z_

    def __lt__(self, other):
        self_ = (self.x ** 2) + (self.y ** 2)
        other_ = (other.x ** 2) + (other.y ** 2)
        return self_ < other_

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def is_equal(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ') '


class Wall:
    wall_number = 0
    p1 = Point
    p2 = Point
    p3 = Point
    p4 = Point

    def __init__(self, p1_, p2_, p3_, p4_, num):
        self.wall_number = num
        self.p1 = Point(p1_.x, p1_.y, p1_.z)
        self.p2 = Point(p2_.x, p2_.y, p2_.z)
        self.p3 = Point(p3_.x, p3_.y, p3_.z)
        self.p4 = Point(p4_.x, p4_.y, p4_.z)
        self.plane_equation = self.get_plane_equation()

    def __str__(self):
        print('Wall ' + str(self.wall_number) + ':')
        print(self.p1)
        print(self.p2)
        print(self.p3)
        print(self.p4)
        return ''

    def get_plane_equation(self):
        list = []
        list.append(
            det(self.p2.y - self.p1.y, self.p3.y - self.p1.y, self.p2.z - self.p1.z, self.p3.z - self.p1.z))
        list.append(
            -det(self.p2.x - self.p1.x, self.p3.x - self.p1.x, self.p2.z - self.p1.z, self.p3.z - self.p1.z))
        list.append(
            det(self.p2.x - self.p1.x, self.p3.x - self.p1.x, self.p2.y - self.p1.y, self.p3.y - self.p1.y))
        list.append(-self.p1.x * list[0] - self.p1.y * list[1] - self.p1.z * list[2])
        return list


class Room:
    def __init__(self, length_, width_, height_, wall1, wall2, wall3, wall4, ceil, floor):
        # walls
        self.length = length_
        self.width = width_
        self.height = height_
        self.walls = [wall1, wall2, wall3, wall4, ceil, floor]
        self.uvw = [wall4.p1 - wall1.p1,
                    wall1.p2 - wall1.p1,
                    wall1.p4 - wall1.p1]

    def __str__(self):
        for wall in self.walls:
            print(wall)
        return ''


class Building455:
    # AP = Point(2, 5, 1)
    # number_of_rooms = 2
    # _3D_measures = [15, 17, 4]

    AP = Point(7, 2, 4)
    number_of_rooms = 2
    _3D_measures = [12, 11, 6]

    def __init__(self):
        self.rooms = []
        self.rooms.append(Room(8, 6, 4,
                               # walls
                               Wall(Point(3, 0, 0), Point(12, 0, 0), Point(12, 0, 5), Point(3, 0, 5), 1),
                               Wall(Point(12, 0, 0), Point(12, 7, 0), Point(12, 7, 5), Point(12, 0, 5), 2),
                               Wall(Point(12, 7, 0), Point(3, 7, 0), Point(3, 7, 5), Point(12, 7, 5), 3),
                               Wall(Point(3, 7, 0), Point(3, 0, 0), Point(3, 0, 5), Point(3, 7, 5), 4),
                               # ceil
                               Wall(Point(3, 0, 5), Point(12, 0, 5), Point(12, 7, 5), Point(3, 7, 5), 5),
                               # floor
                               Wall(Point(3, 0, 0), Point(12, 0, 0), Point(12, 7, 0), Point(3, 7, 0), 6)))

        self.rooms.append(Room(2, 3, 4,
                               # walls
                               Wall(Point(0, 7, 0), Point(3, 7, 0), Point(3, 7, 5), Point(0, 7, 5), 7),
                               Wall(Point(3, 7 , 0), Point(3, 11, 0), Point(3, 11, 5), Point(3, 7, 5), 8),
                               Wall(Point(3, 11, 0), Point(0, 11, 0), Point(0, 11, 5), Point(3, 11, 5), 9),
                               Wall(Point(0, 11, 0), Point(0, 7, 0), Point(0, 7, 5), Point(0, 11, 5), 10),
                               # ceil
                               Wall(Point(0, 7, 5), Point(3, 7, 5), Point(3 ,11, 5), Point(0, 11, 5), 11),
                               # floor
                               Wall(Point(0, 7, 0), Point(3, 7, 0), Point(3, 11, 0), Point(0, 11, 0), 12)))

        '''
        def __init__(self):
        self.rooms = []
        self.rooms.append(Room(3, 6, 4,
                               # walls
                               Wall(Point(1, 0, 0), Point(9, 0, 0), Point(9, 0, 4), Point(1, 0, 4), 1),
                               Wall(Point(9, 0, 0), Point(9, 6, 0), Point(9, 6, 4), Point(9, 0, 4), 2),
                               Wall(Point(9, 6, 0), Point(1, 6, 0), Point(1, 6, 4), Point(9, 6, 4), 3),
                               Wall(Point(1, 6, 0), Point(1, 0, 0), Point(1, 0, 4), Point(1, 6, 4), 4),
                               # ceil
                               Wall(Point(1, 0, 4), Point(9, 0, 4), Point(9, 6, 4), Point(1, 6, 4), 5),
                               # floor
                               Wall(Point(1, 0, 0), Point(9, 0, 0), Point(9, 6, 0), Point(1, 6, 0), 6)))

        self.rooms.append(Room(2, 3, 4,
                               # walls
                               Wall(Point(0, 6, 0), Point(2, 6, 0), Point(2, 6, 4), Point(0, 6, 4), 7),
                               Wall(Point(2, 6 , 0), Point(2, 9, 0), Point(2, 9, 4), Point(2, 6, 4), 8),
                               Wall(Point(2, 9, 0), Point(0, 9, 0), Point(0, 9, 4), Point(2, 9, 4), 9),
                               Wall(Point(0, 9, 0), Point(0, 6, 0), Point(0, 6, 4), Point(0, 9, 4), 10),
                               # ceil
                               Wall(Point(0, 6, 4), Point(2, 6, 4), Point(2 ,9, 4), Point(0, 9 , 4), 11),
                               # floor
                               Wall(Point(0, 6, 0), Point(2, 6, 0), Point(2, 9, 0), Point(0, 9, 0), 12)))
        '''

        '''self.rooms.append(Room(15, 10, 4,
                               # walls
                               Wall(Point(0, 0, 0), Point(15, 0, 0), Point(15, 0, 4), Point(0, 0, 4), 1),
                               Wall(Point(15, 0, 0), Point(15, 10, 0), Point(15, 10, 4), Point(15, 0, 4), 2),
                               Wall(Point(15, 10, 0), Point(0, 10, 0), Point(0, 10, 4), Point(15, 10, 4), 3),
                               Wall(Point(0, 10, 0), Point(0, 0, 0), Point(0, 0, 4), Point(0, 10, 4), 4),
                               # ceil
                               Wall(Point(0, 0, 4), Point(15, 0, 4), Point(15, 10, 4), Point(0, 10, 4), 5),
                               # floor
                               Wall(Point(0, 0, 0), Point(15, 0, 0), Point(15, 10, 0), Point(0, 10, 0), 6)))
        self.rooms.append(Room(6, 7, 4,
                               # walls
                               Wall(Point(0, 10, 0), Point(6, 10, 0), Point(6, 10, 4), Point(0, 10, 4), 7),
                               Wall(Point(6, 10, 0), Point(6, 17, 0), Point(6, 17, 4), Point(6, 10, 4), 8),
                               Wall(Point(6, 17, 0), Point(0, 17, 0), Point(0, 17, 4), Point(6, 17, 4), 9),
                               Wall(Point(0, 17, 0), Point(0, 10, 0), Point(0, 10, 4), Point(17, 10, 4), 10),
                               # ceil
                               Wall(Point(0, 10, 4), Point(6, 10, 4), Point(6, 17, 4), Point(0, 17, 4), 11),
                               # floor
                               Wall(Point(0, 10, 0), Point(6, 10, 0), Point(6, 17, 0), Point(0, 17, 0), 12)))
        '''
        all_walls = []
        for room in self.rooms:
            all_walls.append(room.walls)
        self.all_walls = list(itertools.chain.from_iterable(all_walls))

        self.AP.assigned_room = self.rooms[0]

    def __str__(self):
        for room in self.rooms:
            print(room)
        return ''


class Building503:
    AP = Point(4, 7, 0)  # AP = Point(4, 7, 1)
    number_of_rooms = 2
    _3D_measures = [15, 11, 6]

    def __init__(self):
        self.rooms = []
        self.rooms.append(Room(6, 9, 4,
                               # walls
                               Wall(Point(0, 0, 0), Point(7, 0, 0), Point(7, 0, 5), Point(0, 0, 5), 1),
                               Wall(Point(7, 0, 0), Point(7, 10, 0), Point(7, 10, 5), Point(7, 0, 5), 2),
                               Wall(Point(7, 10, 0), Point(0, 10, 0), Point(0, 10, 5), Point(7, 10, 5), 3),
                               Wall(Point(0, 10, 0), Point(0, 0, 0), Point(0, 0, 5), Point(0, 10, 5), 4),
                               # ceil
                               Wall(Point(0, 0, 5), Point(7, 0, 5), Point(7, 10, 5), Point(0, 10, 5), 5),
                               # floor
                               Wall(Point(0, 0, 0), Point(7, 0, 0), Point(7, 10, 0), Point(0, 10, 0), 6)))
        self.rooms.append(Room(6, 9, 4,
                               # walls
                               Wall(Point(7, 0, 0), Point(14, 0, 0), Point(10, 0, 5), Point(7, 0, 5), 7),
                               Wall(Point(14, 0, 0), Point(14, 10, 0), Point(14, 10, 5), Point(14, 0, 5), 8),
                               Wall(Point(14, 10, 0), Point(7, 10, 0), Point(7, 10, 5), Point(14, 10, 5), 9),
                               Wall(Point(7, 10, 0), Point(7, 0, 0), Point(7, 0, 5), Point(7, 10, 5), 10),
                               # ceil
                               Wall(Point(7, 0, 5), Point(14, 0, 5), Point(14, 10, 5), Point(7, 10, 5), 11),
                               # floor
                               Wall(Point(7, 0, 0), Point(14, 0, 0), Point(14, 10, 0), Point(7, 10, 0), 12)))
        all_walls = []
        for room in self.rooms:
            all_walls.append(room.walls)
        self.all_walls = list(itertools.chain.from_iterable(all_walls))

        self.AP.assigned_room = self.rooms[0]

    def __str__(self):
        for room in self.rooms:
            print(room)
        return ''


class Building503_v2:
    AP = Point(8, 14, 0)  # AP = Point(4, 7, 1)
    number_of_rooms = 2
    _3D_measures = [30, 22, 12]

    def __init__(self):
        self.rooms = []
        self.rooms.append(Room(12, 18, 8,
                               # walls
                               Wall(Point(0, 0, 0), Point(14, 0, 0), Point(14, 0, 10), Point(0, 0, 10), 1),
                               Wall(Point(14, 0, 0), Point(14, 20, 0), Point(14, 20, 10), Point(14, 0, 10), 2),
                               Wall(Point(14, 20, 0), Point(0, 20, 0), Point(0, 20, 10), Point(14, 20, 10), 3),
                               Wall(Point(0, 20, 0), Point(0, 0, 0), Point(0, 0, 10), Point(0, 20, 10), 4),
                               # ceil
                               Wall(Point(0, 0, 10), Point(14, 0, 10), Point(14, 20, 10), Point(0, 20, 10), 5),
                               # floor
                               Wall(Point(0, 0, 0), Point(14, 0, 0), Point(14, 20, 0), Point(0, 20, 0), 6)))
        self.rooms.append(Room(12, 18, 8,
                               # walls
                               Wall(Point(14, 0, 0), Point(28, 0, 0), Point(20, 0, 10), Point(14, 0, 10), 7),
                               Wall(Point(28, 0, 0), Point(28, 20, 0), Point(28, 20, 10), Point(28, 0, 10), 8),
                               Wall(Point(28, 20, 0), Point(14, 20, 0), Point(14, 20, 10), Point(28, 20, 10), 9),
                               Wall(Point(14, 20, 0), Point(14, 0, 0), Point(14, 0, 10), Point(14, 20, 10), 10),
                               # ceil
                               Wall(Point(14, 0, 10), Point(28, 0, 10), Point(28, 20, 10), Point(14, 20, 10), 11),
                               # floor
                               Wall(Point(14, 0, 0), Point(28, 0, 0), Point(28, 20, 0), Point(14, 20, 0), 12)))
        all_walls = []
        for room in self.rooms:
            all_walls.append(room.walls)
        self.all_walls = list(itertools.chain.from_iterable(all_walls))

        self.AP.assigned_room = self.rooms[0]

    def __str__(self):
        for room in self.rooms:
            print(room)
        return ''


class TreeNode(object):
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children_indices = []


class Tree(object):
    def __init__(self, Tx):
        self.tree = [TreeNode(Tx, None)]

    def add_children(self, children, parent_index):
        for child in children:
            self.tree.append(TreeNode(child, self.tree[parent_index]))
            self.tree[parent_index].children_indices.append(len(self.tree) - 1)

    def get_children(self, parent_index):
        children = []
        children_indices = self.tree[parent_index].children_indices
        for i in children_indices:
            children.append(self.tree[i])
        return children

    def get_children_indices(self, parent_index):
        return self.tree[parent_index].children_indices


def det(a, b, c, d):
    return a * d - b * c


def get_direction_vector(a, b):
    return [b.x - a.x, b.y - a.y, b.z - a.z]


def is_pos_in_same_room_with_AP(position, building):
    room_with_AP = building.AP.assigned_room
    s = position - room_with_AP.walls[0].p1
    uvw = room_with_AP.uvw
    return 0 < s*uvw[0] < uvw[0]*uvw[0] and 0 < s*uvw[1] < uvw[1]*uvw[1] and 0 < s*uvw[2] < uvw[2]*uvw[2]


def calculate_transmissions(p1, p2, building):
    transmission_num = 0
    for wall in building.all_walls:
        if get_intersection_point(p1, p2, wall) != -1:
            transmission_num += 1
    return transmission_num


def calculate_reflection_paths(image_tree, last_layer, reflection_number, Tx, Rx, building):
    paths = []
    ray_distance_threshold = 100
    transmissions_threshold = 100
    is_transmissions_considered = True  # not is_pos_in_same_room_with_AP(Tx, building)

    if reflection_number == 0:
        traversed_distance = calculate_traversed_distance(Tx, Rx)
        transmission_num = 0
        if is_transmissions_considered:
            transmission_num = calculate_transmissions(Rx, Tx, building)
        # if traversed distance is too long or too many wall transmissions - skip this path
        if traversed_distance > ray_distance_threshold or transmission_num > transmissions_threshold:
            return paths
        # else find this path if exists
        paths = [[Rx, Tx, traversed_distance, transmission_num]]  # path[-2] contains traversed distance, path[-1] - number of tramsmissions during path
    else:
        for i in last_layer:
            transmission_num = 0
            image_point = image_tree.tree[i].data

            traversed_distance = calculate_traversed_distance(Tx, image_point)
            # if traversed distance is too long - skip this path
            if traversed_distance > ray_distance_threshold:
                continue

            # else find this path if exists
            path = [traversed_distance]  # path[0] contains traversed distance
            is_correct = True
            path.append(Tx)
            wall = image_point.assigned_wall
            intersection_point = get_intersection_point(Tx, image_point, wall)
            if intersection_point != -1:
                if is_transmissions_considered:
                    transmission_num += calculate_transmissions(Tx, intersection_point, building)
                    # if too many wall transmissions - skip this path
                    if transmission_num > transmissions_threshold:
                        continue
                path.append(intersection_point)
            else:
                path.clear()
                continue

            image_point_ = image_tree.tree[i]
            for j in range(2, reflection_number + 1):
                image_point = image_point_.parent.data
                image_point_ = image_point_.parent
                wall = image_point.assigned_wall
                intersection_point = get_intersection_point(path[-1], image_point, wall)
                if intersection_point != -1:
                    if is_transmissions_considered:
                        transmission_num += calculate_transmissions(path[-1], intersection_point, building)
                        # if too many wall transmissions - skip this path
                        if transmission_num > transmissions_threshold:
                            continue
                    is_correct = True
                    path.append(intersection_point)
                else:
                    path.clear()
                    is_correct = False
                    break
            if is_correct:
                if is_transmissions_considered:
                    transmission_num += calculate_transmissions(path[-1], Rx, building)
                    # if too many wall transmissions - skip this path
                    if transmission_num > transmissions_threshold:
                        continue
                path.append(Rx)
                path.reverse()
                path.append(transmission_num)# path[-2] contains traversed distance, path[-1] - number of tramsmissions during path
                paths.append(path)

    return paths


def get_all_paths(image_tree, Tx, Rx, building):
    reflection_number = 0
    paths = [calculate_reflection_paths(image_tree, 0, reflection_number, Tx, Rx, building)]
    reflection_number = 1
    paths.append(calculate_reflection_paths(image_tree, image_tree.get_children_indices(0), reflection_number, Tx, Rx, building))
    reflection_number = 2

    for i in image_tree.get_children_indices(0):
        paths.append(calculate_reflection_paths(image_tree, image_tree.get_children_indices(i), reflection_number, Tx, Rx, building))
        for j in image_tree.get_children_indices(i):
            paths.append(calculate_reflection_paths(image_tree, image_tree.get_children_indices(j), reflection_number + 1, Tx, Rx, building))

    return list(itertools.chain.from_iterable(paths))


def check_if_point_belongs_to_wall(p, wall):
    if (p.x - wall.p1.x) * (p.x - wall.p3.x) <= 0 and (
                p.y - wall.p1.y) * (p.y - wall.p3.y) <= 0 and (
                p.z - wall.p1.z) * (p.z - wall.p3.z) <= 0:
        return True
    return False


def check_if_point_belongs_to_line_segment(a, b, p):
    return ((a.x-p.x)*(b.x-p.x)+(a.y-p.y)*(b.y-p.y)+(a.z-p.z)*(b.z-p.z)) <= 0


# with line segment and plane
def get_intersection_point(p1, p2, wall):
    intersect_epsilon = 0.1
    a = wall.plane_equation[0]
    b = wall.plane_equation[1]
    c = wall.plane_equation[2]
    direction_vector = get_direction_vector(p1, p2)

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

        if math.fabs(p.x) < EPS: p.x = 0
        if math.fabs(p.y) < EPS: p.y = 0
        if math.fabs(p.z) < EPS: p.z = 0
        if p1.is_equal(p) or p2.is_equal(p):
            return -1
        if check_if_point_belongs_to_line_segment(p1, p2, p):
            if check_if_point_belongs_to_wall(p, wall):
                return p
    return -1


def build_image_tree_layer(Tx, walls):
    layer_points = []

    for wall in walls:
        if wall != Tx.assigned_wall:
            pe = wall.plane_equation  # plane equation
            image_point = Point(0, 0, 0)
            common_part = (pe[0] * Tx.x + pe[1] * Tx.y + pe[2] * Tx.z +
                           pe[3]) / (pe[0] * pe[0] + pe[1] * pe[1] + pe[2] * pe[2])
            image_point.x = Tx.x - 2 * pe[0] * common_part
            image_point.y = Tx.y - 2 * pe[1] * common_part
            image_point.z = Tx.z - 2 * pe[2] * common_part
            image_point.assigned_wall = wall
            layer_points.append(image_point)
    return layer_points


def build_image_tree(Tx, walls):
    image_tree = Tree(Tx)
    tree_position = 0
    while tree_position < (1 + len(walls) + len(walls)*(len(walls)-1)):
        new_points = build_image_tree_layer(image_tree.tree[tree_position].data, walls)
        image_tree.add_children(new_points, tree_position)
        tree_position += 1
    return image_tree


#######################################################################################################################
def calculate_traversed_distance(a, b):
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)


# for one position
# from ARIADNE
def get_signal_strength(image_tree, Tx, Rx, building):
    if Tx.is_equal(Rx):
        return 0
    paths = get_all_paths(image_tree, Tx, Rx, building)

    # calculate signal strength for each path, then sum
    signal_strength = 0

    for path in paths:
        num_of_reflections = len(path) - 4  # len(path) minus start point, end point, path length stored in path[-2] and number of transmissions stored n path[-1]
        num_of_transmissions = path[-1]
        traversed_distance = path[-2]
        signal_strength += (P0 - 10 * n * math.log10(traversed_distance) -
                            reflection_coef * num_of_reflections -
                            transmission_coef * num_of_transmissions)
    signal_strength /= len(paths)

    return int(round(signal_strength))


def calculate_signal_strength_matrix(building, signal_strength_matrix):
    image_tree = build_image_tree(building.AP, building.all_walls)

    for room in building.rooms:
        for x in range(room.walls[0].p1.x + 1, room.walls[0].p2.x):
            for y in range(room.walls[3].p2.y + 1, room.walls[3].p1.y):
                for z in range(room.walls[0].p1.z + 1, room.walls[0].p4.z):
                    signal_strength_matrix[x][y][z] = get_signal_strength(image_tree, Point(x, y, z), building.AP, building)


def serialize(data):
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)


def deserialize():
    with open('data.pickle', 'rb') as f:
        return pickle.load(f)
#######################################################################################################################

building = Building503_v2()
signal_strength_matrix = np.zeros((building._3D_measures[0], building._3D_measures[1], building._3D_measures[2]))
signal_strength_matrix = signal_strength_matrix.astype(np.int32)

'''
for y in range(building._3D_measures[1]-1, -1, -1):
    for x in range(0, building._3D_measures[0]):
        signal_strength_matrix[x][y][1]=-40
'''


t = timeit.default_timer()
calculate_signal_strength_matrix(building, signal_strength_matrix)
print("time")
print(timeit.default_timer()-t)
print()

### !! serialization
serialize(signal_strength_matrix)

### !! deserialization
#signal_strength_matrix_s = deserialize()

for y in range(building._3D_measures[1]-1, -1, -1):
    for x in range(0, building._3D_measures[0]):
        print("%3d" % signal_strength_matrix[x][y][1], end=' ')
    print()

data = [
    go.Heatmap(z=signal_strength_matrix[:, :, 1].transpose())
]
plotly.offline.plot(data, filename='basic-heatmap2.html')