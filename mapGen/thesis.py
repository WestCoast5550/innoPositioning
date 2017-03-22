# http://stackoverflow.com/questions/15797920/how-to-convert-wifi-signal-strength-from-quality-percent-to-rssi-dbm
# dBm to percentage

import math
import itertools
import pylab as p
import numpy as np

# import mahotas

EPS = 1E-9
P0 = -40  # dBm
reflection_coef = -12.03  # dBm
transmission_coef = -0.5095  # dBm
n = 5.0  # attenuation exponent (dBm)


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
        self.uvw = [get_direction_vector(wall1.p1, wall4.p1),
                    get_direction_vector(wall1.p1, wall1.p2),
                    get_direction_vector(wall1.p1, wall1.p4)]

    def __str__(self):
        for wall in self.walls:
            print(wall)
        return ''


class Building:
    AP = Point(2, 5, 1)
    number_of_rooms = 2
    _3D_measures = [15, 17, 4]

    def __init__(self):
        self.rooms = []
        self.rooms.append(Room(15, 10, 4,
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

        self.AP.assigned_room = self.rooms[0]

    def __str__(self):
        for room in self.rooms:
            print(room)
        return ''


class Tree_Node(object):
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children_indices = []


class Tree(object):
    def __init__(self, Tx):
        self.tree = [Tree_Node(Tx, None)]

    def add_children(self, children, parent_index):
        for child in children:
            self.tree.append(Tree_Node(child, self.tree[parent_index]))
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


def calculate_reflection_paths2(image_tree, reflection_number, Tx, Rx):
    paths = []
    is_correct = True
    if reflection_number == 0:
        paths.append([Tx, Rx])
    else:
        for image_point in image_tree[-1]:
            path = [Rx]
            wall = image_point.assigned_wall
            p = intersect_line(Rx, image_point, wall)
            if p != -1:
                path.append(p)
            else:
                path.clear()
                is_correct = False
                continue
            for i in range(2, reflection_number + 1):
                wall = image_tree[-i][0].assigned_wall
                p = intersect_line(path[-1], image_tree[-i][0], wall)
                if p != -1:
                    is_correct = True
                    path.append(p)
                else:
                    path.clear()
                    is_correct = False
                    break
            if is_correct:
                path.append(Tx)
                path.reverse()
                paths.append(path)

    return paths


def check_if_point_belongs_to_wall(p, wall):
    if (p.x - wall.p1.x) * (p.x - wall.p3.x) <= 0 and (
                p.y - wall.p1.y) * (p.y - wall.p3.y) <= 0 and (
                p.z - wall.p1.z) * (p.z - wall.p3.z) <= 0:
        return True
    return False


def intersect_line(p1, p2, wall):
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
        if check_if_point_belongs_to_wall(p, wall):
            return p
    return -1


def get_all_paths2(walls, Tx, Rx):
    image_tree = [Tx]
    paths = [calculate_reflection_paths(image_tree, 0, Tx, Rx)]
    image_tree.append(build_image_tree_layer(Tx, walls))  # first layer
    paths.append(calculate_reflection_paths(image_tree, 1, Tx, Rx))
    points_on_a_layer = len(image_tree[-1])
    for i in range(0, points_on_a_layer):
        image_tree.append(build_image_tree_layer(image_tree[-1][0], walls))  # second layer
        paths.append(calculate_reflection_paths(image_tree, 2, Tx, Rx))
        points_on_a_layer_2 = len(image_tree[-1])
        for j in range(0, points_on_a_layer_2):
            image_tree.append(build_image_tree_layer(image_tree[-1][0], walls))  # third layer
            paths.append(calculate_reflection_paths(image_tree, 3, Tx, Rx))
            # delete used 3rd layer and first point on 2nd (which is parent of 3rd)
            image_tree.pop(-1)
            image_tree[-1].pop(0)
            if len(image_tree[-1]) == 0:
                image_tree.pop(-1)

        image_tree[-1].pop(0)

    return list(itertools.chain.from_iterable(paths))


def calculate_reflection_paths(image_tree, max_reflection, Tx, Rx):
    paths = []
    is_correct = True
    paths.append([Rx, Tx])
    tree = image_tree
    first_layer = tree.children
    for image_point in first_layer:
        is_correct = True
        path = [Tx]
        wall = image_point.data.assigned_wall
        p = intersect_line(Tx, image_point.data, wall)
        if p != -1:
            path.append(p)
        else:
            path.clear()
            is_correct = False
            continue
        if is_correct:
            path.append(Rx)
            path.reverse()
            paths.append(path)

    for image_point in first_layer:
        second_layer = image_point.children
        for image_point1 in second_layer:
            is_correct = True
            path = [Tx]
            wall = image_point1.data.assigned_wall
            p = intersect_line(Tx, image_point1.data, wall)
            if p != -1:
                path.append(p)
            else:
                path.clear()
                is_correct = False
                continue
            wall = image_point.data.assigned_wall
            p = intersect_line(path[-1], image_point.data, wall)
            if p != -1:
                path.append(p)
            else:
                path.clear()
                is_correct = False
                continue
            if is_correct:
                path.append(Rx)
                path.reverse()
                paths.append(path)

    for image_point in first_layer:
        second_layer = image_point.children
        for image_point1 in second_layer:
            third_layer = image_point1.children
            for image_point2 in third_layer:
                is_correct = True
                path = [Tx]
                wall = image_point2.data.assigned_wall
                p = intersect_line(Tx, image_point2.data, wall)
                if p != -1:
                    path.append(p)
                else:
                    path.clear()
                    is_correct = False
                    continue
                wall = image_point1.data.assigned_wall
                p = intersect_line(path[-1], image_point1.data, wall)
                if p != -1:
                    path.append(p)
                else:
                    path.clear()
                    is_correct = False
                    continue
                wall = image_point.data.assigned_wall
                p = intersect_line(path[-1], image_point.data, wall)
                if p != -1:
                    path.append(p)
                else:
                    path.clear()
                    is_correct = False
                    continue
                if is_correct:
                    path.append(Rx)
                    path.reverse()
                    paths.append(path)

    return paths

'''
    for reflection_number in range(1, max_reflection):
        for image_point in first_layer:
            path = [Rx]
            wall = image_point.data.assigned_wall
            p = intersect_line(Rx, image_point.data, wall)
            if p != -1:
                path.append(p)
            else:
                path.clear()
                is_correct = False
                continue
            for i in range(2, reflection_number + 1):
                i_p = image_point.parent
                wall = image_point.parent.data.assigned_wall
                p = intersect_line(path[-1], i_p.data, wall)
                if p != -1:
                    is_correct = True
                    path.append(p)
                else:
                    path.clear()
                    is_correct = False
                    break
            if is_correct:
                path.append(Tx)
                path.reverse()
                paths.append(path)
            tree = tree.children

    return list(itertools.chain.from_iterable(paths))
    '''

def build_image_tree(walls, Tx):
    image_tree = Tree(Tx)
    first_layer = build_image_tree_layer(Tx, walls)  # first layer
    image_tree.add_children(first_layer, 0)

    for i in image_tree.get_children_indices(0):
        second_layer = build_image_tree_layer(image_tree.tree[i].data, walls)
        image_tree.add_children(second_layer, i)
        for j in image_tree.get_children_indices(i):
            third_layer = build_image_tree_layer(image_tree.tree[j].data, walls)
            image_tree.add_children(third_layer, j)

    return image_tree


#######################################################################################################################
def get_reflection_length(a, b):
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)


def calculate_traversed_distance(path):
    traversed_distance = 0
    for i in range(0, len(path) - 1):
        traversed_distance += get_reflection_length(path[i], path[i + 1])
    return traversed_distance


# for one position
# from ARIADNE
def get_signal_strength(room, image_tree, Tx, Rx):
    paths = calculate_reflection_paths(image_tree, 3, Tx, Rx)

    if Tx.x == 11 and Tx.y == 3 and Tx.z == 1:
        for path in paths:
            print()
            for p in path:
                print(p)

        print(len(paths))
    # calculate signal strength for each path, then sum
    signal_strength = 0
    '''
    for path in paths:
        num_of_reflections = len(path) - 2
        num_of_transmissions = 0  # TODO
        traversed_distance = calculate_traversed_distance(path)
        if traversed_distance == 0:
            return 0
        signal_strength += (P0 - 10 * math.log10(
            traversed_distance) - reflection_coef * num_of_reflections - transmission_coef * num_of_transmissions)
    signal_strength /= len(paths)
    '''
    return round(signal_strength, 3)


def calculate_signal_strength_matrix(building, room, signal_strength_matrix):
    image_tree = build_image_tree(room.walls, building.AP)
    for x in range(0, room.length):
        for y in range(0, room.width):
            for z in range(room.height):
                signal_strength_matrix[x][y][z] = get_signal_strength(room, image_tree, Point(x, y, z), building.AP)


#######################################################################################################################

building = Building()
signal_strength_matrix = np.zeros((building._3D_measures[0], building._3D_measures[1], building._3D_measures[2]))
calculate_signal_strength_matrix(building, building.rooms[0], signal_strength_matrix)

for x in range(0, building.rooms[0].length):
    for y in range(0, building.rooms[0].width):
        print(signal_strength_matrix[x][y][3], end=" ")
    print()
print()

'''
Tx = Point(2, 5, 1)
Rx = Point(11, 3, 1)

image_tree = [Tx]
paths = [calculate_reflection_paths_line(image_tree, 0, Tx, Rx)]
image_tree.append(build_image_tree(Tx, room.walls))  # first layer

paths.append(calculate_reflection_paths_line(image_tree, 1, Tx, Rx))

paths2 = fun2(room.walls, Tx, Rx)
for p in paths2:
    for pp in p:
        if 15 < pp.x < 0 or 10 < pp.y < 0 or 4 < pp.z < 0:
            print(pp)

one = 0
two = 0
three = 0
for p in paths2:
    if len(p) == 3:
        one += 1

    if len(p) == 4:
        two += 1

    if len(p) == 5:
        three += 1
    for pp in p:
        print(pp)
    print()

print(len(paths2))
print(one, two, three)
'''
