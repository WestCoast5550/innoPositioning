# Define classes for Point, Wall, Room and Building instances
class Point:
    x = 0
    y = 0
    z = 0

    assigned_wall = None  # to which wall image point is assigned

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def is_equal(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ') '


class Wall:
    p1 = Point
    p2 = Point
    p3 = Point
    p4 = Point

    def __init__(self, p1, p2, p3, p4):
        self.p1 = Point(p1.x, p1.y, p1.z)
        self.p2 = Point(p2.x, p2.y, p2.z)
        self.p3 = Point(p3.x, p3.y, p3.z)
        self.p4 = Point(p4.x, p4.y, p4.z)
        self.plane_equation = self.get_plane_equation()

    def __str__(self):
        print(self.p1)
        print(self.p2)
        print(self.p3)
        print(self.p4)
        return ''

    def get_plane_equation(self):
        list = []
        list.append(
            self.det(self.p2.y - self.p1.y, self.p3.y - self.p1.y, self.p2.z - self.p1.z, self.p3.z - self.p1.z))
        list.append(
            -self.det(self.p2.x - self.p1.x, self.p3.x - self.p1.x, self.p2.z - self.p1.z, self.p3.z - self.p1.z))
        list.append(
            self.det(self.p2.x - self.p1.x, self.p3.x - self.p1.x, self.p2.y - self.p1.y, self.p3.y - self.p1.y))
        list.append(-self.p1.x * list[0] - self.p1.y * list[1] - self.p1.z * list[2])
        return list

    def is_a_plane(self):
        if self.plane_equation[0] == 0 and \
           self.plane_equation[1] == 0 and \
           self.plane_equation[2] == 0 and \
           self.plane_equation[3] == 0:
            return False
        return True

    def det(self, a, b, c, d):
        return a * d - b * c