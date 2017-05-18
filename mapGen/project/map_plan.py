from walls_representation import Point, Wall


class MapPlan:
    boundaries = []  # a rectangular is represented with 4 lines
    walls = []
    AP = Point(374.40963973214787, 243.90803739747372, 3)

    def __init__(self, data):
        boundaries = []
        walls = []
        for j in range(0, 4):
            boundaries.append([float(i) for i in data[j].split(',')])
        for j in range(4, len(data)):  #TODO
            walls.append([float(i) for i in data[j].split(',')])
        self.make_plan(boundaries, walls)

    def make_plan(self, boundaries, walls):
        for b in boundaries:
            boundary = Wall(Point(b[0], b[1], b[2]),
                            Point(b[3], b[4], b[5]),
                            Point(b[6], b[7], b[8]),
                            Point(b[9], b[10], b[11]))
            self.boundaries.append(boundary)
        for w in walls:
            wall = Wall(Point(w[0], w[1], w[2]),
                        Point(w[3], w[4], w[5]),
                        Point(w[6], w[7], w[8]),
                        Point(w[9], w[10], w[11]))
            if wall.is_a_plane():
                self.walls.append(wall)
