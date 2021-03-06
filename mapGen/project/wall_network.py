import numpy as np
from walls_representation import Point, Wall
class wall_network:


    def point__is__in__sector(self,point):
        if (point.x >=0 and point.x <= self.x__max and point.y>= 0 and point.y <= self.y__max):
            if ((point.y // self.const)* (self.horisontal// self.const) != 0 ):
                return ( int((point.y // self.const)* (self.horisontal// self.const) + point.x // self.const+1))
        else:
                return (-1)

    def __init__(self, p1, p2, p3, p4,boundaries):
        self.boundaries = boundaries
        self.boundaries.append(Point(p1.x, p1.y, p1.z))
        self.boundaries.append(Point(p2.x, p2.y, p2.z))
        self.boundaries.append(Point(p3.x, p3.y, p3.z))
        self.boundaries.append(Point(p4.x, p4.y, p4.z))
        self.const = 5
        self.x__max = 0
        self.y__max = 0
        self.y__min = self.boundaries[0].y
        self.x__min = self.boundaries[0].x
        self.make_rectangle()
        self.horisontal = self.x__max - self.x__min
        self.vertical = self.y__max - self.y__min
        self.sector__list = []

    def make_rectangle(self):
        for i in self.boundaries:
            if i.x > self.x__max:
                self.x__max = i.x
            if i.x < self.x__min:
                self.x__min = i.x
            if i.y > self.y__max:
                self.y__max = i.y
            if i.y < self.y__min:
                self.y__min = i.y

    def make__sectors (self):
        k = None
        some = 0
        initial__point__x = self.x__min
        initial__point__y = self.y__min
        for i in range((self.vertical//self.const+1)):
            if (i == ((self.vertical // self.const) )):
                for k in range((self.horisontal // self.const) + 1):
                    if (k == ((self.horisontal// self.const) )):
                        p1 = Point(initial__point__x, initial__point__y, 300)
                        p2 = Point(initial__point__x, self.y__max, 300)
                        p3 = Point(self.x__max, initial__point__y, 300)
                        p4 = Point(self.x__max, self.y__max, 300)
                        self.sector__list.append(Sector(p1, p2, p3, p4, some))
                        some += 1
                    else:
                        p1 = Point(initial__point__x, initial__point__y, 300)
                        p2 = Point(initial__point__x, self.y__max, 300)
                        initial__point__x = initial__point__x + self.const
                        p3 = Point(initial__point__x, initial__point__y, 300)
                        p4 = Point(initial__point__x, self.y__max, 300)
                        self.sector__list.append(Sector(p1, p2, p3, p4, some))
                        some += 1
            else:
                for k in range((self.horisontal//self.const)+1):
                    if (k == ((self.horisontal//self.const))):
                        p1 = Point(initial__point__x, initial__point__y, 300)
                        p2 = Point(initial__point__x, initial__point__y + self.const, 300)
                        p3 = Point(self.x__max, initial__point__y, 300)
                        p4 = Point(self.x__max, initial__point__y + self.const, 300)
                        self.sector__list.append(Sector(p1, p2, p3, p4, some))
                        some += 1
                    else:
                        self.sector__list .insert(some,[Point(initial__point__x,initial__point__y,300)])
                        p1 = Point(initial__point__x,initial__point__y,300)
                        p2 = Point(initial__point__x, initial__point__y+self.const, 300)
                        initial__point__x = initial__point__x +self.const
                        p3 = Point(initial__point__x, initial__point__y, 300)
                        p4 = Point(initial__point__x, initial__point__y + self.const, 300)
                        self.sector__list.append(Sector(p1, p2, p3, p4, some))
                        some+=1

            initial__point__x = self.x__min
            initial__point__y += self.const

    def add__wall__to__sector(self,wall):
        if ((self.point__is__in__sector(wall.p1)!= -1) and (self.point__is__in__sector(wall.p1)!= -1)):
            if (self.point__is__in__sector(wall.p1) == self.point__is__in__sector(wall.p2)):
                self.sector__list[self.point__is__in__sector(wall.p1)].walls.append(wall)
            else:
                self.sector__list[self.point__is__in__sector(wall.p1)].walls.append(wall)
                self.sector__list[self.point__is__in__sector(wall.p2)].walls.append(wall)
        else:
            print("Make your code carefully bro!!!")


    def print__sectors(self,p1,p2):
        sectors = []
        s1 = self.point__is__in__sector(Point(min(p1.x,p2.x),min(p1.y,p2.y),300))
        s2 = self.point__is__in__sector(Point(max(p1.x, p2.x), min(p1.y, p2.y), 300))
        s3 = self.point__is__in__sector(Point(min(p1.x, p2.x), max(p1.y, p2.y), 300))
        s4 = self.point__is__in__sector(Point(max(p1.x, p2.x), max(p1.y, p2.y), 300))
        sector = s1
        k = (s3-s1) / (self.vertical-1)
        for i in range (s3-s1):
            for k in range (s2-s1):
                sectors.append(sector)
                sector+=1




class Sector:
    def __init__(self,p1,p2,p3,p4,id):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.id = id
        self.walls = []


    def sector__intersected__with__line(self,line):
        edge1 = Line(self.p1,self.p2)
        edge2 = Line(self.p2, self.p3)
        edge3 = Line(self.p3, self.p4)
        edge4 = Line(self.p4, self.p1)
        if ((self.line__intersected__with__line(line,edge1)) == 1):
            return 1
        elif ((self.line__intersected__with__line(line, edge2)) == 1):
            return 1
        elif ((self.line__intersected__with__line(line, edge3)) == 1):
            return 1
        elif ((self.line__intersected__with__line(line, edge4)) == 1):
            return 1
        else :
            return 0

    def line__intersected__with__line(self,line1,line2):
        a = line1.p1.y - line1.p2.y
        b = line1.p2.x - line1.p1.x
        c = (line1.p2.y - line1.p1.y) * line1.p1.x - (line1.p2.x - line1.p1.x) * line1.p1.y

        d = line2.p1.y - line2.p2.y
        e = line2.p2.x - line2.p1.x
        f = (line2.p2.y - line2.p1.y) * line2.p1.x - (line2.p2.x - line2.p1.x) * line2.p1.y

        den = (-d * b + a * e)

        if (den == 0):
            return 0

        y = (c * d - f * a) / den
        x = (b * f - c * e) / den

        ip = [x,y]
        print("x=",x,y)
        p1 = np.array([line1.p1.x, line1.p1.y])
        p2 = np.array([line1.p2.x, line1.p2.y])
        p3 = np.array([line2.p1.x, line2.p1.y])
        p4 = np.array([line2.p2.x, line1.p2.y])

        if (np.dot(p1- ip, p2 - ip) <= 0) and (np.dot(p3 - ip, p4 - ip) <= 0):
            return 1
        else:
            return 0





class Line:
    def __init__(self,p1,p2):
        self.p1 = p1
        self.p2 = p2
p1 = Point(2,2,300)
p2 = Point(4,2,300)
p3 = Point(2,4,300)
p4 = Point(4,4,300)

sector =  Sector(p1,p2,p3,p4,1)
line = Line(Point(1,1,300),Point(1,5,300))
print(sector.sector__intersected__with__line(line))


print ("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))


# boundaries = []
# wall = wall_network(p1,p2,p3,p4,boundaries)
# wall.make__sectors()
# point = Point(0,5.5,300)
# wall.point__is__in__sector(point)
# print(wall.sector__list)
