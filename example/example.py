
from numpy import *
from polyhedron import Vrep, Hrep

points = random.random((20,3))

def mkhull(points):
    p = Vrep(points)
    return Hrep(p.A, p.b)
p = mkhull(points)

print('Hull vertices:\n',p.generators)

points2 = 1.1*random.random((3,3))
for i in range(len(points2)):
    # if all True then i-th point is in hull
    point = points2[i]
    if alltrue(dot(p.A,point) <= p.b):
        print('point',point,'is IN')
    else:
        print('point',point,'is OUT')
