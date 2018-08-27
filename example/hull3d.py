#!/usr/bin/env python
"""
Convex hull algorithms based on polyhedron and cddlib

hull - create a hull given 2d/3d points
inside - return mask of points that are inside a given hull (2d/3d)
"""
__all__ = ['hull', 'inside']

__author__ = 'Rolv, Pearu'

from numpy import *
from polyhedron import Vrep, Hrep

def _mkhull(points):
    p = Vrep(points)
    return Hrep(p.A, p.b)

def hull(points):
    return _mkhull(points).generators

def inside(p, points):
    if not isinstance(p, Hrep):
        p = _mkhull(p)
    if points.shape[-1] == 1:
        raise ValueError("Cannot do 1d points")
    inside_ = lambda point: alltrue(dot(p.A, point) <= p.b)
    mask = apply_along_axis(inside_, 1, points)
    return mask
    
def _test_2d(n=100000):
    print('percent inside (should be around 0.1):', size(nonzero(inside(
        ((0,0), (0,1), (0.1, 1), (0.1, 0)),
        random.random((n, 2)))
                       ))/float(n))
    
# Domains
_d_1d = asarray([0, 0.1])
#_d_1d = asarray([(0,0,0), (0,1,0), (1,1,0), (1,1,1)])
_d_2d = asarray([(0,0), (0,1), (0.1,1), (0.1,0)])
_d_3d = 0.1**(1/3.)*asarray([(0,0,0), (0,0,1), (0,1,0), (0,1,1), \
                             (1,0,0), (1,0,1), (1,1,0), (1,1,1)])

def _test():
    points = random.random((20,3))
    p = _mkhull(points)
    print('Hull vertices:\n',p.generators)

    points2 = 1.1*random.random((10,3))
    for i in range(len(points2)):
        # if all True then i-th point is in hull
        point = points2[i]
        if alltrue(dot(p.A,point) <= p.b):
            print('point',point,'is IN')
        else:
            print('point',point,'is OUT')
            
    points3 = random.random((3000,3))
    mask = inside(p, points3)
    p3 = points3[nonzero(mask)]
        
    n = 100000
    for d in _d_2d, _d_3d:
        h = hull(d)
        dim = h.shape[-1]
        points = random.random((n, dim))
        print(0.1, dim, size(nonzero(inside(h, points))) / float(n))

def _test_plot():
    try:
        import pylab
    except ImportError:
        pylab=False
    if pylab:

        pylab.ion()
        #import matplotlib.axes3d as p3
        from mpl_toolkits.mplot3d import Axes3D 

        # Create points
        u = r_[0:2*pi:100j]
        v = r_[0:pi:100j]
        x = 10*outer(cos(u), sin(v))
        y = 10*outer(sin(u), sin(v))
        z = 10*outer(ones(size(u)), cos(v))
        
        points = transpose(vstack((x.flat, y.flat, z.flat)))

        hull_ = asarray([(0,0,0), (0,0,1), (-10,0,0), (-10,0,1),
                         (-10,-10,0), (-10,-10,1), (0,-10,0), (0,-10,1)])
        
        # Plot full surface
        ax = Axes3D(pylab.figure())
        ax.plot_surface(x,y,z)
        # Plot slice/hull
        x_,y_,z_ = list(map(squeeze, hsplit(hull_, 3)))
        ax.scatter(x_,y_,z_, color='g')
        s = nonzero(z_==0)
        x,y,z = x_[s].tolist(), y_[s].tolist(), z_[s].tolist()
        x.append(x[0]); y.append(y[0]); z.append(z[0])
        ax.plot(x,y,z, 'g-')
        ax.plot(x,y, ones(len(z)), 'g-')
        # Plot overlap
        x,y,z = list(map(squeeze, hsplit(points[nonzero(inside(hull_, points))], 3)))
        ax.scatter(x,y,z, color='r')

        pylab.draw()
        pylab.savefig('hull3d.png')

if __name__ == '__main__':
    #_test_2d(100)
    #_test()
    _test_plot()
    print(_mkhull(_d_1d))
