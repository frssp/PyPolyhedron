#!/usr/bin/env python
"""
Polyhedron --- Python C/API interface for C-library cddlib.

About cddlib, see http://www.ifor.math.ethz.ch/~fukuda/cdd_home/cdd.html
About polyhedrons, see http://www.ifor.math.ethz.ch/ifor/staff/fukuda/polyfaq/polyfaq.html

Copyright 2000,2007 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Pearu Peterson
"""

from . _cdd import vrep, hrep
import pprint

class Polyhedron:
    def __init__(self):
        pass
    def __str__(self):
        p = pprint.pformat
        ret = 'Inequalities:\n'
        for i in range(len(self.A)):
            if i in self.eq:
                ret = '%s\t%s * x == %s\n'%(ret,p(self.A[i].tolist()),self.b[i])
            else:
                ret = '%s\t%s * x <= %s\n'%(ret,p(self.A[i].tolist()),self.b[i])
        ret = '%sGenerators:\n'%(ret)
        for i in range(len(self.generators)):
            ret = '%s\t%s[%s]=%s\n'%(ret,{0:'ray',1:'vertex'}[self.is_vertex[i]],
                                   i,self.generators[i].tolist())
        ret = '%sExtreme points:\n'%(ret)
        for i in range(len(self.inc)): # i-th generator
            ret = '%s\t%s[%s] belongs to half-spaces %s\n'%(ret,{0:'ray',
                                                             1:'vertex'}[self.is_vertex[i]],
                                                        i,self.inc[i])
        nofn = len(self.A)
        if 0:
            ret = '%sCheck extreme points and orthogonality:\n'%(ret)
            for i in range(len(self.inc)): # i-th generator
                if self.is_vertex[i]:
                    for j in self.inc[i]: # j-th normal
                        ret = '%s\tA[%s] * vertex[%s] == b[%s]'%(ret,j,i,j)
                        ret = '%s (%s == %s)\n'%(ret,scalmul(self.A[j],self.generators[i]),self.b[j])
                else:
                    for j in self.inc[i]: # j-th normal
                        if j<nofn: # defined normal
                            ret = '%s\tA[%s] * ray[%s] == 0.0'%(ret,j,i)
                            ret = '%s (%s == 0.0)\n'%(ret,scalmul(self.A[j],self.generators[i]))
                        else:
                            ret = '%s\tInf[%s] * ray[%s]\n'%(ret,j,i)
        ret = '%sNeighbors:\n'%(ret)
        for i in range(len(self.adj)): # i-th generator
            vs,rs = [],[]
            for j in self.adj[i]: # j-th generator
                if self.is_vertex[j]: vs.append(j)
                else: rs.append(j)
            if self.is_vertex[i]:
                ret = '%s\tvertex[%s] neighbors are vertices %s and rays %s\n'%(ret,i,vs,rs)
            else:
                ret = '%s\tray[%s] neighbors are vertices %s and rays %s\n'%(ret,i,vs,rs)
        ret = '%sHalf-spaces:\n'%(ret)
        for i in range(len(self.ininc)): # i-th normal
            vs,rs = [],[]
            for j in self.ininc[i]: # j-th generator
                if self.is_vertex[j]: vs.append(j)
                else: rs.append(j)
            if i<nofn:
                ret = '%s\tA[%s] contains vertices %s and rays %s\n'%(ret,i,vs,rs)
            else:
                ret = '%s\tInf[%s] contains vertices %s and rays %s\n'%(ret,i,vs,rs)
        ret = '%sHalf-space neighbors:\n'%(ret)
        for i in range(len(self.inadj)): # i-th normal
            if i<nofn:
                ret = '%s\tA[%s] is neighbored by A%s\n'%(ret,i,self.inadj[i])
            else:
                ret = '%s\tInf[%s] is neighbored by A%s\n'%(ret,i,self.inadj[i])
        return ret

class Hrep(Polyhedron):
    def __init__(self,A,b=None,eq=None):
        self.A,self.b,self.eq,self.generators,self.is_vertex,self.fr,self.inc,self.adj,self.ininc,self.inadj = hrep(A,b,eq)
class Vrep(Polyhedron):
    def __init__(self,v,r=None,fr=None):
        self.A,self.b,self.eq,self.generators,self.is_vertex,self.fr,self.inc,self.adj,self.ininc,self.inadj = vrep(v,r,fr)

    
if __name__ == "__main__":
    p = Hrep([[0,1],[1,0],[-1,-1],[0,-1]],[1,2,0,0])
    print(p)
    p = Vrep([[0,1],[1,0],[-1,-1],[0,-1]])
    print(p)
    p = Hrep([[0,1],[1,0],[0,-1]],[1,2,0.1])
    print(p)
    p = Hrep([[1,0],[0,-1]],[2,0.1],[0])
    print(p)
