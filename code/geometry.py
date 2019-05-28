# -*- coding: utf-8 -*-
"""
Geometry of an artery bifurcation
Olga Mula 2019
Modified by Changqing Fu
"""

from dolfin import * # FEM solver
from mshr import * # mesh

import numpy as np

# global variables
diam_steno_vessel=0.1
diam_narrow=0.04
theta_steno=-np.pi/6
diam_healthy_vessel=0.1
theta_healthy=-np.pi/6
length0 = .5
length = .3
diam_trunk = diam_healthy_vessel * np.cos(theta_healthy) + diam_steno_vessel * np.cos(theta_steno)

class Artery():
    def __init__(self, diam_steno_vessel=0.1, diam_narrow=0.04, theta_steno=np.pi/6, diam_healthy_vessel=0.1, theta_healthy=np.pi/6,length0 = .5,length = .3):
        
        self.diam_steno_vessel = diam_steno_vessel
        self.diam_narrow = diam_narrow#diam_narrow
        self.theta_steno = theta_steno
        self.diam_healthy_vessel = diam_healthy_vessel
        self.theta_healthy = theta_healthy
        self.length0 = length0
        self.length = length

    def __vessel_healthy(self):
        """
            Points for the
            Healthy vessel in the upper part of the bifurcation
        """
        D  = self.diam_healthy_vessel   # Diameter vessel

        n = 20 # Number of points to build domain (impact on mesh size)
        length = self.length

        # Bottom
        xref = np.linspace(-length, length, num=n)
        yref = np.zeros(n)
        points_bottom = [Point(x, y) for (x,y) in zip(xref,yref)]

        # Top
        xref = np.linspace(length, -length, num=n)
        yref = D*np.ones(n)
        points_top = [Point(x, y) for (x,y) in zip(xref,yref)]

        vertices = points_bottom + points_top

        # Translate to origin
        vertices = [ Point(p[0]+length,p[1]) for p in vertices ]

        # Rotate
        theta = self.theta_healthy
        vertices = [ Point(np.cos(theta)*p[0]-np.sin(theta)*p[1],np.sin(theta)*p[0]+np.cos(theta)*p[1]) for p in vertices ]
        return vertices


    def __vessel_stenosis(self):
        """
            Points for the
            Stenotic vessel in the lower part of the bifurcation
        """

        D  = self.diam_steno_vessel   # Diameter vessel
        diam_narrow = self.diam_narrow                  # Narrowing in stenosis (diam_narrow < D/2)
        L  = 2*D                      # Length of stenosis
        x0 = 0.                       # location of the center of the stenosis
        length = self.length

        def S(x,L):
            """
                Section of the stenosis following the paper
                "Direct numerical simulation of stenotic flows,
                Part 1: Steady flow" J. Fluid Mech.
            """
            
            # return D/2 * (1-diam_narrow*(1+np.cos(2*np.pi*(x-x0)/L)))
            return D/2 -diam_narrow/2*(1+np.cos(2*np.pi*(x-x0)/L))

        n = 30 # Number of points to build domain (impact on mesh size)

        # Bottom
        xref = np.linspace(-length, length, num=n)
        yref = [ -S(x,L) if -L/2<= x and x <= L/2 else -D/2 for x in xref]
        points_bottom = [Point(x, y) for (x,y) in zip(xref,yref)]

        # Top
        xref = np.linspace(length, -length, num=n)
        yref = [ S(x,L) if -L/2<= x and x <= L/2 else D/2 for x in xref]
        points_top = [Point(x, y) for (x,y) in zip(xref,yref)]

        vertices = points_bottom + points_top

        # Translate to origin
        vertices = [ Point(p[0]+length,p[1]-D/2) for p in vertices ]

        # Rotate
        theta = -self.theta_steno
        vertices = [ Point(np.cos(theta)*p[0]-np.sin(theta)*p[1],np.sin(theta)*p[0]+np.cos(theta)*p[1]) for p in vertices ]
        return vertices

    def __domain(self):
        """
            Construction of the bifurcation geometry as a Polygon object
        """

        vertices_stenosis = self.__vessel_stenosis()
        vertices_healthy = self.__vessel_healthy()
        length0 = self.length0

        n=10

        xl = vertices_stenosis[0][0]
        yl = vertices_stenosis[0][1]
        xref = np.linspace(-length0, xl, num=n, endpoint=False)
        vertices_bottom_left = [ Point(x,yl) for x in xref ]

        xr = vertices_healthy[-1][0]
        yr = vertices_healthy[-1][1]
        xref = np.linspace(xr, -length0, num=n)
        vertices_top_left = [ Point(x,yr) for x in xref ]

        v = vertices_bottom_left + vertices_stenosis + vertices_healthy[1:] + vertices_top_left

        return Polygon(v)

    def mesh(self,mesh_precision = 40):
        """
            Create mesh of the geometry
        """
        # mesh_precision = 20
        return generate_mesh(self.__domain(), mesh_precision)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # diam_steno_vessel = 0.6
    # diam_narrow = 0.4
    # theta_steno = - np.pi /6

    # diam_healthy_vessel = 0.6
    # theta_healthy = + np.pi /6

    mesh_precision = 40
    # artery = Artery(diam_steno_vessel, diam_narrow, theta_steno, diam_healthy_vessel, theta_healthy)
    mesh = Artery().mesh(mesh_precision)
    plot(mesh, title='stenosis')
    plt.savefig('mesh.pdf')