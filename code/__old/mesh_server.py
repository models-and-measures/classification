from mshr import * # mesh
from fenics import *

mesh_precision = 32
sqr2 = 2**.5    #constant for simplicity
Y0 = 2          #Y trunk length
Y1 = 2 * sqr2   #Y branch length
y1 = 2          
Y2 = 2 * sqr2   #Y branch length
y2 = 2
D = .1 * sqr2   #branch radius
d = .1
D0 = 2*d        #trunk radius
A = .5          #shrink length = 2A
a = .5 / sqr2
B = .1 * sqr2   #shrink width = B
b = .1

def compute_mesh(Y0=Y0,Y1=Y1,Y2=Y2,d=d,a=a,b=b,mesh_precision=mesh_precision):
    "Return Y-mesh"
    domain_vertices = [Point(-Y0        , -D0        ),
                       Point(0.0        , -D0        ),
                       Point(y2-d       , -y2-d      ),
                       Point(y2+d       , -y2+d      ),
                       Point(y2/2+d+a   , -y2/2+d-a  ),
                       Point(y2/2+d+a-b , -y2/2+d-a-b),
                       Point(y2/2+d-a-b , -y2/2+d+a-b),
                       Point(y2/2+d-a   , -y2/2+d+a  ),
                       Point(d*2        , 0          ),
                       Point(y1+d       , y1-d       ),
                       Point(y1-d       , y1+d       ),
                       Point(0.0        , D0         ),
                       Point(-Y0        , D0         )]
    polygon = Polygon(domain_vertices)
    domain = polygon
    mesh = generate_mesh(domain, mesh_precision)
    ### Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    return mesh, V, Q

if __name__ == "__main__":
    import os
    import matplotlib as mpl
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    mesh, V, Q = compute_mesh(Y0,Y1,Y2,d,a,b)
    plot(mesh)

    if os.environ.get('DISPLAY','') == '':
        plt.savefig('mesh')
    else:
        plt.show()

    print('mesh plotted.')
