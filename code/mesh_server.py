from mshr import * # mesh
from fenics import *
mesh_precision = 32

sqr2 = 2**.5    #constant for simplicity
Y0 = 2          #Y trunk length
Y1 = 2 * sqr2   #Y branch length
y1 = 2          
Y2 = 2 * sqr2   #Y branch length
y2 = 2
D = .1 * sqr2   #branch width
d = .1
D0 = 2*d        #trunk width
A = .5          #shrink length = 2A
a = .5 / sqr2
B = .1 * sqr2   #shrink width = B
b = .1

def compute_mesh(Y0=Y0,Y1=Y1,Y2=Y2,d=d,a=a,b=b):
    "Return Y-mesh"
    domain_vertices = [Point(-Y0        , -D0        ),
                       Point(0.0        , -D0        ),
                       Point(y2-d       , -y2-d      ),
                       Point(y2+d       , -y2+d      ),
                       Point(y2/2+d+a   , -y2/2+d-a  ),
                       Point(y2/2+d+a-b , -y2/2+d-a-b),
                       Point(y2/2+d-a-b , -y2/2+d+a-b),
                       Point(y2/2+d-a   , -y2/2+d+a  ),
                       Point(d*2          , 0        ),
                       Point(y1+d       , y1-d       ),
                       Point(y1-d       , y1+d       ),
                       Point(0.0        , D0         ),
                       Point(-Y0        , D0         )]
    polygon = Polygon(domain_vertices)
    domain = polygon
    return generate_mesh(domain, mesh_precision)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mesh = compute_mesh(Y0,Y1,Y2,d,a,b)
    plot(mesh)
    print('plotting mesh')
    # plt.show()
    plt.savefig('mesh')
