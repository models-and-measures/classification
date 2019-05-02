from fenics import *
import numpy as np
from scipy.interpolate import interp1d
from mesh_server import * # requires mesh parameters

# global variables, beware of namespace collision
T = 1                   # final time
num_steps = 2000        # number of time steps # must satisfy CFL condition
mu = 0.03               # dynamic viscosity, poise
rho = 1                 # density, g/cm3
# windkessel
c = 1                   #1.6e-5 distant capacitance
Rd = 1e5                #6001.2 distant resistance
Rp = 5e4                #7501.5 proximal resistance
p_windkessel_1 = 1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = 1.06e5 # init val
u0 = 1.                 # init amplitude
s = 0.                  # init asymmetry

### Define boundaries
inflow_str   = 'near(x[0], -2.0)'

### Define inflow with heartbeat
xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])+.2
heartfun0 = interp1d(xp, yp,kind='cubic')
heartfun = lambda x: heartfun0(x % 1.0)

class INFLOW(UserExpression):
    "Inflow function"
    def __init__(self, u0, s, **kwargs):
        super().__init__(**kwargs) #super makes sure all superclass are initiated...
        self.u0 = u0
        self.s = s
    def set_values(self,heartval):
        self.fval = heartval # heart signal (period = 1s)
    def eval(self, values, x):
        "Set value[0] to value at point x"
        tol = 1E-13
        if x[1]-0.2 < - tol and x[1] + 0.2 > tol:
            # print(self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-0.1)**2))
            values[0] = self.u0*self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-self.s)**2)
        else:
            values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)

inflow_expr = INFLOW(u0,s,degree=2)
# bcu_walls = DirichletBC(V, Constant((0, 0)), walls)

def outflow1(x, on_boundary):
        return  near(x[0]+x[1], 2*Y1) and on_boundary
def outflow2(x, on_boundary):
    return  near(x[0]-x[1], 2*Y2) and on_boundary

def walls(x, on_boundary):
    return  ( \
            (near(abs(x[1]), D0) and x[0] <= 0 + DOLFIN_EPS) or \
            (near(x[0] - abs(x[1]), 2*d) and x[0] >= 0.2 - DOLFIN_EPS) or \
            (near(x[0] - abs(x[1]), - 2*d) and x[0] >= 0.0 - DOLFIN_EPS) or \
            (near(abs(x[1] - x[0] + y2), A) and x[0] + x[1] >= 2*d - B - DOLFIN_EPS)  or \
            (near(x[0] + x[1], 2*d - B) and abs(x[0] - x[1] - y2) <= A + DOLFIN_EPS) \
            ) \
            and on_boundary

def compute_bc(V,Q,t,
            p_bdry_1 =p_windkessel_1,
            p_bdry_2 =p_windkessel_2,
            u0=u0,
            s=s,
            inflow_expr=inflow_expr,
            inflow_str=inflow_str,
            heartfun=heartfun,
            # bcu_walls=bcu_walls
            ):
    "In: V, Q. Out: boundary condition"

    heartval=heartfun(t)
    inflow_expr.set_values(heartval)
    bcu_inflow = DirichletBC(V, inflow_expr, inflow_str)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu = [bcu_inflow, bcu_walls]

    bcp_outflow1 = DirichletBC(Q, Constant((p_bdry_1)), outflow1)
    bcp_outflow2 = DirichletBC(Q, Constant((p_bdry_2)), outflow2)
    bcp = [bcp_outflow1,bcp_outflow2]
    return bcu, bcp


if __name__ == '__main__':
    mesh, V, Q = compute_mesh()

    t = 0.

    bcu, bcp = compute_bc(V,Q,t)

    print('BC computation test passed.')

            
            
            
            
            
            
            
            
