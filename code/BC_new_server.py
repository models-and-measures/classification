from fenics import *
import numpy as np
from scipy.interpolate import interp1d
from geometry import * # requires mesh parameters

# global variables, beware of namespace collision
# T = 1                   # final time
# num_steps = 2000        # number of time steps # must satisfy CFL condition
# mu = 0.03               # dynamic viscosity, poise
# rho = 1                 # density, g/cm3
# windkessel
c = 1                   #1.6e-5 distant capacitance
Rd = 1e5                #6001.2 distant resistance
Rp = 5e4                #7501.5 proximal resistance
p_windkessel_1 = 1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = 1.06e5 # init val
u0 = 1.                 # init amplitude
s = 0.                  # init asymmetry

### Define boundaries
def inflow_domain(x, on_boundary):
    return  near(x[0]+length0, 0) and on_boundary

### Define inflow with heartbeat
xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])#+.2
heartfun0 = interp1d(xp, yp,kind='cubic')
heartfun = lambda x: heartfun0(x % 1.0)

class INFLOW(UserExpression):
    "Inflow function"
    def __init__(self, u0, s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy, **kwargs):
        super().__init__(**kwargs) #super makes sure all superclass are initiated...
        self.u0 = u0
        self.s = s
        self.y_plus = diam_healthy_vessel * np.cos(theta_healthy)
        self.y_minus = diam_steno_vessel * np.cos(theta_steno)
    def set_values(self,heartval):
        "Set the only variable parameter, namely the current time"
        self.fval = heartval # heart signal (period = 1s)
    def eval(self, values, x):
        "Set value[0] to value at point x"
        fval = self.fval
        u0 = self.u0
        s = self.s
        y_plus = self.y_plus
        y_minus = self.y_minus
        tol = 1E-13
        if x[1] - y_plus < - tol and x[1] + y_minus > tol:
            # print(self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-0.1)**2))
            values[0] = u0*fval/(y_minus + x[1])/(y_plus - x[1]) * pow(y_minus + y_plus, 2) *np.exp(-0.5*(np.log((y_minus + x[1])/(y_plus - x[1]))-self.s)**2)
        else:
            values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)

# bcu_walls = DirichletBC(V, Constant((0, 0)), walls)

def rotate(theta,x):
    n0 = np.cos(theta)
    n1 = np.sin(theta)
    rotation = np.array([[n0,n1],[-n1,n0]])
    return np.matmul(rotation,x)

def outflow_healthy(x, on_boundary):
    theta = -theta_healthy
    new_x = rotate(theta,x)
    return near(new_x[0],length*2) and on_boundary

def outflow_steno(x, on_boundary):
    theta = theta_steno
    new_x = rotate(theta,x)
    return near(new_x[0],length*2) and on_boundary

def wall_trunk(x):
    y_plus = diam_healthy_vessel * np.cos(theta_healthy)
    y_minus = -diam_steno_vessel * np.cos(theta_steno)
    return x[0] < DOLFIN_EPS and (near(x[1],y_plus) or near(x[1],y_minus))

def wall_healthy(x):
    theta = -theta_steno
    new_x = rotate(theta,x)
    return new_x[0] > - DOLFIN_EPS and (near(new_x[1],diam_healthy_vessel) or near(new_x[1],0))
        
def S(x,L):
    """
        Section of the stenosis following the paper
        "Direct numerical simulation of stenotic flows,
        Part 1: Steady flow" J. Fluid Mech.
    """
    
    # return D/2 * (1-diam_narrow*(1+np.cos(2*np.pi*(x-x0)/L)))
    # L = 2*diam_steno_vessel
    return diam_steno_vessel/2 -diam_narrow/2*(1+np.cos(2*np.pi*(x)/L))

def wall_steno(x):
    tol = 1e-3
    theta = theta_steno
    new_x = rotate(theta,x)
    new_x = new_x + np.array([-length,diam_steno_vessel/2])
    if new_x[0] < - length - DOLFIN_EPS:
        return False
    L = 2*diam_steno_vessel
    if new_x[0] > L/2 or new_x[0] < -L/2:
        return near(new_x[1],-diam_steno_vessel/2) or near(new_x[1],diam_steno_vessel/2)
    else: 
        return new_x[1] > S(new_x[0],L) - tol or new_x[1] < -S(new_x[0],L) + tol

def walls(x, on_boundary):
    return on_boundary and (wall_healthy(x) or wall_steno(x) or wall_trunk(x))

def compute_bc(V,Q,t,
            p_bdry_1,
            p_bdry_2,
            u0,
            s,
            inflow_expr,
            inflow_domain,
            heartfun,
            # bcu_walls=bcu_walls
            ):
    "In: V, Q. Out: boundary condition"

    heartval=heartfun(t)
    inflow_expr.set_values(heartval)
    bcu_inflow = DirichletBC(V, inflow_expr, inflow_domain)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu = [bcu_inflow, bcu_walls]

    bcp_outflow1 = DirichletBC(Q, Constant((p_bdry_1)), outflow_healthy)
    bcp_outflow2 = DirichletBC(Q, Constant((p_bdry_2)), outflow_steno)
    bcp = [bcp_outflow1,bcp_outflow2]
    return bcu, bcp


if __name__ == '__main__':
    c = 1                   #1.6e-5 distant capacitance
    Rd = 1e5                #6001.2 distant resistance
    Rp = 5e4                #7501.5 proximal resistance
    p_windkessel_1 = 1.06e5 # init val, large number could lead to overflow
    p_windkessel_2 = 1.06e5 # init val
    u0 = 2.                 # init amplitude
    s = .5                  # init asymmetry

    diam_steno_vessel=0.1
    diam_narrow=0.02
    theta_steno=np.pi/6
    diam_healthy_vessel=0.1
    theta_healthy=np.pi/6
    length0 = .5
    length = .3
    diam_trunk = diam_healthy_vessel * np.cos(theta_healthy) + diam_steno_vessel * np.cos(theta_steno)
    mesh_precision = 40
    artery = Artery(diam_steno_vessel, diam_narrow, theta_steno, diam_healthy_vessel, theta_healthy)
    mesh = artery.mesh(mesh_precision)

    # plot(mesh, title='stenosis')
    # plt.savefig('mesh.pdf')

    inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree=2)

    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    t = 0.

    bcu, bcp = compute_bc(V,Q,t)

    print('BC computation test passed.')

            
            
            
            
            
            
            
            
