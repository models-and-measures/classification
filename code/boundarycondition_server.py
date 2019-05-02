from fenics import *
import numpy as np
from scipy.interpolate import interp1d

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
a=.5                    # vessel shrink length = 2a
b=.0                    # vessel shrink intensity = b
u0=1.                   # init amplitude


def compute_bc(mesh,
            T = T,
            num_steps = num_steps,
            mu = mu,
            rho = rho,
            c = c,
            Rd = Rd,
            Rp = Rp,
            p_windkessel_1 =p_windkessel_1,
            p_windkessel_2 =p_windkessel_2,
            a=a,
            b=b,
            u0=u0,
            ):
    "In: mesh. Out: boundary condition"
    print(num_steps)
    dt = T / num_steps
    ### Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    
    ### Define boundaries
    inflow   = 'near(x[0], -2.0)'
    def outflow1(x, on_boundary):
        return  near(x[0]+x[1], 4) and on_boundary
    def outflow2(x, on_boundary):
        return  near(x[0]-x[1], 4) and on_boundary
    A = (a*2**.5)
    B = (b*2**.5)
    def walls(x, on_boundary):
        return  ( \
                (near(abs(x[1]), 0.2) and x[0] <= 0 + DOLFIN_EPS) or \
                (near(x[0] - abs(x[1]), 0.2) and x[0] >= 0.2 - DOLFIN_EPS) or \
                (near(x[0] - abs(x[1]), - 0.2) and x[0] >= 0.0 - DOLFIN_EPS) or \
                (near(abs(x[1] - x[0] + 2), A) and x[0] + x[1] >= 0.2 - B - DOLFIN_EPS)  or \
                (near(x[0] + x[1], 0.2 - B) and abs(x[0] - x[1] - 2) <= A + DOLFIN_EPS) \
                ) \
                and on_boundary

    ### Define inflow with heartbeat
    xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
    yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])+.2
    heartfun0 = interp1d(xp, yp,kind='cubic')
    heartfun = lambda x: heartfun0(x % 1.0)
    # xxx=np.linspace(0,1,100)
    # plt.plot(xxx,heartfun(xxx))
    # Define inflow shape
    class INFLOW(UserExpression):
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

    inflow_expr = INFLOW(u0,0.0)
    t = 0
    heartval=heartfun(t)
    # print(type(fval))
    inflow_expr.set_values(heartval)
    bcu_inflow = DirichletBC(V, inflow_expr, inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu = [bcu_inflow, bcu_walls]

    bcp_outflow1 = DirichletBC(Q, Constant((p_windkessel_1)), outflow1)
    bcp_outflow2 = DirichletBC(Q, Constant((p_windkessel_2)), outflow2)
    bcp = [bcp_outflow1,bcp_outflow2]
    return bcu, bcp


if __name__ == '__main__':
    from mesh_server import * 
    mesh = compute_mesh()
    bcu, bcp = compute_bc(mesh)
    print('BC computation complete.')

            
            
            
            
            
            
            
            
