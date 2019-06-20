from fenics import *
import numpy as np
from scipy.interpolate import interp1d
from geometry import * # requires mesh parameters

def inflow_domain(x, on_boundary):
    "Define boundaries"
    return near(x[0]+length0, 0) and on_boundary

def heartfun(x):
    "Define inflow with heartbeat"
    xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
    yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])#+.2
    heartfun0 = interp1d(xp, yp,kind='cubic')
    return heartfun0(x % 1.0)# periodic

# alternatively,
class Heart():
    def __init__(self):
        xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
        yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])
        self.heartfun0 = interp1d(xp, yp,kind='cubic')
            
    def beat(self,t):
        return self.heartfun0(t % 1.0)

# heart = Heart()

class INFLOW(UserExpression):
    "Inflow function"
    def __init__(self, u0, s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy, **kwargs):
        super().__init__(**kwargs) #super makes sure all superclass are initiated...
        self.u0 = u0
        self.s = s
        self.y_plus = diam_healthy_vessel * np.cos(theta_healthy)
        self.y_minus = diam_steno_vessel * np.cos(theta_steno)
    def set_values(self,heartval):
        "Set the only varying parameter, namely the current time"
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
            # print(type(u0*fval/(y_minus + x[1])/(y_plus - x[1]) * pow(y_minus + y_plus, 2) *np.exp(-0.5*(np.log((y_minus + x[1])/(y_plus - x[1]))-self.s)**2)),type(values[0]))
            values[0] = u0*fval/(y_minus + x[1])/(y_plus - x[1]) * pow(y_minus + y_plus, 2) *np.exp(-0.5*(np.log((y_minus + x[1])/(y_plus - x[1]))-self.s)**2)
            # values[0] = 1
        else:
            values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)

# bcu_walls = DirichletBC(V, Constant((0, 0)), walls)

def rotate(theta,x):
    "rotate CLOCKWISE"
    n0 = np.cos(theta)
    n1 = np.sin(theta)
    rotation = np.array([[n0,n1],[-n1,n0]])
    return np.matmul(rotation,x)

def outflow_healthy(x, on_boundary):
    theta = theta_healthy
    new_x = rotate(theta,x)
    return near(new_x[0],length*2) and on_boundary

def outflow_steno(x, on_boundary):
    theta = -theta_steno
    new_x = rotate(theta,x)
    return near(new_x[0],length*2) and on_boundary

def outflow(x, on_boundary):
    return on_boundary and not inflow(x, on_boundary) and not walls(x, on_boundary)

def wall_trunk(x):
    y_plus = diam_healthy_vessel * np.cos(theta_healthy)
    y_minus = -diam_steno_vessel * np.cos(theta_steno)
    return x[0] < DOLFIN_EPS and (near(x[1],y_plus) or near(x[1],y_minus))

def wall_healthy(x):
    theta = theta_healthy
    new_x = rotate(theta,x)
    return new_x[0] > - DOLFIN_EPS and (near(new_x[1],diam_healthy_vessel) or near(new_x[1],0))

def S(x):
    """
        Section of the stenosis following the paper
        "Direct numerical simulation of stenotic flows,
        Part 1: Steady flow" J. Fluid Mech.
    """
    
    # return D/2 * (1-diam_narrow*(1+np.cos(2*np.pi*(x-x0)/L)))
    # L = 2*diam_steno_vessel
    return diam_steno_vessel/2 - diam_narrow/2*(1+np.cos(2*np.pi*(x)/length_steno))

def wall_steno(x):
    tol = .0008
    theta = -theta_steno
    new_x = rotate(theta,x)
    new_x = new_x + np.array([-length,diam_steno_vessel/2])
    if new_x[0] <= - length - DOLFIN_EPS:
        return False
    if new_x[0] >= length_steno/2 or new_x[0] < -length_steno/2:
        return near(new_x[1],-diam_steno_vessel/2) or near(new_x[1],diam_steno_vessel/2)
    else: 
        return near(new_x[1],S(new_x[0]),tol) or near(new_x[1],-S(new_x[0]),tol)

def walls(x, on_boundary):
    return on_boundary and (wall_healthy(x) or wall_steno(x) or wall_trunk(x))

def compute_funsp(mesh):
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    return V,Q

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
    "P2-P1 mesh. In: V, Q. Out: boundary condition"
    heartval=heartfun(t)
    inflow_expr.set_values(heartval)
    bcu_inflow = DirichletBC(V, inflow_expr, inflow_domain)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu = [bcu_inflow, bcu_walls]

    bcp_outflow1 = DirichletBC(Q, Constant((p_bdry_1)), outflow_healthy)
    bcp_outflow2 = DirichletBC(Q, Constant((p_bdry_2)), outflow_steno)
    bcp = [bcp_outflow1,bcp_outflow2]
    return bcu, bcp

def compute_mini(mesh):
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    B = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
    Q = P1
    V = VectorElement(P1 + B)
    mix = V * Q
    # print(mix) # <Mixed element: (<vector element with 2 components of <<CG1 on a triangle> + <B3 on a triangle>>>, <CG1 on a triangle>)>
    Mini = FunctionSpace(mesh, mix)
    return Mini

def compute_bc_mini(Mini,t,
            p_bdry_1,
            p_bdry_2,
            u0,
            inflow_expr,
            inflow_domain,
            heartfun,
            # bcu_walls=bcu_walls
            ):
    "`mini' mesh. In: mesh. Out: boundary condition"
    heartval=heartfun(t)
    # heartval=heart.beat(t)
    inflow_expr.set_values(heartval)
    bcu_inflow = DirichletBC(Mini.sub(0), project(inflow_expr, Mini.sub(0).collapse()), inflow_domain)
    noslip = project(Constant((0, 0)), Mini.sub(0).collapse())
    bcu_walls = DirichletBC(Mini.sub(0), noslip, walls)
    # bcu = [bcu_inflow, bcu_walls]

    bcp_outflow1 = DirichletBC(Mini.sub(1), Constant((p_bdry_1)), outflow_healthy)
    bcp_outflow2 = DirichletBC(Mini.sub(1), Constant((p_bdry_2)), outflow_steno)
    # bcp_outflow1 = DirichletBC(Mini.sub(1), Constant((p_bdry_1)), inflow_domain)
    # bcp_outflow2 = DirichletBC(Mini.sub(1), Constant((p_bdry_2)), inflow_domain)
    # bcp = [bcp_outflow1,bcp_outflow2]
    return [bcu_inflow, bcu_walls, bcp_outflow1, bcp_outflow2]

if __name__ == '__main__':
    artery = Artery(diam_steno_vessel, diam_narrow, theta_steno, diam_healthy_vessel, theta_healthy,length0,length, length_steno)
    mesh = artery.mesh(mesh_precision)
    Mini = compute_mini(mesh)
    inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree=2)
    t = 0.
    bcs = compute_bc_mini(Mini,t,
            p_bdry_1 =1,
            p_bdry_2 =1,
            u0=1,
            inflow_expr=inflow_expr,
            inflow_domain=inflow_domain,
            heartfun=heartfun,)

    V,Q = compute_funsp(mesh)
    compute_bc(V,Q,t,
            1,
            1,
            u0,
            s,
            inflow_expr,
            inflow_domain,
            heartfun)

    print('BC computation test passed.')
