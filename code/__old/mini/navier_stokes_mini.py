from dolfin import *
from mshr import * # mesh

# from tqdm import tqdm # progress bar
import numpy as np
import matplotlib.pyplot as plt
from geometry import * # requires mesh parameters
from scipy.interpolate import interp1d

####################
# diam_steno_vessel=0.1
# diam_narrow=0.02
# theta_steno=-np.pi/6
# diam_healthy_vessel=0.1
# theta_healthy=-np.pi/6
# length0 = .5
# length = .3
# diam_trunk = diam_healthy_vessel * np.cos(theta_healthy) + diam_steno_vessel * np.cos(theta_steno)

c = 1                   #1.6e-5 distant capacitance
Rd = 1e5                #6001.2 distant resistance
Rp = 5e4                #7501.5 proximal resistance
p_windkessel_1 = 1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = 1.06e5 # init val
u0 = 1.                 # init amplitude
s = 0.                  # init asymmetry
####################
results_dir = './result/'

rho = 1.
mu  = 0.03

T = .01
num_step = 10
u0 = 1.
s = 0.

dt  = T/num_step

t = dt

with_teman  = False
with_bf_est = True
####################

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
    theta = theta_steno
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
    theta = -theta_steno
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


### Define inflow with heartbeat
xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])
heartfun0 = interp1d(xp, yp,kind='cubic')
heartfun = lambda x: heartfun0(x % 1.0)


# Heart
class Heart():
        def __init__(self):
            xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
            yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])
            self.heartfun0 = interp1d(xp, yp,kind='cubic')
                
        def beat(self,t):
                return self.heartfun0(t % 1.0)

heart = Heart()

# Artery and mesh
artery = Artery(diam_steno_vessel=diam_steno_vessel, diam_narrow=diam_narrow, theta_steno=theta_steno, diam_healthy_vessel=diam_healthy_vessel, theta_healthy=theta_healthy,length0 = length0,length = length)
mesh_precision = 40
mesh = artery.mesh(mesh_precision)
plot(mesh, title='stenosis')
plt.savefig(results_dir+'mesh.pdf')
plt.close()

# Define function spaces
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
B = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
Q = P1
V = VectorElement(P1 + B)
mix = V * Q
# print(mix) # <Mixed element: (<vector element with 2 components of <<CG1 on a triangle> + <B3 on a triangle>>>, <CG1 on a triangle>)>
Mini = FunctionSpace(mesh, mix)

# No-slip boundary condition for velocity
noslip = project(Constant((0, 0)), Mini.sub(0).collapse())
bcu_walls = DirichletBC(Mini.sub(0), noslip, walls)

# Inflow bc with heartbeat
inflow_expr = Expression(("5+2.5*sin(2*DOLFIN_PI/100*t)", "0"), t = 0, degree = 1) # TODO: Change this
def inflow_domain(x, on_boundary):
    return  near(x[0]+length0, 0) and on_boundary
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
        tol = 1E-10
        if x[1] - y_plus < - tol and x[1] + y_minus > tol:
            # print(self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-0.1)**2))
            values[0] = u0*fval/(y_minus + x[1])/(y_plus - x[1]) * pow(y_minus + y_plus, 2) *np.exp(-0.5*(np.log((y_minus + x[1])/(y_plus - x[1]))-self.s)**2)
        else:
            values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)

# inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree = 1)
# inflow_expr.set_values(t)

def compute_bc(Mini,t,
            p_bdry_1 =p_windkessel_1,
            p_bdry_2 =p_windkessel_2,
            u0=u0,
            s=s,
            inflow_expr=inflow_expr,
            inflow_domain=inflow_domain,
            heartfun=heartfun,
            # bcu_walls=bcu_walls
            ):
    "In: V, Q. Out: boundary condition"

    # heartval=heartfun(t)
    # heartval=heart.beat(t)
    # inflow_expr.set_values(heartval)
    bcu_inflow = DirichletBC(Mini.sub(0), project(inflow_expr, Mini.sub(0).collapse()), inflow_domain)
    noslip = project(Constant((0, 0)), Mini.sub(0).collapse())
    bcu_walls = DirichletBC(Mini.sub(0), noslip, walls)
    # bcu = [bcu_inflow, bcu_walls]

    bcp_outflow1 = DirichletBC(Mini.sub(1), Constant((p_bdry_1)), outflow_healthy)
    bcp_outflow2 = DirichletBC(Mini.sub(1), Constant((p_bdry_2)), outflow_steno)
    # bcp = [bcp_outflow1,bcp_outflow2]
    return [bcu_inflow, bcu_walls]
            #, bcp_outflow1,bcp_outflow2]

# Collect boundary conditions
# bcs = [bcu_walls, bcu_inflow]

# Define variational problem
w0 = Function(Mini)
# u0 = Function(Mini.sub(0).collapse())
u0 = project(inflow_expr, Mini.sub(0).collapse())
p0 = Function(Mini.sub(1).collapse())

(u, p) = TrialFunctions(Mini)
(v, q) = TestFunctions(Mini)

# Variatonal form a = L
def abs_n(x):
    return 0.5*(x - abs(x))

n = FacetNormal(mesh)

a   = rho/dt*inner(u,  v)*dx \
    + mu*inner(grad(u), grad(v))*dx + q*div(u)*dx \
    - div(v)*p*dx \
    + rho*inner(grad(u)*u0, v)*dx \

if with_teman:
    a = a + 0.5*rho*(div(u0)*inner(u, v))*dx 
if with_bf_est:
    a = a - 0.1*rho*abs_n(dot(u0, n))*inner(u, v)*ds(2) 

L   = rho/dt*inner(u0, v)*dx

## Define Subdomain marker
# boundary_markers = FacetFunction('size_t', mesh)
# class ourflow_boundary(SubDomain):
#     tol = 1E-14
#     def inside(self, x, on_boundary):
#         return outflow(x,on_boundary)
# ob = ourflow_boundary()
# ob.mark(boundary_markers, 2)
# ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Configure solver
w = Function(Mini)

# # ******************************************
# # Output files
# ufile_pvd = File(results_dir + "velocity.pvd")
# pfile_pvd = File(results_dir + "pressure.pvd")
# ufile_h5 = HDF5File(mesh.mpi_comm(), results_dir + "velocity.h5", "w")
# pfile_h5 = HDF5File(mesh.mpi_comm(), results_dir + "pressure.h5", "w")
# # Save initial condition to file
# ufile_pvd << u0, 0
# pfile_pvd << p0, 0
# ufile_h5.write(u0, "velocity_0") 
# pfile_h5.write(p0, "pressure_0")

# ******************************************
# Time loop
# pbar = tqdm(total=T)
while t < T + dt:
    
    # Update bc
    # inflow_expr.t = t
    bcu_inflow = DirichletBC(Mini.sub(0), project(inflow_expr, Mini.sub(0).collapse()), inflow_domain)

    # collect boundary conditions and assemble system
    bcs = [bcu_walls, bcu_inflow] # TODO: Add Windkessel
    # bcs = compute_bc(Mini,t)
    A, b = assemble_system(a, L, bcs)

    # solving linear system
    solve(A, w.vector(), b, 'superlu')

    ut, pt = w.split(deepcopy = True)

    # Save solution to file
    # ufile_pvd << ut, t
    # pfile_pvd << pt, t
    # ufile_h5.write(ut, "velocity_" + str(t))
    # pfile_h5.write(pt, "pressure_" + str(t))
               
    # update variables
    u0.assign(ut)
    p0.assign(pt)
    t += dt

    plot(ut)
    plt.savefig(results_dir+'u-t-'+str(t)+'.pdf')
    plt.close()

    c = plot(pt)
    plt.savefig(results_dir+'p-t-'+str(t)+'.pdf', scalarbar = True)
    plt.close()

    # Update progress bar
    print(t)
    # pbar.update(dt)
    # pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])


