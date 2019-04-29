# -*- coding: utf-8 -*-
"""
Navier Stokes Equation with Windkessel Model
Changqing Fu 2019
"""

"""

    rho(u' + u . nabla(u)) - div(sigma(u, p)) = f = 0
                                       div(u) = 0
    sigma(u,p) = 2 mu epsilon(u) - p I
    epsilon(u) = (nabla(u) + nabla(u)^T) / 2 = sym(nabla_grad(u))
"""
### Import 
from tqdm import tqdm # status bar
from dolfin import * # FEM solver
from mshr import * # mesh
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from scipy.interpolate import interp1d
np.seterr(invalid='raise')

### debugging
# dynamic BC for debugging
flag_dynamic = True

# output movie or not
flag_movie = True

### Parameters
T = .1                   # final time
num_steps = 200        # number of time steps # must satisfy CFL condition
dt = T / num_steps      # time step size
mu = 0.03               # dynamic viscosity, poise
rho = 1                 # density, g/cm3
# windkessel
c = 1.6e-5              # distant capacitance
Rd = 6001.2             # distant resistance
Rp = 7501.5             # proximal resistance
p_windkessel_1 = 1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = 1.06e5 # init val
a=.5                    # vessel shrink length = 2a
b=.1                    # vessel shrink intensity = b
u0=20
mesh_precision = 32

#### Create mesh
domain_vertices = [Point(-2.0, -0.2),
                   Point(0.0, -0.2),
                   Point(1.9,-2.1),
                   Point(2.1,-1.9),
                   Point(1.1+a/2**.5,-0.9-a/2**.5),
                   Point(1.1+a/2**.5-b/2**.5,-0.9-a/2**.5-b/2**.5),
                   Point(1.1-a/2**.5-b/2**.5,-0.9+a/2**.5-b/2**.5),
                   Point(1.1-a/2**.5,-0.9+a/2**.5),
                   Point(0.2,0),
                   Point(2.1,1.9),
                   Point(1.9,2.1),
                   Point(0.0,0.2),
                   Point(-2.0,0.2)]
polygon = Polygon(domain_vertices)
domain = polygon
mesh = generate_mesh(domain, mesh_precision)

### Plot mesh
plot(mesh)
plt.draw()
plt.pause(1)
plt.show()

### Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

### Define boundaries
inflow   = 'near(x[0], -2.0)'
# def outflow(x, on_boundary):
#     return  (near(x[0]-x[1], 4) or near(x[0]+x[1], 4)) and on_boundary
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
yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])
heartfun0 = interp1d(xp, yp,kind='cubic')
heartfun = lambda x: heartfun0(x % 1.0)
# xxx=np.linspace(0,1,100)
# plt.plot(xxx,heartfun(xxx))
# Define inflow shape
class INFLOW(UserExpression):
    def __init__(self, u0, **kwargs):
        super().__init__(**kwargs) #super makes sure all superclass are initiated...
        self.u0 = u0
    def set_values(self,heartval):
        self.fval = heartval # heart signal (period = 1s)
    def eval(self, values, x):
        "Set value[0] to value at point x"
        tol = 1E-13
        if x[1]-0.2 < - tol and x[1] + 0.2 > tol:
            # print(self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-0.1)**2))
            values[0] = self.u0*self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-0.1)**2)
        else:
            values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)

inflow_expr = INFLOW(u0)
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

#### Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

#### Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

#### Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)
# print(dt/c*(p/R))

#### Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

#### Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

#### Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

#### Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

#### Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

#### Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

#### Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# # Create XDMF files for visualization output
# xdmffile_u = XDMFFile('Y_shape_solution/velocity.xdmf')
# xdmffile_p = XDMFFile('Y_shape_solution/pressure.xdmf')

# # Create time series (for use in reaction_system.py)
# timeseries_u = TimeSeries('Y_shape_solution/velocity_series')
# timeseries_p = TimeSeries('Y_shape_solution/pressure_series')

# # Save mesh to file (for use in reaction_system.py)
# xdmffile_mesh = File('Y_shape_solution/Y.xml.gz')
# xdmffile_mesh << mesh

### define diagnosis intersection surface/line
nn=10
tol=1e-6
xx0=-1*np.ones(nn)
yy0=np.linspace(-.2 +tol,.2  -tol,nn)
xx1=np.linspace(1.9 +tol,2.1 -tol,nn)
yy1=np.linspace(-2.1+tol,-1.9-tol,nn)
xx2=np.linspace(1.9 +tol,2.1 -tol,nn)
yy2=np.linspace(2.1 -tol,1.9 +tol,nn)
xx3=np.linspace(0.0 +tol,0.2 -tol,nn)
yy3=np.linspace(-0.2+tol,0.0 -tol,nn)
xy0=zip(xx0,yy0)
xy1=zip(xx1,yy1)
xy2=zip(xx2,yy2)
xy3=zip(xx3,yy3)
# print([x for x in xy0])# fine

### defin integration operator
def average_over_line(pressure,grid):
    return np.mean([pressure(np.array(i)) for i in grid])


# Create progress bar
#progress = dolfin.cpp.log.Progress('Time-stepping')
#set_log_level(PROGRESS)
pbar = tqdm(total=T)

### Time-stepping
files = []
t = 0.0
# init windkessel pressure
p_bdry_1 = 0.0
p_bdry_2 = 0.0
# init diagnostic profile
t_grid = []
p_at_bd_0 = []
p_at_bd_1 = []
p_at_bd_2 = []
p_at_bd_3 = []
for n in range(num_steps):
    # Update current time
    t += dt
    xy0=zip(xx0,yy0)
    xy1=zip(xx1,yy1)
    xy2=zip(xx2,yy2)
    xy3=zip(xx3,yy3)
    if flag_dynamic == True:
        # u_at_bd_1 = [u_(np.array(i)) for i in xy1]
        # u_normal_1 = [(x[0]-x[1])/np.sqrt(2) for x in u_at_bd_1]
        # u_avg_1 = sum(u_normal_1)*0.2*2**0.5/nn

        # u_at_bd_2 = [u_(np.array(i)) for i in xy2]
        # u_normal_2 = [(x[0]-x[1])/np.sqrt(2) for x in u_at_bd_2]
        # u_avg_2 = sum(u_normal_2)*0.2*2**0.5/nn

        # # print(u_at_bd_1)
        # p_windkessel_1 += dt/c*(-p_windkessel_1/Rd+u_avg_1)
        # p_windkessel_2 += dt/c*(-p_windkessel_2/Rd+u_avg_2)

        p_bdry_1 = p_windkessel_1 #+ Rp * u_avg_1
        p_bdry_2 = p_windkessel_2 #+ Rp * u_avg_2

        bcp_outflow1 = DirichletBC(Q, Constant((p_bdry_1)), outflow1)
        bcp_outflow2 = DirichletBC(Q, Constant((p_bdry_2)), outflow2)
        bcp = [bcp_outflow1,bcp_outflow2]

        # print(type(f))
        heartval=heartfun(t)
        # inflow_expr = Expression('(fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *exp(-0.5*(log((0.2+x[1])/(0.2-x[1]))-0.1)**2),0)',
        #          degree=2, fval=fval)
        inflow_expr.set_values(heartval)
        # bcu_inflow = DirichletBC(V, inflow_expr, inflow)
        # bcu = [bcu_inflow, bcu_walls]

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # # Save solution to file (XDMF/HDF5)
    # xdmffile_u.write(u_, t)
    # xdmffile_p.write(p_, t)

    # # Save nodal values to file
    # timeseries_u.store(u_.vector(), t)
    # timeseries_p.store(p_.vector(), t)

    # Update previous solution
    p_n.assign(p_)
    u_n.assign(u_)

    # Update pressure at profile
    
    # xx0=-1*np.ones(nn)
    # yy0=np.linspace(-.2,.2,nn)
    # xx1=np.linspace(1.9,2.1,nn)
    # yy1=np.linspace(-2.1,-1.9,nn)
    # xx2=np.linspace(1.9,2.1,nn)
    # yy2=np.linspace(2.1,1.9,nn)
    # xx3=np.linspace(0.0,0.2,nn)
    # yy3=np.linspace(-0.2,0.0,nn)
    xy0=zip(xx0,yy0)
    xy1=zip(xx1,yy1)
    xy2=zip(xx2,yy2)
    xy3=zip(xx3,yy3)

    # print([x for x in xy0])
    diag0 = np.mean([p_(np.array(i)) for i in xy0])
    diag1 = np.mean([p_(np.array(i)) for i in xy1])
    diag2 = np.mean([p_(np.array(i)) for i in xy2])
    diag3 = np.mean([p_(np.array(i)) for i in xy3])
    t_grid.append(t)
    p_at_bd_0.append(diag0)
    p_at_bd_1.append(diag1)
    p_at_bd_2.append(diag2)
    p_at_bd_3.append(diag3)

    if flag_movie:
        # # Plot solution
        plt.cla()
        plot(p_)
        plot(u_, title = "t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])
        fname = '_tmp%03d.png' % n
        # print('Saving frame', fname)
        plt.savefig(fname)
        files.append(fname)
    

    # Update progress bar
#    progress.update(t / T)
    # if n % 20 == 0:
    pbar.update(dt)
    pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])
    
pbar.close()

#plot pressure at bdry:
plt.figure()
plt.plot(t_grid,p_at_bd_0,label='slice0')
plt.plot(t_grid,p_at_bd_1,label='slice1')
plt.plot(t_grid,p_at_bd_2,label='slice2')
plt.plot(t_grid,p_at_bd_3,label='slice3')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.8,0.05])
gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

ax1 = plt.subplot(gs[0,0])
plt1 = plot(p_)
plt2 = plot(u_, title = "Pressure and Velocity,t = %.4f" % t)
cbax = plt.subplot(gs[0,1])
cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')
plt.show()


if flag_movie:
    print('Making animation - this may take a while')
    # subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
    #                 "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)
    subprocess.call("ffmpeg -i _tmp%03d.png -pix_fmt yuv420p ../output/output.mp4", shell=True)

    # cleanup
    for fname in files:
        os.remove(fname)