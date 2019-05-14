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
import matplotlib
# from geometry import * 


import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.8,0.05])
gs.update(left=0.05, right=0.9, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

### debugging
# dynamic BC for debugging
flag_dynamic = True

# output movie or not
flag_movie = False

## Parameters
T = .05                    # final time
num_steps = 30         # number of time steps # must satisfy CFL condition
dt = T / num_steps      # time step size
mu = .03#0.03               # dynamic viscosity, poise
rho = 1                 # density, g/cm3
# windkessel
c = 1.6e-5    # 1          # distant capacitance
Rd = 2000#60012               # distant resistance
Rp = Rd * 1.2#75015             # proximal resistance
p_windkessel_1 = 1e3#140000#1.06e5#2000#1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = p_windkessel_1 * 1.#1.06e5#2000#1.06e5 # init val
u0=1.#1.
mesh_precision = 40
pmin_plot = 0#50000
pmax_plot = 10#150000

a = .5 
b = .2

# sqr2 = 2**.5    #constant for simplicity
# Y0 = 2          #Y trunk length
# Y1 = 2 * sqr2   #Y branch length
# y1 = 2          
# Y2 = 2 * sqr2   #Y branch length
# y2 = 2
# D = .1 * sqr2   #branch radius
# d = .1
# D0 = 2*d        #trunk radius
# A = .5 * sqr2         #shrink length = 2A
# a = .5 
# B = .1 * sqr2   #shrink width = B
# b = .1
# domain_vertices = [Point(-Y0        , -D0        ),
#                        Point(0.0        , -D0        ),
#                        Point(y2-d       , -y2-d      ),
#                        Point(y2+d       , -y2+d      ),
#                        Point(y2/2+d+a   , -y2/2+d-a  ),
#                        Point(y2/2+d+a-b , -y2/2+d-a-b),
#                        Point(y2/2+d-a-b , -y2/2+d+a-b),
#                        Point(y2/2+d-a   , -y2/2+d+a  ),
#                        Point(d*2        , 0          ),
#                        Point(y1+d       , y1-d       ),
#                        Point(y1-d       , y1+d       ),
#                        Point(0.0        , D0         ),
#                        Point(-Y0        , D0         )]

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

# diam_steno_vessel = 0.2
# diam_narrow = 0.2
# theta_steno = - np.pi /6

# diam_healthy_vessel = 0.2
# theta_healthy = + np.pi /6


# artery = Artery(diam_steno_vessel, diam_narrow, theta_steno, diam_healthy_vessel, theta_healthy)
# mesh = artery.mesh(40)

### Plot mesh
# plot(mesh)
# # plt.draw()
# # plt.pause(1)
# plt.show()

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

### Draft inflow with heartbeat
xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
yp=np.array([0.17,0.15,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.15])+.2
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
xx1=np.linspace(1.9 +tol-.1,2.1 -tol-.1,nn)
yy1=np.linspace(2.1 -tol-.1,1.9 +tol-.1,nn)
xx2=np.linspace(1.9 +tol-.1,2.1 -tol-.1,nn)
yy2=np.linspace(-2.1+tol+.1,-1.9-tol+.1,nn)
xx3=np.linspace(0.0 +tol,0.2 -tol,nn)
yy3=np.linspace(-0.2+tol,0.0 -tol,nn)
xy0=[i for i in zip(xx0,yy0)]
xy1=[i for i in zip(xx1,yy1)]
xy2=[i for i in zip(xx2,yy2)]
xy3=[i for i in zip(xx3,yy3)]
# print([x for x in xy0])# fine

### define a function to save data
def save_val_sameline(fhandle,pressure,grid):
    for i in grid:
        fhandle.write("%f, " % pressure(np.array(i)))
    fhandle.seek(0, os.SEEK_END)
    fhandle.seek(fhandle.tell() - 2, os.SEEK_SET)
    fhandle.truncate()
    fhandle.write("\n")
    return 0

### define integration operator
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
pmax = []
pmin = []
uavg1 = []
uavg2 = []
delta_windkessel1 = []
delta_windkessel2 = []

file_x = open("../output/X.txt", 'w+')
file_y = open("../output/Y.txt", 'w+')
for n in range(num_steps):
    # Update current time
    t += dt
    # xy0=zip(xx0,yy0)
    # xy1=zip(xx1,yy1)
    # xy2=zip(xx2,yy2)
    # xy3=zip(xx3,yy3)
    if flag_dynamic == True:
        u_at_bd_1 = [u_(np.array(i)) for i in xy1]
        u_normal_1 = [np.dot(u, np.array([1., 1.])/np.sqrt(2)) for u in u_at_bd_1]
        u_avg_1 = sum(u_normal_1)/nn*0.2*np.sqrt(2)
        # print(u_avg_1)
        uavg1.append(u_avg_1)

        u_at_bd_2 = [u_(np.array(i)) for i in xy2]
        u_normal_2 = [np.dot(u, np.array([1., -1.])/np.sqrt(2)) for u in u_at_bd_2]
        u_avg_2 = sum(u_normal_2)/nn*0.2*np.sqrt(2)
        # print(u_avg_2)
        uavg2.append(u_avg_2)

        # print(p_windkessel_1)
        delta1 = dt/c*(-p_windkessel_1/Rd+u_avg_1)
        delta_windkessel1.append(delta1)
        p_windkessel_1 += delta1
        delta2 = dt/c*(-p_windkessel_2/Rd+u_avg_2)
        p_windkessel_2 += delta2
        delta_windkessel2.append(delta2)

        p_bdry_1 = p_windkessel_1 + Rp * u_avg_1
        p_bdry_2 = p_windkessel_2 + Rp * u_avg_2
        # print(p_bdry_1,p_bdry_2)

        bcp_outflow1 = DirichletBC(Q, Constant((p_bdry_1)), outflow1)
        bcp_outflow2 = DirichletBC(Q, Constant((p_bdry_2)), outflow2)
        bcp = [bcp_outflow1,bcp_outflow2]

        # # print(type(f))
        heartval=heartfun(t)
        # inflow_expr = Expression('(fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *exp(-0.5*(log((0.2+x[1])/(0.2-x[1]))-0.1)**2),0)',
        #          degree=2, fval=heartval)
        inflow_expr.set_values(heartval)
        bcu_inflow = DirichletBC(V, inflow_expr, inflow)
        bcu = [bcu_inflow, bcu_walls]

    
    #### Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

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
    save_val_sameline(file_x,p_,xy0)
    save_val_sameline(file_y,p_,xy1)

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
    # xy0=zip(xx0,yy0)
    # xy1=zip(xx1,yy1)
    # xy2=zip(xx2,yy2)
    # xy3=zip(xx3,yy3)

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
    pmax.append(p_.vector().vec().max()[1])
    pmin.append(p_.vector().vec().min()[1])

    if flag_movie:
        # Plot solution
        # plt.cla()
        ax1 = plt.subplot(gs[0,0])
        plt1 = plot(p_, norm=matplotlib.colors.Normalize(vmin=0, vmax=pmax_plot, clip=False))
        plot(u_, title = "Pressure and Velocity,t = %.4f" % t)
        cbax = plt.subplot(gs[0,1])
        cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')


        # plt.show()
        # plt.cla()
        # pplot = plot(p_,norm=matplotlib.colors.Normalize(vmin=0, vmax=pmax_plot, clip=False))
        # uplot = plot(u_, title = "t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])
        fname = './tmp/_tmp%05d.png' % n
        # # print('Saving frame', fname)
        plt.savefig(fname)
        files.append(fname)
    

    # Update progress bar
#    progress.update(t / T)
    # if n % 20 == 0:
    pbar.update(dt)
    pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])
    
pbar.close()
file_x.close()
file_y.close()

#plot pressure at bdry:
plt.figure()
plt.subplot(311)
plt.plot(t_grid,p_at_bd_0,label='slice0')
plt.plot(t_grid,p_at_bd_1,label='slice1')
plt.plot(t_grid,p_at_bd_2,label='slice2')
plt.plot(t_grid,p_at_bd_3,label='slice3')
plt.legend(bbox_to_anchor=(1.4, 1), loc=2, borderaxespad=0.)

plt.subplot(312)
plt.plot(t_grid,uavg1,label='uavg')
plt.plot(t_grid,uavg2,label='uavg')
plt.subplot(313)
# plt.plot(t_grid,delta_windkessel1,label='slice1')
# plt.plot(t_grid,delta_windkessel2,label='slice2')
titlename = "T = %f, mu = %f, c = %f, Rd = %f p_windkessel_1 = %f, u0 = %f, rho = %f" % (T,mu,c,Rd,p_windkessel_1,u0,rho)
plt.title(titlename)
plt.savefig("../output/pressure.png")


# def mplot_function(ax, f, **kwargs):
#     #############plotting################

#     assert mesh == f.function_space().mesh()
#     gdim = mesh.geometry().dim()
#     tdim = mesh.topology().dim()

#     # Extract the function vector in a way that also works for
#     # subfunctions
#     try:
#         fvec = f.vector()
#     except RuntimeError:
#         fspace = f.function_space()
#         try:
#             fspace = fspace.collapse()
#         except RuntimeError:
#             return
#         fvec = dolfin.interpolate(f, fspace).vector()

#     if fvec.size() == mesh.num_cells():
#         # DG0 cellwise function
#         C = fvec.array()  # NB! Assuming here dof ordering matching cell numbering
#         if gdim == 2 and tdim == 2:
#             return ax.tripcolor(mesh2triang(mesh), C, **kwargs)
#         elif gdim == 3 and tdim == 2:  # surface in 3d
#             # FIXME: Not tested, probably broken
#             xy = mesh.coordinates()
#             shade = kwargs.pop("shade", True)
#             return ax.plot_trisurf(mesh2triang(mesh), xy[:, 2], C, shade=shade,
#                                    **kwargs)
#         elif gdim == 1 and tdim == 1:
#             x = mesh.coordinates()[:, 0]
#             nv = len(x)
#             # Insert duplicate points to get piecewise constant plot
#             xp = np.zeros(2*nv-2)
#             xp[0] = x[0]
#             xp[-1] = x[-1]
#             xp[1:2*nv-3:2] = x[1:-1]
#             xp[2:2*nv-2:2] = x[1:-1]
#             Cp = np.zeros(len(xp))
#             Cp[0:len(Cp)-1:2] = C
#             Cp[1:len(Cp):2] = C
#             return ax.plot(xp, Cp, *kwargs)
#         # elif tdim == 1:  # FIXME: Plot embedded line
#         else:
#             raise AttributeError('Matplotlib plotting backend only supports 2D mesh for scalar functions.')

#     elif f.value_rank() == 0:
#         # Scalar function, interpolated to vertices
#         # TODO: Handle DG1?
#         C = f.compute_vertex_values(mesh)
#         if gdim == 2 and tdim == 2:
#             mode = kwargs.pop("mode", "contourf")
#             if mode == "contourf":
#                 levels = kwargs.pop("levels", 40)
#                 return ax.tricontourf(mesh2triang(mesh), C, levels, **kwargs)
#             elif mode == "color":
#                 shading = kwargs.pop("shading", "gouraud")
#                 return ax.tripcolor(mesh2triang(mesh), C, shading=shading,
#                                     **kwargs)
#             elif mode == "warp":
#                 from matplotlib import cm
#                 cmap = kwargs.pop("cmap", cm.jet)
#                 linewidths = kwargs.pop("linewidths", 0)
#                 return ax.plot_trisurf(mesh2triang(mesh), C, cmap=cmap,
#                                        linewidths=linewidths, **kwargs)
#             elif mode == "wireframe":
#                 return ax.triplot(mesh2triang(mesh), **kwargs)
#             elif mode == "contour":
#                 return ax.tricontour(mesh2triang(mesh), C, **kwargs)
#         elif gdim == 3 and tdim == 2:  # surface in 3d
#             # FIXME: Not tested
#             from matplotlib import cm
#             cmap = kwargs.pop("cmap", cm.jet)
#             return ax.plot_trisurf(mesh2triang(mesh), C, cmap=cmap, **kwargs)
#         elif gdim == 3 and tdim == 3:
#             # Volume
#             # TODO: Isosurfaces?
#             # Vertex point cloud
#             X = [mesh.coordinates()[:, i] for i in range(gdim)]
#             return ax.scatter(*X, c=C, **kwargs)
#         elif gdim == 1 and tdim == 1:
#             x = mesh.coordinates()[:, 0]
#             ax.set_aspect('auto')

#             p = ax.plot(x, C, **kwargs)

#             # Setting limits for Line2D objects
#             # Must be done after generating plot to avoid ignoring function
#             # range if no vmin/vmax are supplied
#             vmin = kwargs.pop("vmin", None)
#             vmax = kwargs.pop("vmax", None)
#             ax.set_ylim([vmin, vmax])

#             return p
#         # elif tdim == 1: # FIXME: Plot embedded line
#         else:
#             raise AttributeError('Matplotlib plotting backend only supports 2D mesh for scalar functions.')

#     elif f.value_rank() == 1:
#         # Vector function, interpolated to vertices
#         w0 = f.compute_vertex_values(mesh)
#         nv = mesh.num_vertices()
#         if len(w0) != gdim*nv:
#             raise AttributeError('Vector length must match geometric dimension.')
#         X = mesh.coordinates()
#         X = [X[:, i] for i in range(gdim)]
#         U = [w0[i*nv: (i + 1)*nv] for i in range(gdim)]

#         # Compute magnitude
#         C = U[0]**2
#         for i in range(1, gdim):
#             C += U[i]**2
#         C = np.sqrt(C)

#         mode = kwargs.pop("mode", "glyphs")
#         if mode == "glyphs":
#             args = X + U + [C]
#             if gdim == 3:
#                 # 3d quiver plot works only since matplotlib 1.4
#                 import matplotlib
#                 if StrictVersion(matplotlib.__version__) < '1.4':
#                     cpp.warning('Matplotlib version %s does not support 3d '
#                                 'quiver plot. Continuing without plotting...'
#                                 % matplotlib.__version__)
#                     return

#                 length = kwargs.pop("length", 0.1)
#                 return ax.quiver(*args, length=length, **kwargs)
#             else:
#                 return ax.quiver(*args, **kwargs)
#         elif mode == "displacement":
#             Xdef = [X[i] + U[i] for i in range(gdim)]
#             import matplotlib.tri as tri
#             if gdim == 2 and tdim == 2:
#                 # FIXME: Not tested
#                 triang = tri.Triangulation(Xdef[0], Xdef[1], mesh.cells())
#                 shading = kwargs.pop("shading", "flat")
#                 return ax.tripcolor(triang, C, shading=shading, **kwargs)
#             else:
#                 # Return gracefully to make regression test pass without vtk
#                 cpp.warning('Matplotlib plotting backend does not support '
#                             'displacement for %d in %d. Continuing without '
#                             'plotting...' % (tdim, gdim))
#                 return
    #############end plotting################


# plt.figure()
# ax1 = plt.subplot(gs[0,0])
# plt1 = plot(p_)
# plt2 = plot(u_, title = "Pressure and Velocity,t = %.4f" % t)
# cbax = plt.subplot(gs[0,1])
# cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')
# plt.show()


# ax1 = plt.subplot(gs[0,0])
# ax1.set_aspect('equal')
# import matplotlib
# plt1 = mplot_function(ax1,p_,norm=matplotlib.colors.Normalize(vmin=0, vmax=2000, clip=False))
# plt2 = mplot_function(ax1,u_)#,norm=matplotlib.colors.Normalize(vmin=0, vmax=10000, clip=False))
# cbax = plt.subplot(gs[0,1])
# cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')
# plt.show()



if flag_movie:
    print('Making animation - this may take a while')
    # subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
    #                 "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)
    subprocess.call("ffmpeg -i ./tmp/_tmp%05d.png -r 60 -pix_fmt yuv420p ../output/output.mp4", shell=True)

    # cleanup
    for fname in files:
        os.remove(fname)


# ## test: extract solution
# def save_val(fname,pressure,grid):
#     with open(fname, 'w+') as f:
#         for i in grid:
#             f.write("%f, %f, %f\n" % (i[0], i[1], pressure(np.array(i))))
#     return 0

# save_val("../output/pressure0.txt",p_,xy0)
# save_val("../output/pressure1.txt",p_,xy1)
# save_val("../output/pressure2.txt",p_,xy2)


