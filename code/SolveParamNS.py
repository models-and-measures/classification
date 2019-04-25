# -*- coding: utf-8 -*-
"""

"""

"""

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""
from tqdm import tqdm
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess


## TODO: object programming

# dynamic BC
flag_dynamic = True

# output or not
flag_movie = True

# Parameters
T = 2              # final time
num_steps = 2000      # number of time steps # must satisfy CFL condition
dt = T / num_steps  # time step size
mu = 0.03            # dynamic viscosity, poise
rho = 1              # density, g/cm3
c = 1.6e-5
R = 7501.5
p_windkessel_1 = 1.06e5 # large number leads to overflow
p_windkessel_2 = 1.06e5

#### Create mesh
a=.5
b=.1
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
mesh = generate_mesh(domain, 100)
# plot(mesh)
# plt.draw()
# plt.pause(1)
# plt.show()

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], -2.0)'
# outflow  = 'near(x[0]-x[1], 4) || near(x[0]+x[1], 4)'
# outflow1  = 'near(x[0]-x[1], 4)'
# outflow2  = 'near(x[0]+x[1], 4)'
def outflow(x, on_boundary):
    return  (near(x[0]-x[1], 4) or near(x[0]+x[1], 4)) and on_boundary
def outflow1(x, on_boundary):
    return  near(x[0]-x[1], 4) and on_boundary
def outflow2(x, on_boundary):
    return  near(x[0]+x[1], 4) and on_boundary
# # Define walls without shrinkage (abandoned)
# walls    = '((near(x[1], -0.2) || near(x[1], 0.2)) && x[0]<=0)' \
#             +' || ((near(x[1] - x[0], 0.2) || near(x[1] + x[0], -0.2)) && x[0]>0 )' \
#             +' || ((near(x[1] - x[0], -0.2) || near(x[1] + x[0], 0.2)) && x[0]>0.2 ) ' 
# # Define Cylinder (abandoned)
# cylinder = 'on_boundary && x[0]>0.91 && x[0]<1.09 && x[1]>-1.09 && x[1]<-0.91 ' 
# Define walls with shrinkage
# # this doesn't work:
# A = str(a*2**.5)
# B = str(b*2**.5)
# gamma0   = '(near(abs(x[1]), 0.2 && x[0]<=0) '
# gamma1   = '(near(x[0] - abs(x[1]), 0.2) && x[0] >= 0) '
# gamma2   = '(near(abs(x[0] - x[1] + 1), ' + A + ') && x[0] + x[1] >= 0.2 - ' + B + ') ' 
# gamma3   = '(near(x[0] + x[1], 0.2 - ' + B + ') && abs(x[0] - x[1]) <= ' + A + ') '
# walls    = gamma0 + '||' + gamma1 + '||' + gamma2 + '||' + gamma3

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

# walls = '(near(abs(x[1]), 0.2) && x[0] <= 0) || ' + \
# '(near(x[0] - abs(x[1]), 0.2) && x[0] >= 0.2) || ' + \
# '(near(x[0] - abs(x[1]), - 0.2) && x[0] >= 0.0) ||' + \
# '(near(abs(x[1] - x[0] - 2), '+f"{A:.17f}"+') && x[0] + x[1] >= 0.2 - '+f"{B:.17f}"+') || ' + \
# '(near(x[0] + x[1], 0.2 - '+f"{B:.17f}"+') && abs(x[0] - x[1] - 2) <= '+f"{A:.17f}"+') '

# Define inflow profile
def inflow_g(t):
    return '%.7f' % 1

t = 0

# def inflow_exp(t):
#     return Expression(('x[1]', '0'), degree = 2)
# inflow_str = Expression(inflow_g(t)+'*'+inflow_f,degree=2)

# inflow_str = ('4.0*1.5*(x[1] + 0.2)*(0.2 - x[1]) / pow(0.4, 2)', '0')

# inflow_str = '4.0*1.5*(x[1] + 0.2)*(0.2 - x[1]) / pow(0.4, 2)'
# inflow_exp = Expression((inflow_str, '0')
from scipy.interpolate import interp1d

xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1])
yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17])
xxx=np.linspace(0,1,100)
f = interp1d(xp, yp,kind='cubic')
# plt.plot(xxx,f(xxx))

class INFLOW(UserExpression):
    def set_values(self,fval):
        self.fval = fval
    def eval(self, value, x):
        "Set value[0] to value at point x"
        tol = 1E-13
        if x[1]-0.2 < - tol and x[1] + 0.2 > tol:
            value[0] = self.fval/(x[1] + 0.2)/(0.2 - x[1]) * pow(0.4, 2) *np.exp(-0.5*(np.log((0.2+x[1])/(0.2-x[1]))-0.1)**2)
        else:
            value[0] = 0
        value[1] = 0

    def value_shape(self):
        return (2,)

#### Define boundary conditions
inflow_expr = INFLOW()
t = 0
fval=f(t)
print(type(fval))
inflow_expr.set_values(fval)
bcu_inflow = DirichletBC(V, inflow_expr, inflow)

bcu_walls = DirichletBC(V, Constant((0, 0)), walls)

# bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow1 = DirichletBC(Q, Constant((p_windkessel_1)), outflow1)
bcp_outflow2 = DirichletBC(Q, Constant((p_windkessel_2)), outflow2)
# bcp_outflow = DirichletBC(Q, Constant((0)), outflow)
bcu = [bcu_inflow, bcu_walls]
# bcp = [bcp_outflow]
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

# Create progress bar
#progress = dolfin.cpp.log.Progress('Time-stepping')
#set_log_level(PROGRESS)
pbar = tqdm(total=T)

nn=10
xx1=np.linspace(1.9,2.1,nn)
yy=np.linspace(-2.1,-1.9,nn)
xy1=zip(xx1,yy)
# u_at_bd_1 = [i for i in map(u_,[np.array(i) for i in xy1])]
# u_normal_1 = map(lambda x:(x[0]-x[1])/np.sqrt(2),u_at_bd_1)
# u_avg_1 = sum(u_normal)*0.2*2**0.5/nn

xx2=np.linspace(1.9,2.1,nn)
# yy=np.linspace(2.1,1.9,nn)
xy2=zip(xx2,yy)
# u_at_bd_2 = [i for i in map(u_,[np.array(i) for i in xy2])]
# u_normal_2 = map(lambda x:(x[0]+x[1])/np.sqrt(2),u_at_bd_2)
# u_avg_2 = sum(u_normal)*0.2*2**0.5/nn

# Time-stepping
files = []
t = 0


for n in range(num_steps):
    if flag_dynamic == True:
        u_at_bd_1 = [i for i in map(u_,[np.array(i) for i in xy1])]
        u_normal_1 = map(lambda x:(x[0]-x[1])/np.sqrt(2),u_at_bd_1)
        u_avg_1 = sum(u_normal_1)*0.2*2**0.5/nn

        u_at_bd_2 = [i for i in map(u_,[np.array(i) for i in xy2])]
        u_normal_2 = map(lambda x:(x[0]+x[1])/np.sqrt(2),u_at_bd_2)
        u_avg_2 = sum(u_normal_2)*0.2*2**0.5/nn

        p_windkessel_1 += dt/c*(p_windkessel_1/R+u_avg_1)
        p_windkessel_2 += dt/c*(p_windkessel_2/R+u_avg_1)
        bcp_outflow1 = DirichletBC(Q, Constant((p_windkessel_1)), outflow1)
        bcp_outflow2 = DirichletBC(Q, Constant((p_windkessel_1)), outflow2)
        bcp = [bcp_outflow1,bcp_outflow2]

        inflow_expr.set_values(t)
        bcu_inflow = DirichletBC(V, inflow_expr, inflow)
        bcu = [bcu_inflow, bcu_walls]

    # Update current time
    t += dt

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

    if flag_movie:
        # # Plot solution
        plt.cla()
        plot(p_)
        plot(u_, title = "Pressure and Velocity,t = %.4f" % t)
        fname = '_tmp%03d.png' % n
        # print('Saving frame', fname)
        plt.savefig(fname)
        files.append(fname)
    

    # Update progress bar
#    progress.update(t / T)
    # if n % 20 == 0:
    pbar.update(dt)
    pbar.set_description('u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])
    
pbar.close()

if flag_movie:
    print('Making animation - this may take a while')
    # subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
    #                 "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)
    subprocess.call("ffmpeg -i _tmp%03d.png -pix_fmt yuv420p ../output/output.mp4", shell=True)

    # cleanup
    for fname in files:
        os.remove(fname)
else:
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.8,0.05])
    gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    
    ax1 = plt.subplot(gs[0,0])
    plt1 = plot(p_)
    plt2 = plot(u_, title = "Pressure and Velocity,t = %.4f" % t)
    cbax = plt.subplot(gs[0,1])
    cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')
    plt.show()

