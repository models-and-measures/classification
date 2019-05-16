from mesh_server import *
from boundarycondition_server import *
from tqdm import tqdm # status bar
import os
import subprocess
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

# global variables, beware of namespace collision
T = 1                   # final time
num_steps = 2000        # number of time steps # must satisfy CFL condition
dt = T / num_steps
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

#### Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

#### Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# post computation: define diagnosis section
x0=1.
x1=1.9
x2=1.9
x3=1.

nn=10 # number of monte carlo samples
tol=1e-6 # tolerance from boundary

# diagnosis on the trunk
xx0=-x0*np.ones(nn)
yy0=np.linspace(-D0 +tol,D0 -tol,nn)
# diagnosis on branch 1
xx1=np.linspace(x1-d +tol,x1+d -tol,nn)
yy1=np.linspace(x1+d -tol,x1-d +tol,nn)
# diagnosis on branch 2
xx2=np.linspace(x2-d +tol,x2+d -tol,nn)
yy2=np.linspace(-x2-d+tol,-x2+d-tol,nn)
# diagnosis on illness part
xx3=np.linspace(x2-d +tol,x2+d -b-tol,nn)
yy3=np.linspace(-y2-d+tol,-x2+d-b-tol,nn)

def slice(xx,yy):
    "In: x-y ranges. Out: grid, list of np array" #avoid zip variable which can be used only once
    return [np.array(i) for i in zip(xx,yy)]

def average_over_line(u_or_p,grid):
    "In: grid from function above. Out: mean value of the given function (1d or 2d)"
    return np.mean([u_or_p(i) for i in grid],axis=0)

xy0=slice(xx0,yy0)
xy1=slice(xx1,yy1)
xy2=slice(xx2,yy2)
xy3=slice(xx3,yy3)

section_len = 2*B

def compute_NSsolution(mesh, V, Q,
            T = T,
            num_steps = num_steps,
            mu = mu,
            rho = rho,
            c = c,
            Rd = Rd,
            Rp = Rp,
            p_windkessel_1 =p_windkessel_1,
            p_windkessel_2 =p_windkessel_2,
            u0=u0,
            s=s,
            inflow_expr=inflow_expr,
            inflow_str=inflow_str,
            heartfun=heartfun,
            xy0=xy0,
            xy1=xy1,
            xy2=xy2,
            xy3=xy3,
    ):
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
        xy0=slice(x0)
        xy1=slice(x1)
        xy2=slice(x2)
        xy3=slice(x3)
        if flag_dynamic == True:
            # u_at_bd_1 = [u_(i) for i in xy1]
            # u_normal_1 = [np.dot(u, np.array([1., 1.])/np.sqrt(2)) for u in u_at_bd_1]
            u_avg_1 = np.dot(average_over_line(u_,xy1),[1,1])*section_len

            # u_at_bd_2 = [u_(np.array(i)) for i in xy2]
            # u_normal_2 = [np.dot(u, np.array([1., -1.])/np.sqrt(2)) for u in u_at_bd_2]
            # u_avg_2 = sum(u_normal_2)/nn*0.2*np.sqrt(2)
            u_avg_2 = np.dot(average_over_line(u_,xy2),[1,1])*section_len

            p_windkessel_1 += dt/c*(-p_windkessel_1/Rd+u_avg_1)
            p_windkessel_2 += dt/c*(-p_windkessel_2/Rd+u_avg_2)

            p_bdry_1 = p_windkessel_1 + Rp * u_avg_1
            p_bdry_2 = p_windkessel_2 + Rp * u_avg_2

            bcu,bcp = compute_bc(V,Q,t,
                                p_bdry_1 =p_bdry_1,
                                p_bdry_2 =p_bdry_2,
                                u0=u0,
                                s=s,
                                inflow_expr=inflow_expr,
                                inflow_str=inflow_str,
                                heartfun=heartfun,)


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

        # Update previous solution
        p_n.assign(p_)
        u_n.assign(u_)

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
            # plt.figure()
            # plt.cla()
            # ax1 = plt.subplot(gs[0,0])
            # plt1 = plot(p_)
            # plt2 = plot(u_, title = "Pressure and Velocity,t = %.4f" % t)
            # cbax = plt.subplot(gs[0,1])
            # cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')


            # plt.show()
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


if __name__ == '__main__':
    mesh, V, Q = compute_mesh()
    # bcu, bcp = compute_bc(V,Q,0.)
    u_, p_ = compute_NSsolution(mesh, V, Q)

    xy0 = slice(x0)
    xy1 = slice(x1)
    xy2 = slice(x2)
    xy3 = slice(x3)
    average_over_line(p_,xy0)
    average_over_line(p_,xy1)
    average_over_line(p_,xy2)
    average_over_line(p_,xy3)
    print("test completed.")