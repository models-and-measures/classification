# from mesh_server import *
from BC_new_server import *
from tqdm import tqdm # status bar
import os
import subprocess
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

def linspace_2d(p1,p2,nn=6):
    xx = np.linspace(p1[0],p2[0],nn)
    yy = np.linspace(p1[1],p2[1],nn)
    return [i for i in zip(xx,yy)]

def eval_fun(u,grid):
    "return a list of evaluation of function u on grid"
    return [u(i) for i in grid]

def flux(u,p1,p2,theta,arclength,nn=6):
    grid = linspace_2d(p1,p2,nn)
    funval = eval_fun(u,grid)
    # print(funval)
    n0 = np.cos(theta)
    n1 = np.sin(theta)
    direction = np.array([n0,n1])
    u_normal = [np.dot(i, direction) for i in funval]
    out = sum(u_normal)/nn*arclength
    # print(out)
    # print()
    return out

def integrate_over_line(p,p1,p2,arclength,nn=6):
    grid = linspace_2d(p1,p2,nn)
    funval = eval_fun(p,grid)
    out = sum(funval)/nn*arclength

def plot_solution(u_,p_,fname = "solution.pdf",vmin=5000, vmax=12000):
    import matplotlib
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.8,0.05])
    gs.update(left=0.05, right=0.9, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    ax1 = plt.subplot(gs[0,0])
    ax1.set_aspect('equal')
    import matplotlib
    plt1 = plot(p_,norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False))
    plt2 = plot(u_,title = 'Velocity')#,norm=matplotlib.colors.Normalize(vmin=0, vmax=10000, clip=False))
    cbax = plt.subplot(gs[0,1])
    cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')
    plt.savefig(fname)

def compute_NSsolution(mesh,    
    T = .1                  ,
    num_steps = 100         ,
    mu = 0.03               ,
    rho = 1                 ,
    # windkessel,
    c = 1                   ,
    Rd = 1e5                ,
    Rp = 5e4                ,
    p_windkessel_1 = 1.06e5 ,
    p_windkessel_2 = 1.06e5 ,
    u0=20.                  ,
    flag_movie = False):
    dt = T / num_steps
    
    ### Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
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

    tol=.001

    # def find_endpoint(x,y,Y,theta,tol_narrow = tol, tol_shift = tol):
    # # find endpoints of diagnosis line
    #     p1 = np.array([x-tol_shift,y + tol_narrow])
    #     p1 = rotate(theta,out)
    #     p1 = np.array([x-tol_shift,Y - tol_narrow])
    #     p1 = rotate(theta,out)
    #     return out

    # Points:
    ### Flux surface for u
    p1_healthy = np.array([length*2-tol,tol])
    p1_healthy = rotate(theta_healthy,p1_healthy)
    p2_healthy = np.array([length*2-tol,diam_healthy_vessel - tol])
    p2_healthy = rotate(theta_healthy,p2_healthy)
    p1_steno = np.array([length*2-tol,-tol])
    p1_steno = rotate(-theta_steno,p1_steno)
    p2_steno = np.array([length*2-tol,-diam_steno_vessel + tol])
    p2_steno = rotate(-theta_steno,p2_steno)

    ### Diagnosis surface for p
    p1_steno_before = np.array([length*2-tol,-tol])
    p1_steno_before = rotate(-theta_steno,p1_steno)
    p2_steno_before = np.array([length*2-tol,-diam_steno_vessel + tol])
    p2_steno_before = rotate(-theta_steno,p2_steno)
    p1_steno_after = np.array([length*2-tol,-tol])
    p1_steno_after = rotate(-theta_steno,p1_steno)
    p2_steno_after = np.array([length*2-tol,-diam_steno_vessel + tol])
    p2_steno_after = rotate(-theta_steno,p2_steno)

    ### status bar
    pbar = tqdm(total=T)

    ### Time-stepping
    t = 0.0
    files = []
    flux_healthy = []
    flux_stenosis = []
    p_int_inflow = []
    p_int_before_stenosis = []
    p_int_after_stenosis = []
    p_int_healthy = []
    for n in range(num_steps):
        # Update current time
        t += dt
        u_avg_1 = flux(u_,p1_healthy,p2_healthy,theta_healthy,diam_healthy_vessel)
        u_avg_2 = flux(u_,p1_steno,p2_steno,-theta_steno,diam_steno_vessel)
        delta1 = dt/c*(-p_windkessel_1/Rd+u_avg_1)
        # delta_windkessel1.append(delta1)
        p_windkessel_1 += delta1
        delta2 = dt/c*(-p_windkessel_2/Rd+u_avg_2)
        p_windkessel_2 += delta2
        # delta_windkessel2.append(delta2)

        p_bdry_1 = Rp * u_avg_1 + p_windkessel_1
        p_bdry_2 = Rp * u_avg_2 + p_windkessel_2
        bcu, bcp = compute_bc(V, Q, t, p_bdry_1 = p_bdry_1, p_bdry_2 = p_bdry_2)

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

        # Update previous solution
        p_n.assign(p_)
        u_n.assign(u_)

        # diagnosis on p
        # # p_int_inflow            .append(integrate_over_line(p_,p1_int_inflow    ,p2_int_inflow      ,diam_trunk))
        # p_int_before_stenosis   .append(integrate_over_line(p_,p1_steno_before  ,p2_steno_before    ,diam_steno_vessel))
        # p_int_after_stenosis    .append(integrate_over_line(p_,p1_steno_after   ,p2_steno_after     ,diam_steno_vessel))
        # p_int_healthy           .append(integrate_over_line(p_,p1_healthy       ,p2_healthy         ,diam_healthy_vessel))

        # Update progress bar
        pbar.update(dt)
        pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])

        if flag_movie:
            fname = './tmp/_tmp%05d.png' % n
            plot_solution(u_,p_,fname)
            files.append(fname)

    pbar.close()
    return u_,p_,files

def cleanup(files):
    for fname in files:
        os.remove(fname)

def movie():
    # print('Making animation - this may take a while')
    subprocess.call("ffmpeg -i ./tmp/_tmp%05d.png -r 60 -pix_fmt yuv420p ../output/output.mp4", shell=True)


if __name__ == '__main__':
    mesh_precision = 40
    flag_movie = True
    mesh = Artery().mesh(mesh_precision)
    # bcu, bcp = compute_bc(V,Q,0.)
    # u_,p_,files = compute_NSsolution(mesh,T=1,num_steps = 1000,
    # mu = 0.03               ,
    # rho = 1                 ,
    # # windkessel,
    # c = 1                   ,
    # Rd = 1                ,
    # Rp = 1                ,
    # p_windkessel_1 = 1 ,
    # p_windkessel_2 = 1 ,
    # u0=1                  ,
    # flag_movie = flag_movie)
    u_,p_,files = compute_NSsolution(mesh,
    T = .3                  ,
    num_steps = 300         ,
    mu = 0.03               ,
    rho = 1                 ,
    # windkessel,
    c = 1.6e-5              ,
    Rd = 60012              ,
    Rp = 7501.5             ,
    p_windkessel_1 = 1.06e5 ,
    p_windkessel_2 = 1.06e5 ,
    u0=20.                  ,
    flag_movie = flag_movie)
    print('NS computation test passed.')
    if flag_movie:
        movie()
        # cleanup
        cleanup(files)
