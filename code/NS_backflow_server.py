# from mesh_server import *
from BC_backflow_server import *
from tqdm import tqdm # status bar
import os
import subprocess
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import dolfin 

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

def compute_NSsolution_mini(mesh,    
    T = .1                  ,
    num_steps = 100         ,
    mu = 0.03               ,
    rho = 1                 ,
    # # windkessel,
    # c = 1                   ,
    # Rd = 1e5                ,
    # Rp = 5e4                ,
    # p_windkessel_1 = 1.06e5 ,
    # p_windkessel_2 = 1.06e5 ,
    # u0=20.                  ,
    c = 1                   ,
    Rd = 1                ,
    Rp = 1                ,
    p_windkessel_1 = 1 ,
    p_windkessel_2 = 1 ,
    u0=1.                  ,
    flag_movie = False,
    with_teman = False,
    with_bf_est = False,
    freq_plot = 1):
    dt = T / num_steps
    
    ### Define function spaces
    Mini = compute_mini(mesh)
    # Define variational problem
    w0 = Function(Mini)
    u0 = Function(Mini.sub(0).collapse())
    # u0 = project(inflow_expr, Mini.sub(0).collapse())
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

    ## Define Subdomain marker
    boundary_markers = MeshFunction('size_t', mesh, dim=2)
    class outflow_boundary(SubDomain):
        # tol = 1E-14
        def inside(self, x, on_boundary):
            return outflow(x,on_boundary)
    ob = outflow_boundary()
    ob.mark(boundary_markers, 2)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)


    if with_teman:
        a = a + 0.5*rho*(div(u0)*inner(u, v))*dx 
    if with_bf_est:
        a = a - 0.5*rho*abs_n(dot(u0, n))*inner(u, v)*ds(2)

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

    tol=.000001

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
    num_plot = 0
    inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree=2)
    t = 0.
    for n in range(num_steps):
        # Update current time
        t += dt

        ## Windkessel
        u_avg_1 = flux(u0,p1_healthy,p2_healthy,theta_healthy,diam_healthy_vessel)
        u_avg_2 = flux(u0,p1_steno,p2_steno,-theta_steno,diam_steno_vessel)
        delta1 = dt/c*(-p_windkessel_1/Rd+u_avg_1)
        # delta_windkessel1.append(delta1)
        p_windkessel_1 += delta1
        delta2 = dt/c*(-p_windkessel_2/Rd+u_avg_2)
        p_windkessel_2 += delta2
        # delta_windkessel2.append(delta2)

        p_bdry_1 = Rp * u_avg_1 + p_windkessel_1
        p_bdry_2 = Rp * u_avg_2 + p_windkessel_2

        # bcu_inflow = DirichletBC(Mini.sub(0), project(inflow_expr, Mini.sub(0).collapse()), inflow_domain)

        # collect boundary conditions and assemble system
        # bcs = [bcu_walls, bcu_inflow] # TODO: Add Windkessel
        # bcs = compute_bc(Mini,t,p_bdry_1 = 1, p_bdry_2 = 1)
        # bcs = compute_bc(Mini,t,p_bdry_1 = 0, p_bdry_2 = 0)
        bcs = compute_bc(Mini,t,
            p_bdry_1 =p_bdry_1,
            p_bdry_2 =p_bdry_2,
            u0=u0,
            inflow_expr=inflow_expr,
            inflow_domain=inflow_domain,
            heartfun=heartfun,)
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

        # diagnosis on p
        # p_int_inflow            .append(integrate_over_line(p0 ,p1_inflow    ,p2_inflow      ,artery.diam_trunk))
        # p_int_before_stenosis   .append(integrate_over_line(p0 ,p1_steno_before  ,p2_steno_before    ,artery.diam_steno_vessel))
        # p_int_after_stenosis    .append(integrate_over_line(p0 ,p1_steno_after   ,p2_steno_after     ,artery.diam_steno_vessel))
        p_int_healthy           .append(integrate_over_line(p0 ,p1_healthy       ,p2_healthy         ,artery.diam_healthy_vessel))


        # Update progress bar
        pbar.update(dt)
        pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u0.vector().vec().max()[1] + 'p_max:%.2f ' % p0.vector().vec().max()[1])

        if flag_movie:
            if n % freq_plot == 0:
                fname = './tmp/_tmp%05d.png' % num_plot
                plot_solution(u0,p0,fname)
                files.append(fname)
                num_plot +=1

    pbar.close()
    return u0,p0,files

def cleanup(files):
    for fname in files:
        os.remove(fname)

def movie():
    # print('Making animation - this may take a while')
    subprocess.call("ffmpeg -i ./tmp/_tmp%05d.png -r 60 -pix_fmt yuv420p ./output/output.mp4", shell=True)


if __name__ == '__main__':
    mesh_precision = 40
    u0 = 1.                 # init amplitude
    s = .5                  # init asymmetry
    artery = Artery(diam_steno_vessel=0.1, 
        diam_narrow=0., 
        theta_steno=np.pi/6, 
        diam_healthy_vessel=0.1, 
        theta_healthy=np.pi/6,
        length0 = .5,
        length = .3)
    mesh = artery.mesh(mesh_precision)
    Mini = compute_mini(mesh)
    u0,p0,files = compute_NSsolution_mini(mesh,
    T = 1                  ,
    num_steps = 1200         ,
    mu = 0.03               ,
    rho = 1                 ,
    # # windkessel,
    c = 1                   ,
    Rd = 1                ,
    Rp = 1                ,
    p_windkessel_1 = 1 ,
    p_windkessel_2 = 1 ,
    u0=1.                  ,
    flag_movie = True,
    with_teman = False,
    with_bf_est = True,
    freq_plot = 1)
    print('NS computation test passed.')
    if flag_movie:
        movie()
        # cleanup
        cleanup(files)
