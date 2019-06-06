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
    "several points on line segments"
    xx = np.linspace(p1[0],p2[0],nn)
    yy = np.linspace(p1[1],p2[1],nn)
    return [i for i in zip(xx,yy)]

def eval_fun(u,grid):
    "return a list of evaluation of function u on grid"
    return [u(i) for i in grid]

def find_endpoint(x,y,Y,theta,tol_narrow = .0001, tol_shift = .0001):
    "find endpoints of diagnosis line. Out: rotation(theta,(x,y)) and rotation(theta,(x,Y)) "
    p1 = np.array([x-tol_shift,y + tol_narrow])
    p1 = rotate(theta,p1)
    p2 = np.array([x-tol_shift,Y - tol_narrow])
    p2 = rotate(theta,p2)
    return p1,p2

def flux(u,p1,p2,theta,arclength,nn=6):
    "output flux, necessary for Windkessel model."
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
    "line intergral of p between two points."
    grid = linspace_2d(p1,p2,nn)
    funval = eval_fun(p,grid)
    out = sum(funval)/nn*arclength

def plot_solution(u_,p_,fname = "solution.pdf",title = 'Velocity',vmin=0, vmax=150):
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

def plot_diagnosis(T,num_steps,diagnoses,fname,pmin=0, pmax=60, head_off = 3):
    tt = np.linspace(0,T,num_steps-head_off)
    plt.figure()
    for p in diagnoses:
        for i in range(head_off):
            p.pop(i)
        plt.plot(tt,p)
    # print(integrate_over_line(u_,p1_healthy,p2_healthy,1))
    plt.ylim(pmin,pmax)
    plt.savefig(fname)

def compute_mixedspace_IPCS(mesh):
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    return V,Q

def compute_NSsolution(mesh,    
    T,
    num_steps,
    mu            ,
    rho       ,
    # windkessel,
    c          ,
    Rd       ,
    Rp        ,
    p_windkessel_1  ,
    p_windkessel_2  ,
    u0 ,
    s,
    uname,
    pname,
    flag_movie = False,
    flag_diagnosis = True):
    "IPCS Scheme"
    dt = T / num_steps
    
    ### Define function spaces
    V, Q = compute_mixedspace_IPCS(mesh)

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

    inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree=2)


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

    # Points:
    ### Flux surface for u
    p1_healthy = np.array([length*2-tol,tol])
    p1_healthy = rotate(-theta_healthy,p1_healthy)
    p2_healthy = np.array([length*2-tol,diam_healthy_vessel - tol])
    p2_healthy = rotate(-theta_healthy,p2_healthy)
    p1_steno = np.array([length*2-tol,-tol])
    p1_steno = rotate(theta_steno,p1_steno)
    p2_steno = np.array([length*2-tol,-diam_steno_vessel + tol])
    p2_steno = rotate(theta_steno,p2_steno)

    # ### Diagnosis surface for p
    # p1_inflow = np.array([-length0+tol,diam_healthy_vessel * np.cos(theta_healthy)-tol])
    # p2_inflow = np.array([-length0+tol,-diam_steno_vessel * np.cos(theta_steno)+tol])
    # p1_healthy_mid = np.array([length+1.1*length_steno-tol,tol])
    # p1_healthy_mid = rotate(-theta_healthy,p1_healthy_mid)
    # p2_healthy_mid = np.array([length+1.1*length_steno-tol,diam_healthy_vessel - tol])
    # p2_healthy_mid = rotate(-theta_healthy,p2_healthy_mid)
    # p1_steno_before = np.array([length-1.1*length_steno-tol,-tol])
    # p1_steno_before = rotate(theta_steno,p1_steno_before)
    # p2_steno_before = np.array([length-1.1*length_steno-tol,-diam_steno_vessel + tol])
    # p2_steno_before = rotate(theta_steno,p2_steno_before)
    # p1_steno_after = np.array([length+1.1*length_steno-tol,-tol])
    # p1_steno_after = rotate(theta_steno,p1_steno_after)
    # p2_steno_after = np.array([length+1.1*length_steno-tol,-diam_steno_vessel + tol])
    # p2_steno_after = rotate(theta_steno,p2_steno_after)

    ### status bar
    pbar = tqdm(total=T)

    # Create time series (for use in reaction_system.py)
    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)

    ### Time-stepping
    t = 0.0
    files = []
    flux_healthy = []
    flux_stenosis = []
    # p_int_inflow = []
    # p_int_before_stenosis = []
    # p_int_after_stenosis = []
    # p_int_healthy = []
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
        # p_bdry_1 = 10
        # p_bdry_2 = 10
        bcu, bcp = compute_bc(V,Q,t,p_bdry_1,p_bdry_2,u0,s,inflow_expr,inflow_domain,heartfun,)

        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcp]
        [bc.apply(A3) for bc in bcu]

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        # [bc.apply(b1) for bc in bcp]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        [bc.apply(b3) for bc in bcu]
        # [bc.apply(b3) for bc in bcp]
        solve(A3, u_.vector(), b3, 'cg', 'sor')

        # Save nodal values to file
        timeseries_u.store(u_.vector(), t)
        timeseries_p.store(p_.vector(), t)

        # Update previous solution
        p_n.assign(p_)
        u_n.assign(u_)

        # # diagnosis on p
        # p_int_inflow            .append(p_(p1_inflow        ))#integrate_over_line(p_ ,p1_inflow        ,p2_inflow      ,artery.diam_trunk))
        # p_int_before_stenosis   .append(p_(p1_steno_before  ))#integrate_over_line(p_ ,p1_steno_before  ,p2_steno_before    ,artery.diam_steno_vessel))
        # p_int_after_stenosis    .append(p_(p1_steno_after   ))#integrate_over_line(p_ ,p1_steno_after   ,p2_steno_after     ,artery.diam_steno_vessel))
        # p_int_healthy           .append(p_(p1_healthy_mid   ))#integrate_over_line(p_ ,p1_healthy       ,p2_healthy         ,artery.diam_healthy_vessel))

        # Update progress bar
        pbar.update(dt)
        pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])

        # if flag_movie:
        #     fname = './tmp/_tmp%05d.png' % n
        #     plot_solution(u_,p_,fname)
        #     files.append(fname)

    pbar.close()

    # if flag_diagnosis:
    #     # p_int_inflow,p_int_before_stenosis,p_int_after_stenosis,p_int_healthy
    #     diagnoses = [p_int_inflow,p_int_before_stenosis,p_int_after_stenosis,p_int_healthy]
    #     plot_diagnosis(T,num_steps,diagnoses,"diagnoses/diagnoses.pdf",pmin=-20,pmax=120)
    #     print("pressure plotted.")

    return u_,p_,files

def movie(fname = "./output/output.mp4"):
    # print('Making animation - this may take a while')
    subprocess.call("ffmpeg -i ./tmp/_tmp%05d.png -r 60 -pix_fmt yuv420p "+fname, shell=True)

def make_movie_IPCS(mesh, 
                    T,
                    num_steps,
                    uname = 'NSdata/u_series',
                    pname = 'NSdata/p_series',
                    writename = './output/output.mp4',
                    tol = 0.001):
    # Points:
    ### Flux surface for u
    # p1_healthy, p2_healthy = find_endpoint(length*2,0,diam_healthy_vessel,-theta_healthy)
    # p2_steno, p1_steno = find_endpoint(length*2,-diam_steno_vessel,0,theta_steno)
    ## alternatively,
    # p1_healthy = np.array([length*2-tol,tol])
    # p1_healthy = rotate(-theta_healthy,p1_healthy)
    # p2_healthy = np.array([length*2-tol,diam_healthy_vessel - tol])
    # p2_healthy = rotate(-theta_healthy,p2_healthy)
    # p1_steno = np.array([length*2-tol,-tol])
    # p1_steno = rotate(theta_steno,p1_steno)
    # p2_steno = np.array([length*2-tol,-diam_steno_vessel + tol])
    # p2_steno = rotate(theta_steno,p2_steno)

    ### Diagnosis surface for p
    p1_inflow = np.array([-length0+tol,diam_healthy_vessel * np.cos(theta_healthy)-tol])
    p2_inflow = np.array([-length0+tol,-diam_steno_vessel * np.cos(theta_steno)+tol])
    p1_healthy_mid = np.array([length-tol,tol])
    p1_healthy_mid = rotate(-theta_healthy,p1_healthy_mid)
    p2_healthy_mid = np.array([length-tol,diam_healthy_vessel - tol])
    p2_healthy_mid = rotate(-theta_healthy,p2_healthy_mid)
    p1_steno_before = np.array([length-1.1*length_steno-tol,-tol])
    p1_steno_before = rotate(theta_steno,p1_steno_before)
    p2_steno_before = np.array([length-1.1*length_steno-tol,-diam_steno_vessel + tol])
    p2_steno_before = rotate(theta_steno,p2_steno_before)
    p1_steno_after = np.array([length+1.1*length_steno-tol,-tol])
    p1_steno_after = rotate(theta_steno,p1_steno_after)
    p2_steno_after = np.array([length+1.1*length_steno-tol,-diam_steno_vessel + tol])
    p2_steno_after = rotate(theta_steno,p2_steno_after)

    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)
    V,Q = compute_mixedspace_IPCS(mesh)
    u = Function(V)
    p = Function(Q)
    t = 0
    dt = T / num_steps
    ### status bar
    pbar = tqdm(total=T)
    p_int_inflow = []
    p_int_before_stenosis = []
    p_int_after_stenosis = []
    p_int_healthy = []
    for n in range(num_steps):
        # Update current time
        t += dt
        # Read velocity from file
        timeseries_u.retrieve(u.vector(), t)
        timeseries_p.retrieve(p.vector(), t)
        # diagnosis on p
        p_int_inflow            .append(p(p1_inflow        ))#integrate_over_line(p_ ,p1_inflow        ,p2_inflow      ,artery.diam_trunk))
        p_int_before_stenosis   .append(p(p1_steno_before  ))#integrate_over_line(p_ ,p1_steno_before  ,p2_steno_before    ,artery.diam_steno_vessel))
        p_int_after_stenosis    .append(p(p1_steno_after   ))#integrate_over_line(p_ ,p1_steno_after   ,p2_steno_after     ,artery.diam_steno_vessel))
        p_int_healthy           .append(p(p1_healthy_mid   ))#integrate_over_line(p_ ,p1_healthy       ,p2_healthy         ,artery.diam_healthy_vessel))

        fname = './tmp/_tmp%05d.png' % n
        plot_solution(u,p,fname)
        pbar.update(dt)
        pbar.set_description("Plotting... t = %.4f" % t)

    pbar.close()
    movie(writename)

def compute_NSsolution_mini(mesh,    
    T             ,
    num_steps     ,
    mu            ,
    rho           ,
    # # windkessel,
    # c = 1                   ,
    # Rd = 1e5                ,
    # Rp = 5e4                ,
    # p_windkessel_1 = 1.06e5 ,
    # p_windkessel_2 = 1.06e5 ,
    # u0=20.                  ,
    c               ,
    Rd              ,
    Rp             ,
    p_windkessel_1,
    p_windkessel_2,
    u0               ,
    flag_movie = False,
    with_teman = False,
    with_bf_est = False,
    freq_plot = 1):
    dt = T / num_steps
    
    ### Define function spaces
    Mini = compute_mini(mesh)
    # Define variational problem
    # w_ = Function(Mini)
    u_ = Function(Mini.sub(0).collapse())
    # u_ = project(inflow_expr, Mini.sub(0).collapse())
    p_ = Function(Mini.sub(1).collapse())

    (u, p) = TrialFunctions(Mini)
    (v, q) = TestFunctions(Mini)

    # Variatonal form a = L
    def abs_n(x):
        return 0.5*(x - abs(x))

    n = FacetNormal(mesh)

    a   = rho/dt*inner(u,  v)*dx \
        + mu*inner(grad(u), grad(v))*dx + q*div(u)*dx \
        - div(v)*p*dx \
        + rho*inner(grad(u)*u_, v)*dx \

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
        a = a + 0.5*rho*(div(u_)*inner(u, v))*dx 
    if with_bf_est:
        a = a - 0.5*rho*abs_n(dot(u_, n))*inner(u, v)*ds(2)

    L   = rho/dt*inner(u_, v)*dx

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

    inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree=2)

    # Points:
    ### Flux surface for u
    p1_healthy, p2_healthy = find_endpoint(length*2,0,diam_healthy_vessel,-theta_healthy)
    p2_steno, p1_steno = find_endpoint(length*2,-diam_steno_vessel,0,theta_steno)
    ## alternatively,
    # p1_healthy = np.array([length*2-tol,tol])
    # p1_healthy = rotate(-theta_healthy,p1_healthy)
    # p2_healthy = np.array([length*2-tol,diam_healthy_vessel - tol])
    # p2_healthy = rotate(-theta_healthy,p2_healthy)
    # p1_steno = np.array([length*2-tol,-tol])
    # p1_steno = rotate(theta_steno,p1_steno)
    # p2_steno = np.array([length*2-tol,-diam_steno_vessel + tol])
    # p2_steno = rotate(theta_steno,p2_steno)

    # ### Diagnosis 
    # p1_inflow = np.array([-length0+tol,diam_healthy_vessel * np.cos(theta_healthy)-tol])
    # p2_inflow = np.array([-length0+tol,-diam_steno_vessel * np.cos(theta_steno)+tol])
    # p1_healthy_mid = np.array([length-tol,tol])
    # p1_healthy_mid = rotate(-theta_healthy,p1_healthy_mid)
    # p2_healthy_mid = np.array([length-tol,diam_healthy_vessel - tol])
    # p2_healthy_mid = rotate(-theta_healthy,p2_healthy_mid)
    # p1_steno_before = np.array([length-1.1*length_steno-tol,-tol])
    # p1_steno_before = rotate(theta_steno,p1_steno_before)
    # p2_steno_before = np.array([length-1.1*length_steno-tol,-diam_steno_vessel + tol])
    # p2_steno_before = rotate(theta_steno,p2_steno_before)
    # p1_steno_after = np.array([length+1.1*length_steno-tol,-tol])
    # p1_steno_after = rotate(theta_steno,p1_steno_after)
    # p2_steno_after = np.array([length+1.1*length_steno-tol,-diam_steno_vessel + tol])
    # p2_steno_after = rotate(theta_steno,p2_steno_after)

    ### status bar
    pbar = tqdm(total=T)

    ### Time-stepping
    t = 0.0
    files = []
    flux_healthy = []
    flux_stenosis = []
    # p_int_inflow = []
    # p_int_before_stenosis = []
    # p_int_after_stenosis = []
    # p_int_healthy = []
    num_plot = 0
    for n in range(num_steps):
        # Update current time
        t += dt

        ## Windkessel
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

        # p_bdry_1 = 0
        # p_bdry_2 = 0

        # bcu_inflow = DirichletBC(Mini.sub(0), project(inflow_expr, Mini.sub(0).collapse()), inflow_domain)

        # collect boundary conditions and assemble system
        # bcs = [bcu_walls, bcu_inflow] # TODO: Add Windkessel
        # bcs = compute_bc(Mini,t,p_bdry_1 = 1, p_bdry_2 = 1)
        # bcs = compute_bc(Mini,t,p_bdry_1 = 0, p_bdry_2 = 0)
        bcs = compute_bc_mini(Mini,t,
            p_bdry_1 = p_bdry_1,
            p_bdry_2 = p_bdry_2,
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
        u_.assign(ut)
        p_.assign(pt)

        # # diagnosis on p
        # p_int_inflow            .append(p_(p1_inflow        ))#integrate_over_line(p_ ,p1_inflow        ,p2_inflow      ,artery.diam_trunk))
        # p_int_before_stenosis   .append(p_(p1_steno_before  ))#integrate_over_line(p_ ,p1_steno_before  ,p2_steno_before    ,artery.diam_steno_vessel))
        # p_int_after_stenosis    .append(p_(p1_steno_after   ))#integrate_over_line(p_ ,p1_steno_after   ,p2_steno_after     ,artery.diam_steno_vessel))
        # p_int_healthy           .append(p_(p1_healthy       ))#integrate_over_line(p_ ,p1_healthy       ,p2_healthy         ,artery.diam_healthy_vessel))

        # p_int_inflow            .append(integrate_over_line(p_ ,p1_inflow        ,p2_inflow          ,artery.diam_trunk))
        # p_int_before_stenosis   .append(integrate_over_line(p_ ,p1_steno_before  ,p2_steno_before    ,artery.diam_steno_vessel))
        # p_int_after_stenosis    .append(integrate_over_line(p_ ,p1_steno_after   ,p2_steno_after     ,artery.diam_steno_vessel))
        # p_int_healthy           .append(integrate_over_line(p_ ,p1_healthy       ,p2_healthy         ,artery.diam_healthy_vessel))
        # Update progress bar
        pbar.update(dt)
        pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])

        if flag_movie:
            if n % freq_plot == 0:
                fname = './tmp/_tmp%05d.png' % num_plot
                plot_solution(u_,p_,fname,vmin=0, vmax=60)
                files.append(fname)
                num_plot +=1

    pbar.close()


def cleanup(files):
    for fname in files:
        os.remove(fname)


# def diagnose(p_series,p_int_inflow,p_int_before_stenosis,p_int_after_stenosis,p_int_healthy):
#     diagnoses = [p_int_inflow,p_int_before_stenosis,p_int_after_stenosis,p_int_healthy]
#     plot_diagnosis(T,num_steps,diagnoses,"output/diagnoses.pdf",pmin=-20,pmax=120)


if __name__ == '__main__':
    # mesh
    artery = Artery(diam_steno_vessel, diam_narrow, theta_steno, diam_healthy_vessel, theta_healthy)
    mesh = artery.mesh(mesh_precision)
    # File('NSdata/artery.xml.gz') << mesh
    # mesh = Mesh('NSdata/artery.xml.gz')

    # Solver
    T = .01
    num_steps = 10
    flag_movie = False
    flag_cleanup = True 
    flag_diagnosis = True
    flag_IPCS = True
    with_teman = False
    with_bf_est = False
    freq_plot = 1
    uname = 'NSdata/u_series'
    pname = 'NSdata/p_series'


    if flag_IPCS:
        u,p,files = compute_NSsolution(mesh,
        T=T,
        num_steps=num_steps,
        mu=mu,
        rho=rho,
        c=c,
        Rd=Rd,
        Rp=Rp,
        p_windkessel_1=p_windkessel_1,
        p_windkessel_2=p_windkessel_2,
        u0=u0,
        s = s,
        uname = uname,
        pname = pname,
        flag_movie=flag_movie)
        make_movie_IPCS(mesh, 
                    T,
                    num_steps,
                    uname = 'NSdata/u_series',
                    pname = 'NSdata/p_series',
                    writename = './output/output.mp4')

    else: #mini 
        u,p,files = compute_NSsolution_mini(mesh,
        T = T                  ,
        num_steps = num_steps         ,
        mu = mu               ,
        rho = rho                 ,
        # # windkessel,
        c = c                   ,
        Rd = Rd                ,
        Rp = Rp                ,
        p_windkessel_1 = p_windkessel_1 ,
        p_windkessel_2 = p_windkessel_2 ,
        u0=u0                  ,
        flag_movie = flag_movie,
        with_teman = with_teman,
        with_bf_est = with_bf_est,
        freq_plot = freq_plot)
    print('NS computation completed.')
    
    if flag_movie:
        movie()

    if flag_cleanup:
        cleanup(files)
