import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
# mesh
from mshr import * 
# FEM solver
from fenics import * 
import sys
from scipy.interpolate import interp1d
import argparse

######## Check environment ########
if os.environ.get('DISPLAY','') == '':
    print('currently on server/docker. Using non-interactive Agg backend')
    import matplotlib as mpl
    mpl.use('Agg')
    FLAG_TQDM = False
    FLAG_MOVIE = False
    flag_argparse = True
else:
    FLAG_TQDM = True
    FLAG_MOVIE = True
    try:
        get_ipython().config 
        from tqdm import tqdm_notebook as tqdm
        print('currently on local notebook')
        flag_argparse = False
        # call init() on notebook to set global variables
    except NameError:
        from tqdm import tqdm # status bar
        print('currently on local')
        flag_argparse = True

######## Pass Variables as global variables (default val = 0) ########
if flag_argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help="symmetry", type= float)
    parser.add_argument('-diam_narrow', '-d', type= float, help="diam_narrow")
    args=parser.parse_args()
    # print(args)
    # global s
    # global diam_narrow
    if args.s:  
        pass
        S=args.s
    else:
        S=0
    if args.diam_narrow:
        pass
        D = args.diam_narrow
    else:
        D = 0

s = S
diam_narrow = D
######## Solver Variables ########
T = .46 # Final time
critical_time = 0.46 # to save time, compute up until backflow time
num_steps = 250 # mind CFL condition (TBA: computation)
flag_ts = False # save time series data for movie and diagnosis, necessary for the following two flags
flag_movie = False # generate mp4 movie after computaion
flag_diagnosis = False # record pressure change
flag_cleanup = False # clean up image files after filming
flag_xray = False # posterior xray, a.k.a. summary statistics
flag_IPCS = True # solver selection (IPCS Projection method with P2-P1 element or intrinsic method with Mini-element)
plot_while_solving = False # produce plots for movie, which does not require saving and reloading data (in case in fail sometimes on cluster) but slows down the solver
flag_xray_after_solving = True # perform xray immediately after solving at terminal time
flag_tqdm = FLAG_TQDM # whether to show status bar
flag_xdmf = False # save data in xdmf format
flag_pvd = False # save data in pvd format, for visualization e.g. paraview
freq_plot = 1 # downsample frames to film movies faster
uname = 'NSdata/u_series' # where to store data
pname = 'NSdata/p_series' 
xname = 'xray/'
# only for mini
with_teman = False # stablization scheme for mini element
with_bf_est = False # stablization scheme for mini element

print("using parameters (s, d) = ",s,diam_narrow)

######## Geometry Variables ########
diam_steno_vessel=0.1
theta_steno=np.pi/6
diam_healthy_vessel=0.1
theta_healthy=np.pi/6
length0 = .5
length = .3
length_steno  = .2 # 2*diam_steno_vessel                      # Length of stenosis
diam_trunk = diam_healthy_vessel * np.cos(theta_healthy) + diam_steno_vessel * np.cos(theta_steno)
mesh_precision = 30
######## Physical Variables ########
mu = 0.03 # viscosity
rho = 1 # homogeneous density
u0 = 1 # 20#2.                 # init amplitude

######## Windkessel Variables ########
c = 1#1.6e-5              # distant capacitance
Rd = 1#6001.2             #1e5 #6001.2 distant resistance
Rp = 1#7501.5             #5e4 #7501.5 proximal resistance
p_windkessel_1 = 1#1.06e5 #1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = 1#1.06e5 #1.06e5 # init val

######## Mesh ########
class Artery():
    def __init__(self, diam_steno_vessel=0.1, diam_narrow=0.04, theta_steno=np.pi/6, diam_healthy_vessel=0.1, theta_healthy=np.pi/6,length0 = .5,length = .3, length_steno = .2):
        self.diam_steno_vessel = diam_steno_vessel
        self.diam_narrow = diam_narrow#diam_narrow
        self.theta_steno = theta_steno
        self.diam_healthy_vessel = diam_healthy_vessel
        self.theta_healthy = theta_healthy
        self.length0 = length0
        self.length = length
        self.length_steno = length_steno
        self.diam_trunk = diam_healthy_vessel * np.cos(theta_healthy) + diam_steno_vessel * np.cos(theta_steno)
    def __vessel_healthy(self):
        """
            Points for the
            Healthy vessel in the upper part of the bifurcation
        """
        D  = self.diam_healthy_vessel   # Diameter vessel

        n = 20 # Number of points to build domain (impact on mesh size)
        length = self.length

        # Bottom
        xref = np.linspace(-length, length, num=n)
        yref = np.zeros(n)
        points_bottom = [Point(x, y) for (x,y) in zip(xref,yref)]

        # Top
        xref = np.linspace(length, -length, num=n)
        yref = D*np.ones(n)
        points_top = [Point(x, y) for (x,y) in zip(xref,yref)]

        vertices = points_bottom + points_top

        # Translate to origin
        vertices = [ Point(p[0]+length,p[1]) for p in vertices ]

        # Rotate
        theta = self.theta_healthy
        vertices = [ Point(np.cos(theta)*p[0]-np.sin(theta)*p[1],np.sin(theta)*p[0]+np.cos(theta)*p[1]) for p in vertices ]
        return vertices
    def __vessel_stenosis(self):
        """
            Points for the
            Stenotic vessel in the lower part of the bifurcation
        """
        D  = self.diam_steno_vessel   # Diameter vessel
        diam_narrow = self.diam_narrow                  # Narrowing in stenosis (diam_narrow < D/2)
        # L  = 2*D                      # Length of stenosis
        L  = self.length_steno     # Length of stenosis
        x0 = 0.                       # location of the center of the stenosis
        length = self.length
        def S(x,length_steno):
            """
                Section of the stenosis following the paper
                "Direct numerical simulation of stenotic flows,
                Part 1: Steady flow" J. Fluid Mech.
            """
            
            # return D/2 * (1-diam_narrow*(1+np.cos(2*np.pi*(x-x0)/L)))
            return D/2 -diam_narrow/2*(1+np.cos(2*np.pi*(x-x0)/L))
        n = 30 # Number of points to build domain (impact on mesh size)
        # Bottom
        xref = np.linspace(-length, length, num=n)
        yref = [ -S(x,L) if -L/2<= x and x <= L/2 else -D/2 for x in xref]
        points_bottom = [Point(x, y) for (x,y) in zip(xref,yref)]

        # Top
        xref = np.linspace(length, -length, num=n)
        yref = [ S(x,L) if -L/2<= x and x <= L/2 else D/2 for x in xref]
        points_top = [Point(x, y) for (x,y) in zip(xref,yref)]

        vertices = points_bottom + points_top

        # Translate to origin
        vertices = [ Point(p[0]+length,p[1]-D/2) for p in vertices ]

        # Rotate
        theta = -self.theta_steno
        vertices = [ Point(np.cos(theta)*p[0]-np.sin(theta)*p[1],np.sin(theta)*p[0]+np.cos(theta)*p[1]) for p in vertices ]
        return vertices

    def __domain(self):
        """
            Construction of the bifurcation geometry as a Polygon object
        """

        vertices_stenosis = self.__vessel_stenosis()
        vertices_healthy = self.__vessel_healthy()
        length0 = self.length0

        n=10

        xl = vertices_stenosis[0][0]
        yl = vertices_stenosis[0][1]
        xref = np.linspace(-length0, xl, num=n, endpoint=False)
        vertices_bottom_left = [ Point(x,yl) for x in xref ]

        xr = vertices_healthy[-1][0]
        yr = vertices_healthy[-1][1]
        xref = np.linspace(xr, -length0, num=n)
        vertices_top_left = [ Point(x,yr) for x in xref ]

        v = vertices_bottom_left + vertices_stenosis + vertices_healthy[1:] + vertices_top_left

        return Polygon(v)

    def mesh(self,mesh_precision = 40):
        """
            Create mesh of the geometry
        """
        # mesh_precision = 20
        return generate_mesh(self.__domain(), mesh_precision)

######## BC ########
def inflow_domain(x, on_boundary):
    "Define boundaries"
    return near(x[0]+length0, 0) and on_boundary

def heartfun(x):
    "Define inflow with heartbeat"
    xp=np.array([0,0.025,0.17,0.3,0.38,0.45,0.55,0.65,0.75,0.9,1,1.025])
    yp=np.array([0.17,0.1,1,0.23,0.27,0.0,0.35,0.22,0.22,0.17,0.17,0.1])+.02
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

######## Tool ########
def linspace_2d(p1,p2,nn=20):
    "several points on line segments"
    xx = np.linspace(p1[0],p2[0],nn)
    yy = np.linspace(p1[1],p2[1],nn)
    return [i for i in zip(xx,yy)]

def eval_fun(u,grid):
    "return a list of evaluation of function u on grid"
    return [u(i) for i in grid]

def find_endpoint(x,y,Y,theta,tol_narrow = .001, tol_shift = .001):
    "find endpoints of diagnosis line. Out: rotation(theta,(x,y)) and rotation(theta,(x,Y)) "
    p1 = np.array([x-tol_shift,y + tol_narrow])
    p1 = rotate(theta,p1)
    p2 = np.array([x-tol_shift,Y - tol_narrow])
    p2 = rotate(theta,p2)
    return p1,p2

def flux(u,p1,p2,theta,arclength,nn=20):
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

def integrate_over_line(p,p1,p2,arclength,nn=20):
    "line intergral of p between two points."
    grid = linspace_2d(p1,p2,nn)
    funval = eval_fun(p,grid)
    out = sum(funval)/nn*arclength

def find_gridpoints(x,X,y,Y,nx = 20,ny = 10):
    xx = np.linspace(x, X, nx)
    yy = np.linspace(y, Y, ny)
    XX, YY = np.meshgrid(xx, yy)
    XX = XX.reshape(1,-1).tolist()[0]
    YY = YY.reshape(1,-1).tolist()[0]
    points = [np.array(i) for i in zip(XX,YY)]
    return points

def plot_solution(u_,p_,fname = "solution.pdf",title = 'Velocity',vmin=0, vmax=150):
    import matplotlib
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.8,0.05])
    gs.update(left=0.05, right=0.9, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    ax1 = plt.subplot(gs[0,0])
    ax1.set_aspect('equal')
    import matplotlib
    plt1 = plot(p_,norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False))
    plt2 = plot(u_,title = title)#,norm=matplotlib.colors.Normalize(vmin=0, vmax=10000, clip=False))
    cbax = plt.subplot(gs[0,1])
    cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'vertical', ticklocation = 'right')
    plt.savefig(fname)

def plot_diagnosis(T,num_steps,diagnoses,fname = 'diagnoses.pdf',pmin=0, pmax=60, head_off = 3):
    tt = np.linspace(0,T,num_steps-head_off)
    plt.figure()
    for p in diagnoses:
        for i in range(head_off):
            p.pop(i)
        plt.plot(tt,p)
    # print(integrate_over_line(u_,p1_healthy,p2_healthy,1))
    plt.ylim(pmin,pmax)
    plt.savefig(fname)

def compute_space_IPCS(mesh):
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    return V,Q

def movie(fname = "./output/output.mp4"):
    # print('Making animation - this may take a while')
    subprocess.call("ffmpeg -i ./tmp/_tmp%05d.png -r 60 -pix_fmt yuv420p "+fname, shell=True)

def make_movie(mesh, 
    T,
    num_steps,
    uname = 'NSdata/u_series',
    pname = 'NSdata/p_series',
    # writename = './output/output.mp4',
    tol = 0.001,
    flag_tqdm = True,
    flag_IPCS = True):

    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)
    if flag_IPCS:
        V,Q = compute_space_IPCS(mesh)
        u = Function(V)
        p = Function(Q)
    else:
        Mini = compute_mini(mesh)
        u = Function(Mini.sub(0).collapse())
        p = Function(Mini.sub(1).collapse())

    t = 0
    dt = T / num_steps
    ### status bar
    if flag_tqdm:
        pbar = tqdm(total=T)
    files = []
    for n in range(num_steps):
        # Update current time
        t += dt
        # Read velocity from file
        timeseries_u.retrieve(u.vector(), t)
        timeseries_p.retrieve(p.vector(), t)
        fname = './tmp/_tmp%05d.png' % n
        files.append(fname)
        plot_solution(u,p,fname,title = '%.2f' % t)
        if flag_tqdm:
            pbar.update(dt)
            pbar.set_description("Plotting... t = %.4f" % t)
    if flag_tqdm:
        pbar.close()
    folder = 'diagnoses/'
    name = folder+'s=%.2f,diam_narrow=%2f.mp4' % (s,diam_narrow)
    movie(name)
    return files

def xray(mesh, 
    critical_time,
    s = 0,
    diam_narrow = 0,
    uname = 'NSdata/u_series',
    pname = 'NSdata/p_series',
    xname = 'diagnoses/',
    x=-.4,X=-.2,y=-.03,Y=.03,
    flag_IPCS = True):
    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)
    if flag_IPCS:
        V,Q = compute_space_IPCS(mesh)
        u = Function(V)
        p = Function(Q)
    else:
        Mini = compute_mini(mesh)
        u = Function(Mini.sub(0).collapse())
        p = Function(Mini.sub(1).collapse())
    t = critical_time
    # Read velocity from file
    timeseries_u.retrieve(u.vector(), t)
    timeseries_p.retrieve(p.vector(), t)
    # diagnosis 
    upoints = find_gridpoints(x,X,y,Y)
    udata = eval_fun(u,upoints)
    folder = xname
    name = folder + 's=%.2f,diam_narrow=%2f.csv' % (s,diam_narrow)
    np.savetxt(name, udata, delimiter=",")
    return udata,s,diam_narrow

def diagnosis(mesh, 
    T,num_steps,
    s = 0,
    diam_narrow = 0,
    uname = 'NSdata/u_series',
    pname = 'NSdata/p_series',
    dname = 'diagnoses/',
    flag_tqdm = True,
    flag_IPCS = True,
    tol = 0.001):
    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)
    if flag_IPCS:
        V,Q = compute_space_IPCS(mesh)
        u = Function(V)
        p = Function(Q)
    else:
        Mini = compute_mini(mesh)
        u = Function(Mini.sub(0).collapse())
        p = Function(Mini.sub(1).collapse())

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
    p1_healthy, p2_healthy = find_endpoint(length*2,0,diam_healthy_vessel,-theta_healthy)
    p2_steno, p1_steno = find_endpoint(length*2,-diam_steno_vessel,0,theta_steno)

    t = 0
    dt = T / num_steps
    ### status bar
    if flag_tqdm:
        pbar = tqdm(total=T)
    p_int_inflow = []
    p_int_before_stenosis = []
    p_int_after_stenosis = []
    p_int_healthy = []
    for n in range(num_steps):
        # Update current time
        t += dt
        # Read velocity from file
        # timeseries_u.retrieve(u.vector(), t)
        timeseries_p.retrieve(p.vector(), t)
        # diagnosis on p
        p_int_inflow            .append(p(p1_inflow        ))#integrate_over_line(p_ ,p1_inflow        ,p2_inflow      ,artery.diam_trunk))
        p_int_before_stenosis   .append(p(p1_steno_before  ))#integrate_over_line(p_ ,p1_steno_before  ,p2_steno_before    ,artery.diam_steno_vessel))
        p_int_after_stenosis    .append(p(p1_steno_after   ))#integrate_over_line(p_ ,p1_steno_after   ,p2_steno_after     ,artery.diam_steno_vessel))
        p_int_healthy           .append(p(p1_healthy_mid   ))#integrate_over_line(p_ ,p1_healthy       ,p2_healthy         ,artery.diam_healthy_vessel))
        if flag_tqdm:
            pbar.update(dt)
            pbar.set_description("Diagnosing... t = %.4f" % t)
    if flag_tqdm:
        pbar.close()

    # plt.figure()
    plot = plt.plot(p_int_inflow)
    plt.plot(p_int_before_stenosis)
    plt.plot(p_int_after_stenosis)
    plt.plot(p_int_healthy)
    # plt.show()
    plt.legend(['p_int_inflow', 'p_int_before_stenosis', 'p_int_after_stenosis', 'p_int_healthy'])
    folder = dname
    name = folder+'s=%.2f,diam_narrow=%2f.pdf' % (s,diam_narrow)
    plt.savefig(name)
    diagnoses = [p_int_inflow, p_int_before_stenosis, p_int_after_stenosis, p_int_healthy]
    return diagnoses

def compute_NSsolution_IPCS(mesh,
    T,
    num_steps,
    mu,
    rho,
    # windkessel,
    c,
    Rd,
    Rp,
    p_windkessel_1,
    p_windkessel_2,
    u0,
    s,
    uname = 'NSdata/u_series',
    pname = 'NSdata/p_series',
    xname = "xray/",
    # flag_movie = False,
    flag_IPCS = True,
    flag_xray = True,
    flag_tqdm = True,
    flag_xdmf = True,
    flag_pvd = True,
    flag_ts = True,
    plot_while_solving = False,
    flag_xray_after_solving = True):
    "IPCS Scheme"
    dt = T / num_steps
    inflow_expr = INFLOW(u0,s,diam_steno_vessel, theta_steno, diam_healthy_vessel, theta_healthy,degree=2)
    tol=.001
    # Points:
    ### Flux surface for u
    p1_healthy, p2_healthy = find_endpoint(length*2,0,diam_healthy_vessel,-theta_healthy)
    p2_steno, p1_steno = find_endpoint(length*2,-diam_steno_vessel,0,theta_steno)
    ### status bar
    if flag_tqdm:
        pbar = tqdm(total=T)
    if flag_xdmf:
        xdmffile_u = XDMFFile('NSdata/velocity.xdmf')
        xdmffile_p = XDMFFile('NSdata/pressure.xdmf')
    if flag_pvd:
        pvdfile_u = File("NSdata/velocity.pvd")
        pvdfile_p = File("NSdata/pressure.pvd")
    if flag_ts:
        # Create time series (for use in reaction_system.py)
        timeseries_u = TimeSeries(uname)
        timeseries_p = TimeSeries(pname)
    ### Time-stepping
    t = 0.0
    # files = []
    flux_healthy = []
    flux_stenosis = []

    if flag_IPCS:
        ### Define function spaces
        V, Q = compute_space_IPCS(mesh)
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
        ds = Measure('exterior_facet', subdomain_id='everywhere')
        # print(ds)
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
        # p_int_inflow = []
        # p_int_before_stenosis = []
        # p_int_after_stenosis = []
        # p_int_healthy = []
        for n in range(num_steps):
            # Update current time
            t += dt
            # u_avg_1 = flux(u_,p1_healthy,p2_healthy,theta_healthy,diam_healthy_vessel)
            # u_avg_2 = flux(u_,p1_steno,p2_steno,-theta_steno,diam_steno_vessel)
            # delta1 = dt/c*(-p_windkessel_1/Rd+u_avg_1)
            # # delta_windkessel1.append(delta1)
            # p_windkessel_1 += delta1
            # delta2 = dt/c*(-p_windkessel_2/Rd+u_avg_2)
            # p_windkessel_2 += delta2
            # # delta_windkessel2.append(delta2)
            # p_bdry_1 = Rp * u_avg_1 + p_windkessel_1
            # p_bdry_2 = Rp * u_avg_2 + p_windkessel_2
            # print(p_bdry_1,p_bdry_2)
            # p_bdry_1 = 10
            # p_bdry_2 = 10
            # bcu, bcp = compute_bc(V,Q,t,p_bdry_1,p_bdry_2,u0,s,inflow_expr,inflow_domain,heartfun,)
            bcu, bcp = compute_bc(V,Q,t,0,0,u0,s,inflow_expr,inflow_domain,heartfun,)
            [bc.apply(A1) for bc in bcu]
            [bc.apply(A2) for bc in bcp]
            [bc.apply(A3) for bc in bcu]
            # Step 1: Tentative velocity step
            b1 = assemble(L1)
            [bc.apply(b1) for bc in bcu]
            [bc.apply(b1) for bc in bcp]
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
            if flag_xdmf:
                xdmffile_u.write(u_, t)
                xdmffile_p.write(p_, t)
            if flag_pvd:
                pvdfile_u << (u_, t)
                pvdfile_p << (p_, t)
            if flag_ts:
                # Create time series (for use in reaction_system.py)
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
            if flag_tqdm:
                pbar.update(dt)
                pbar.set_description("t = %.4f " % t + 'u_max: %.2f, ' % u_.vector().vec().max()[1] + 'p_max: %.2f ' % p_.vector().vec().max()[1])
            if plot_while_solving:
                fname = './tmp/_tmp%05d.png' % n
                plot_solution(u_,p_,fname)
                # files.append(fname)
    else:
        ################ Mini
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
            solve(A, w.vector(), b)
            # solve(A, w.vector(), b, 'superlu')

            ut, pt = w.split(deepcopy = True)

            # Save solution to file
            # ufile_pvd << ut, t
            # pfile_pvd << pt, t
            # ufile_h5.write(ut, "velocity_" + str(t))
            # pfile_h5.write(pt, "pressure_" + str(t))
            if flag_xdmf:
                xdmffile_u.write(u_, t)
                xdmffile_p.write(p_, t)
            if flag_pvd:
                pvdfile_u << (u_, t)
                pvdfile_p << (p_, t)
            if flag_ts:
                # Create time series (for use in reaction_system.py)
                timeseries_u.store(u_.vector(), t)
                timeseries_p.store(p_.vector(), t)
                       
            # update variables
            u_.assign(ut)
            p_.assign(pt)
            
            # Update progress bar
            if flag_tqdm:
                pbar.update(dt)
                pbar.set_description("t = %.4f" % t + 'u_max:%.2f, ' % u_.vector().vec().max()[1] + 'p_max:%.2f ' % p_.vector().vec().max()[1])
            if plot_while_solving:
                fname = './tmp/_tmp%05d.png' % n
                plot_solution(u_,p_,fname)
                # files.append(fname)

    ##################

    if flag_xray_after_solving:
        # equiv to xray function
        x=-.4
        X=-.2
        y=-.03
        Y=.03
        upoints = find_gridpoints(x,X,y,Y)
        udata = eval_fun(u_,upoints)
        folder = xname
        name = folder + 's=%.2f,diam_narrow=%2f.csv' % (s,diam_narrow)
        np.savetxt(name, udata, delimiter=",")

    if flag_tqdm:
        pbar.close()
    return u_,p_

def cleanup(files):
    for fname in files:
        os.remove(fname)

if __name__ == '__main__':
    ######## Main porcess ########
    artery = Artery(diam_steno_vessel, diam_narrow, theta_steno, diam_healthy_vessel, theta_healthy)
    mesh = artery.mesh(mesh_precision)
    # File('NSdata/artery.xml.gz') << mesh
    # mesh = Mesh('NSdata/artery.xml.gz')
    plot(mesh)
    plt.savefig("mesh.jpg")

    u,p= compute_NSsolution_IPCS(mesh,
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
    xname = xname,
    flag_IPCS = flag_IPCS,
    flag_tqdm = FLAG_TQDM,
    flag_xdmf = flag_xdmf,
    flag_pvd = flag_pvd,
    flag_ts = flag_ts,
    plot_while_solving = plot_while_solving,
    flag_xray_after_solving = flag_xray_after_solving
    )
    # else: #mini 
    #     u,p = compute_NSsolution_mini(mesh,
    #     T = T                  ,
    #     num_steps = num_steps         ,
    #     mu = mu               ,
    #     rho = rho                 ,
    #     # # windkessel,
    #     c = c                   ,
    #     Rd = Rd                ,
    #     Rp = Rp                ,
    #     p_windkessel_1 = p_windkessel_1 ,
    #     p_windkessel_2 = p_windkessel_2 ,
    #     u0 = u0                  ,
    #     # flag_movie = flag_movie,
    #     with_teman = with_teman,
    #     with_bf_est = with_bf_est,
    #     freq_plot = freq_plot)

    if flag_movie: # plotting is slow
        files = make_movie(mesh, 
                T,
                num_steps,
                uname = uname,
                pname = pname,
                # writename = './output/output.mp4',
                flag_tqdm = FLAG_TQDM,
                flag_IPCS = flag_IPCS)
        if flag_cleanup:
            cleanup(files)

    if flag_xray:
        udata,s,diam_narrow = xray(mesh, 
                critical_time,
                s = s,
                diam_narrow = diam_narrow,
                xname = xname,
                uname = uname,
                pname = pname,
                flag_IPCS = flag_IPCS)
    if flag_diagnosis:
        diagnoses = diagnosis(mesh, 
                    T,num_steps,
                    s = s,
                    diam_narrow = diam_narrow,
                    dname = dname,
                    uname = uname,
                    pname = pname,
                    flag_IPCS = True)
    

#     # print('NS computation completed.')