# from mesh_server import *
from BC import *
import os
import subprocess
import numpy as np
import tqdm
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# import matplotlib.pyplot as plt
# import dolfin 

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


def find_gridpoints(x,X,y,Y,nx = 10,ny = 4):
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
    p1_healthy, p2_healthy = find_endpoint(length*2,0,diam_healthy_vessel,-theta_healthy)
    p2_steno, p1_steno = find_endpoint(length*2,-diam_steno_vessel,0,theta_steno)


    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)
    V,Q = compute_space_IPCS(mesh)
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
    files = []
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
        files.append(fname)
        plot_solution(u,p,fname,title = '%.2f' % t)
        pbar.update(dt)
        pbar.set_description("Plotting... t = %.4f" % t)
    pbar.close()
    movie(writename)
    diagnoses = [p_int_inflow, p_int_before_stenosis, p_int_after_stenosis, p_int_healthy]
    return diagnoses,files

def diagnosis_IPCS(mesh, 
                    T,
                    s = s,
                    uname = 'NSdata/u_series',
                    pname = 'NSdata/p_series',
                    x=-.4,X=-.2,y=-.03,Y=.03):
    timeseries_u = TimeSeries(uname)
    timeseries_p = TimeSeries(pname)
    V,Q = compute_space_IPCS(mesh)
    u = Function(V)
    p = Function(Q)
    t = T
    # Read velocity from file
    timeseries_u.retrieve(u.vector(), t)
    timeseries_p.retrieve(p.vector(), t)
    # diagnosis 
    upoints = find_gridpoints(x,X,y,Y)
    udata = eval_fun(u,upoints)
    np.savetxt('tmp/s=%.2f,diam_narrow=%2f.csv' % (s,diam_narrow), udata, delimiter=",")
#     pbar.close()
    # movie(writename)
    return udata,s,diam_narrow

