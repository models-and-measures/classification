import numpy as np
# global variables
# reasons why using it: 
# - local variables are not easy to fit in fenics built-in functions, namely geometric boundary, which take only the value of x (location).
# - simplicity and ease to modify

# Heart source
u0 = 1#20#2.                 # init amplitude
# symmetry!!!!!!!!!!
s = 0.#.5                  # init asymmetry
#shrink!!!!!!!!!!!!!
diam_narrow=0.03#0.02
diam_steno_vessel=0.1
theta_steno=np.pi/6
diam_healthy_vessel=0.1
theta_healthy=np.pi/6
length0 = .5
length = .3
length_steno  = .2 # 2*diam_steno_vessel                      # Length of stenosis
diam_trunk = diam_healthy_vessel * np.cos(theta_healthy) + diam_steno_vessel * np.cos(theta_steno)
mesh_precision = 40
#physics
mu = 0.03
rho = 1

#old geometry
sqr2 = 2**.5    #constant for simplicity
Y0 = 2          #Y trunk length
Y1 = 2 * sqr2   #Y branch length
y1 = 2          
Y2 = 2 * sqr2   #Y branch length
y2 = 2
D = .1 * sqr2   #branch radius
d = .1
D0 = 2*d        #trunk radius
A = .5          #shrink length = 2A
a = A / sqr2
B = .0          #shrink width = B
b = B / sqr2

# # windkessel,
c = 1#1.6e-5              # distant capacitance
Rd = 1#6001.2             #1e5 #6001.2 distant resistance
Rp = 1#7501.5             #5e4 #7501.5 proximal resistance
p_windkessel_1 = 1#1.06e5 #1.06e5 # init val, large number could lead to overflow
p_windkessel_2 = 1#1.06e5 #1.06e5 # init val

