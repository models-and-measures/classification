# from __future__ import print_function
from dolfin import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import mesh_server

if __name__ == '__main__':
    print("hello")
    plt.plot([1,2,3,2,1])
    plt.savefig('temp.png')
