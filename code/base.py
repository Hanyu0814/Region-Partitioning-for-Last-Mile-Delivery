import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import integrate
from sympy.integrals.intpoly import *
from IPython import display
from sympy.plotting import plot
import datetime
from shapely.geometry import Polygon
import shapely
from scipy import interpolate

class MyError(Exception):
    pass

def intpoly(f, x, y, error=0.00001): 
    '''Calculate the double integral of polygon.'''
    xy = np.vstack((x, y))
    _,idx = np.unique(xy, axis=1, return_index=True)
    xy = xy[:,np.sort(idx)]
        
    x = xy[0,:]
    y = xy[1,:]
    
    index_xmin = x.argmin()
    x = np.roll(x,-(index_xmin))
    y = np.roll(y,-(index_xmin))
    index_xmax = x.argmax()
    index_xmin = x.argmin()
   
    y_mid = min(y[index_xmax],y[index_xmin])
    if y[1] >= y_mid:
        up = list(range(0,index_xmax+1))
        down = list(range(index_xmax,(len(y)))) + [0]
    else:
        down = list(range(0,index_xmax+1))
        up = list(range(index_xmax,(len(y)))) + [0]
        
    # check if there are two nodes with the same x axis.
    uplist = np.vstack((x[up], y[up]))
    downlist = np.vstack((x[down], y[down]))
    
    # check if there are nodes with the same x axis.
    if (len(np.where(uplist[0,:] == uplist[0,:].min())[0])) > 1: # check if x_min duplicate
        # remove the element with lower y
        idx = np.where(uplist[0,:] == uplist[0,:].min())[0]
        lowest_x_min_index = idx[uplist[1,idx].argmin()]
        uplist = np.delete(uplist, lowest_x_min_index, 1)
    elif (len(np.where(downlist[0,:] == downlist[0,:].min())[0])) > 1:
        # remove the element with higher y
        idx = np.where(downlist[0,:] == downlist[0,:].min())[0]
        highest_x_min_index = idx[downlist[1,idx].argmax()]
        downlist = np.delete(downlist, highest_x_min_index, 1)
    
    if (len(np.where(uplist[0,:] == uplist[0,:].max())[0])) > 1: # check if x_max duplicate
        # remove the element with lower y
        idx = np.where(uplist[0,:] == uplist[0,:].max())[0]
        lowest_x_min_index = idx[uplist[1,idx].argmin()]
        uplist = np.delete(uplist, lowest_x_max_index, 1)
    elif (len(np.where(downlist[0,:] == downlist[0,:].max())[0])) > 1:
        # remove the element with higher y
        idx = np.where(downlist[0,:] == downlist[0,:].max())[0]
        highest_x_max_index = idx[downlist[1,idx].argmax()]
        downlist = np.delete(downlist, highest_x_max_index, 1)

    ymax = interpolate.interp1d(uplist[0,:], uplist[1,:], kind='linear')
    ymin = interpolate.interp1d(downlist[0,:], downlist[1,:], kind='linear')

    int_value = abs(integrate.dblquad(f, x.min(), x.max(), ymin, ymax, epsabs=error)[0])
    return int_value

def poly_vertices(p): # Get polygon vertices.
    allArrays = np.concatenate([np.array(p.vertices[i], dtype=np.float32).reshape((1, 2)) 
                          for i in range(len(p.vertices))]) 
    return allArrays

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)

def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def gvalue(p,n,f,g,epsilon_r=0.001,epsilon_integral=0.00001): 
    '''Calculate g_value for three partitions.'''
    n = int(n)
    Vertices = poly_vertices(p) # Change p!
    A = abs(intpoly(f, Vertices[:,0], Vertices[:,1],epsilon_integral)); #calculate mu1
    A2 = abs(intpoly(g, Vertices[:,0], Vertices[:,1],epsilon_integral)) #calculate mu2
    x1=min(Vertices[:,0])
    x2=max(Vertices[:,0])
    M=1 # to count the number of vertical lines
    Cg=[] # a list to save created halfplanes
    S=[0] * (n-1) # Create a list for signs of g-1 halfplanes

    p2=Polygon(np.column_stack((Vertices[:,0],Vertices[:,1])))
    vertical_line=[]
    while M<n: # find g-1 vertical lines
        while True:
            x = (x1 + x2)/2; # find middle x
            S1 = np.array([-10,-10,x,-10, x,10, -10,10,-10,-10]).reshape((5,2)); # define a vertcal cut by x location
            p1 = Polygon(S1)
            [X,Y] = np.array(p2.intersection(p1).exterior.coords.xy); # use the plane S1 to cut the polygon
            D = sp.Polygon(*(tuple(zip(X,Y))))
            A1 = intpoly(f,poly_vertices(D)[:,0],poly_vertices(D)[:,1],epsilon_integral);
            if A1 < (A*M)/n - epsilon_r:
                x1 = x;
            elif A1 > (A*M)/n + epsilon_r:     
                x2 = x;
            else:
                break;

        x = (x1 + x2)/2; #find middle x that cut the area
        S1 = np.array([-10,-10,x,-10, x,10, -10,10,-10,-10]).reshape((5,2)); # the cutting hyperplane
        p1 = Polygon(S1) 
        C1 = np.column_stack(p2.intersection(p1).exterior.coords.xy); # find cutted area
        Cg.append(C1) # the cutted area list
        vertical_line.append(x)
        
        m2 = intpoly(g,C1[:,0], C1[:,1],epsilon_integral) # find the sign of the halfplane
        if m2<(A2*M)/n:
            S1=-1
        else:
            S1=1

        S[M-1]=S1
        M=M+1
        x1=max(C1[:,0]) # update x1: the cutted plane x-axis
        x2=max(Vertices[:,0]) # update x2: the end point.

    print(S);
    n2 = int((n-1)/2)
    numbers = list(range(1, n))
    goodnums = {n-x for x in numbers if x<=n2} & {x for x in numbers if x>n2}
    pairs = {(n-x, x) for x in goodnums}  
    gvalues={(x, y) for (x,y) in pairs if S[x-1]==S[y-1]}
    boundary = []
    if len(gvalues)==0:
        gvalues={(x,y,z) for x in range(1,n-1) for  y in range(1,n-x) for z in [n-x-y] if S[x-1]==S[y-1]==S[z-1]}
        g1 = next(iter(gvalues))[0]
        boundary=[vertical_line[g1-1],vertical_line[n-g1-1]]

    return [next(iter(gvalues)),boundary]