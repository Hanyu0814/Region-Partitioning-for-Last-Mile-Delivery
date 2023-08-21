"""Two partition function."""

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
from base import *

def ham_sandwich_cut(C,f,g,target_m1,target_m2,epsilon_theta=0.002,epsilon_r=0.001,epsilon_integral=0.00001):
    """ Two partition function: ham sandwich cut"""
    Vertices2 = poly_vertices(C);
    m1 = intpoly(f,Vertices2[:,0], Vertices2[:,1],epsilon_integral); # double integral for the first function inside the polygon C
    max_num = 30;   # limit for the existence test
    
    [C1,C2] = get_eq_area2(C,0,g,target_m2,epsilon_r,epsilon_integral); # find the two equal partition C1 and C2 for area C and theta=0
    poly_C1 = sp.Polygon(*(tuple(zip(C1[:,0],C1[:,1]))))
    
    m1_C1 = intpoly(f,C1[:,0],C1[:,1],epsilon_integral);
    
    if m1_C1 < target_m1 - epsilon_theta:
        theta1 = 0; theta2 = np.pi; 
        sign = -1;
    elif m1_C1 > target_m1 + epsilon_theta:
        theta1 = -np.pi; theta2 = 0;
        sign = 1;
    else:
        print("Finish Ham sandwich.")
        return C1,C2
    flag = False;
    
    # Calculate the sign to decide initialization.
    [C11, C21] = get_eq_area2(C,np.pi,g,target_m2,epsilon_r,epsilon_integral)
    poly_C11 = sp.Polygon(*(tuple(zip(C11[:,0],C11[:,1]))))
    m1_C11 = intpoly(f,C11[:,0],C11[:,1],epsilon_integral);
    sign_pi = np.sign(m1_C11 - target_m1);
    if sign_pi == sign:
        temp_list = [[-np.pi, 0], [0, np.pi]];
        temp_sign = sign_pi;
        while (temp_sign == sign) and (len(temp_list) <= max_num):
            bracket = temp_list.pop(0);
            mid_bracket = (bracket[0] + bracket[1])/2;
            temp_list.append([bracket[0], mid_bracket]);
            temp_list.append([mid_bracket, bracket[1]]);
            [tempC11, tempC21] = get_eq_area2(C,mid_bracket,g,target_m2,epsilon_r,epsilon_integral)
            temp_m1_C11 = intpoly(f,tempC11[:,0],tempC11[:,1],epsilon_integral);
            temp_sign = np.sign(temp_m1_C11 - target_m1);
        
        if len(temp_list) > max_num:
            return None, None   # infeasibility
        if temp_sign > 0:
            theta1 = bracket[0];
            theta2 = mid_bracket;
        else:
            theta1 = mid_bracket;
            theta2 = bracket[1];
    
    # Binary search to find the cut to partition the region into two.
    while(flag == False):
        
        theta = (theta1+theta2)/2; # reset theta

        [C1,C2] = get_eq_area2(C,theta,g,target_m2,epsilon_r,epsilon_integral); # find the two equal partition C1 and C2 for area C and theta
        
        poly_C1 = sp.Polygon(*(tuple(zip(C1[:,0],C1[:,1]))))
        
        m1_C1 = intpoly(f,C1[:,0],C1[:,1],epsilon_integral);
        

        if(m1_C1 > target_m1 - epsilon_theta)&(m1_C1 < target_m1 + epsilon_theta):
            return C1,C2
        else:
            print(theta, m1_C1, target_m1)
            if m1_C1 < target_m1 - epsilon_theta:
                theta1 = theta;
            else:
                theta2 = theta;
                
def get_eq_area2(C,theta,g,target_m,epsilon_r=0.001,epsilon_integral=0.00001):
    """ Find line cut to get eqaul area."""

    Vertices = poly_vertices(C);
        
    [TH,R] = cart2pol(Vertices[:,0], Vertices[:,1]); # Transform the location for vertices to theta and R (Polar)
    TH = TH-theta; # Rotate all vertices with theta be horizontal.
    [CX,CY] = pol2cart(TH,R); # transfer vertices to cartesian coordinate
    y1 = min(CY); # the lowest point y
    # print("y1", y1)
    y2 = max(CY); # the hight point y
    # print("y2", y2)
    p2 = Polygon(np.column_stack((CX,CY)))
    
    # Binary search.
    while True:
        y = (y1 + y2)/2; # find middle y
        S1 = np.array([-10,-10,10,-10,10,y,-10,y,-10,-10]).reshape((5,2)); # define a horizontal cut by y location
        # the value 10  should be replaced by a big number so the cut goes through the whole region
        p1 = Polygon(S1)

        [X,Y] = np.array(p2.intersection(p1).exterior.coords.xy); # use the plane S1 to cut the polygon
        [TH,R] = cart2pol(X,Y); # find the cutted area polar coordinate
        TH = TH + theta; # turn it back
        [X,Y]= pol2cart(TH,R); # find the original cutted area cartisian coordinate

        D = sp.Polygon(*(tuple(zip(X,Y))))

        A1 = intpoly(g,poly_vertices(D)[:,0],poly_vertices(D)[:,1],epsilon_integral);

        if A1 < target_m - epsilon_r: # if m1(inter) < m1(all area)/2 move up y1
            y1 = y;
        elif A1 > target_m + epsilon_r:       # if m1(inter) < m1(all area)/2 move down y2
            y2 = y;
        else:
            break;
            
    # stop when y1 y2 met
    y = (y1 + y2)/2;
    S1 = np.array([-10,-10,10,-10,10,y,-10,y,-10,-10]).reshape((5,2));
    p1 = Polygon(S1)
    
    C1 = np.array(p2.intersection(p1.buffer(0.00001)).exterior.coords.xy); # find cutted area 1 c1
    C2 = np.array(p2.difference(p1.buffer(0.00001)).exterior.coords.xy); # find cutted area 2 c2
    [TH,R] = cart2pol(C1[0],C1[1]); 
    TH = TH + theta;
    C1 = np.column_stack(pol2cart(TH,R));

    [TH,R] = cart2pol(C2[0],C2[1]);
    TH = TH + theta;
    C2 = np.column_stack(pol2cart(TH,R));
    
    
    return C1,C2;