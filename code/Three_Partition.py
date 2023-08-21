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

def get_one_side_partition(f,X,Y,x,y,theta1,theta2,target_m,side,epsilon_theta=0.002,epsilon_integral=0.00001):
    """Find angle of the partiton ray."""
    theta = 0;
    # Binary search to find the cutting ray angle.
    while abs(theta1 - theta2) > 0.0001:
        theta = (theta1+theta2)/2
        # Change the theta value, and find the cutting plane.
        X1 = X-x
        Y1 = Y-y
        L_p = Polygon(np.column_stack((X1,Y1)))
        S1 = np.array([-1,-1,1,-1,1,0,-1,0,-1,-1]).reshape((5,2)); # define a horizontal cut by y location
        [TH,R] = cart2pol(S1[:,0],S1[:,1]);
        # Find the ray's angle.
        if (side == "right"):
            TH = TH - (np.pi/2-theta);
        elif (side == "left"):
            TH = TH + (np.pi/2-theta);
        else:
            raise MyError("Unknown side parameter.")
        [X2,Y2]= pol2cart(TH,R);
        p1 = Polygon(np.column_stack((X2,Y2)))

        [X2,Y2] = np.array(L_p.intersection(p1.buffer(0.00001)).exterior.coords.xy); # use the plane S1 to cut the polygon
        X2 = X2+x
        Y2 = Y2+y
        A1 = intpoly(f,X2,Y2,epsilon_integral); # Calculate intergral

        if A1 < target_m - epsilon_theta:
            theta1 = theta;
        elif A1 > target_m + epsilon_theta:     
            theta2 = theta;
        else:
            return X2,Y2,theta;
            break;
    return None, None, None
            
def three_cut(p,f,g,n1,n2,n3,l1,l2,epsilon_theta = 0.002,epsilon_integral=0.00001):
    """Three partition function"""

    Vertices = poly_vertices(p);
    n = n1 + n2 + n3;
    m1 = float(intpoly(f, Vertices[:,0], Vertices[:,1],epsilon_integral));
    m2 = float(intpoly(g, Vertices[:,0], Vertices[:,1],epsilon_integral));
    orig_p=Polygon(np.column_stack((Vertices[:,0],Vertices[:,1])))
    print(epsilon_theta)
    print("m1, ", m1, ", m2, ", m2)
    
    x = l1
    h_x = 0.003 # step size in x
    h_y = 0.003   # step size in y
    theta = 0 # initial angle
    theta1 = 0 # angle lower bound
    theta2 = np.pi # angle upper bound

    # Grid search to find the apex coordinate.
    while x <= l2:
        print("x", x)
        # cut the left hand side
        S1 = np.array([-10,-10,x,-10, x,10, -10,10,-10,-10]).reshape((5,2)); # define a vertcal cut by x location
        p1 = Polygon(S1)
        [L_X,L_Y] = np.array(orig_p.intersection(p1).exterior.coords.xy); # use the plane S1 to cut the polygon
        L_p = Polygon(np.column_stack((L_X,L_Y)))

        # cut the right hand side
        S1 = np.array([10,-10,x,-10, x,10, 10,10,10,-10]).reshape((5,2)); # define a vertcal cut by x location
        p1 = Polygon(S1)
        [R_X,R_Y] = np.array(orig_p.intersection(p1).exterior.coords.xy); # use the plane S1 to cut the polygon
        R_p = Polygon(np.column_stack((R_X,R_Y)))
        
        itemindex = np.where(R_X==x)
        y1_high = max(R_Y[itemindex[0][0]],R_Y[itemindex[0][1]])
        y1_low = min(R_Y[itemindex[0][0]],R_Y[itemindex[0][1]])
        print("y1_high: ", y1_high," y1_low: ", y1_low)
        
        y = y1_low;  
        A1 = intpoly(f,L_X,L_Y,epsilon_integral);
        A2 = intpoly(g,L_X,L_Y,epsilon_integral);
        if (A1 <= ((m1*n1)/n - epsilon_theta)) or (A2 <= ((m2 * n1)/n - epsilon_theta)):
            x = x + h_x;
            continue;
        while y <= y1_high:
            Partition = [];
            target_m = (m1*n1)/n;
            
            # Calculate the ray angle.
            [X2,Y2,theta] = get_one_side_partition(f,L_X,L_Y,x,y,0,np.pi,target_m,"left",epsilon_theta,epsilon_integral)
            A1 = intpoly(f,X2,Y2,epsilon_integral);
            A2 = intpoly(g,X2,Y2,epsilon_integral);
            
            if (A2 >= (m2*n1)/n + epsilon_theta) or (A2 <= (m2*n1)/n - epsilon_theta):
                y = y + h_y;
            else:
                
                XY_2 = np.column_stack((X2,Y2))
                
                Partition.append(XY_2)
                
                # find the cut on the other side.
                target_m = (m1*n3)/n;
                [X2,Y2,theta] = get_one_side_partition(f,R_X,R_Y,x,y,np.pi-theta,np.pi,target_m,"right",epsilon_theta,epsilon_integral)   # Sheng
                if theta is not None:

                    A1 = intpoly(f,X2,Y2,epsilon_integral);
                    A2 = intpoly(g,X2,Y2,epsilon_integral);
                                        
                    if (A2 <= (m2*n3)/n + epsilon_theta) & (A2 >= (m2*n3)/n - epsilon_theta):
                        XY_2 = np.column_stack((X2,Y2))
                        Partition.append(XY_2)
                        
                        p1 = Polygon(Partition[0])
                        p2 = Polygon(Partition[1])

                        XY_3 = np.array(orig_p.difference(p1.buffer(0.0001)).difference(p2.buffer(0.0001)).exterior.coords.xy); #f ind cutted area 3 c2: Sheng: add buffer
                        XY_3 = np.column_stack((XY_3[0],XY_3[1]))
                        
                        Partition.append(XY_3)
                        return Partition[0],Partition[2],Partition[1]
                        break;
                    else:
                        y = y + h_y;
                else:
                    y = y + h_y;
                    
        x = x + h_x;
