import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import integrate
from IPython import display
import datetime
import shapely.geometry as sg
import shapely
import random
from scipy import interpolate
from base import *
import math
import os
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class MyError(Exception):
    pass

class demand_node:
    """Demand node class."""

    finish_time = None
    wait_time = None
    def __init__(self,index, x,y, time = None):
        self.x = x
        self.y = y
        self.index = index
        self.initial_time = time

class deliver_car:
    """Cars class."""
    available_time = 0
    def __init__(self,index=0,speed=0.5,state=0,service_list=[]):
        self.index = index
        self.speed = speed
        self.state = state
        self.service_list = service_list

def intervals(parts, lower, upper):
    part_duration = (upper - lower) / parts
    return [lower + i * part_duration for i in range(parts+1)]

def polygon_intersect_x(poly, x_val):
    if x_val < poly.bounds[0] or x_val > poly.bounds[2]:
        raise ValueError('`x_val` is outside the limits of the Polygon.')
    if isinstance(poly, sg.Polygon):
        poly = poly.boundary
    vert_line = sg.LineString([[x_val, poly.bounds[1]],
                               [x_val, poly.bounds[3]]])
    intersect_y = poly.intersection(vert_line)
    minx, miny, maxx, maxy = intersect_y.bounds
    if miny < maxy:
        pts = [pt.xy[1][0] for pt in intersect_y.geoms]   # for older versions of Shapely (<2.0), delete geoms
        pts.sort()
    else:
        pts = [intersect_y.xy[1][0],intersect_y.xy[1][0]]
    return pts

def random_node_y(p_xy, node_x, y_list, p_x):
    """Generate y-coordinate according to the joint pdf p_xy and marginal pdf p_x."""
    grid_y = 100;
    ygivenx_list = intervals(grid_y, y_list[0], y_list[1])
    
    ygivenx_list[0] = y_list[0]
    ygivenx_list[-1] = y_list[1]
    
    
    pdf_ygivenx_list = []
    tmp_cdf = 0
    cdf_ygivenx_list = []
    for j in range(len(ygivenx_list)-1):
        y_lb = ygivenx_list[j]
        y_ub = ygivenx_list[j+1]
        if p_x == 0:
            pdf_ygivenx = 0
        else:
            pdf_ygivenx = p_xy((y_lb+y_ub)/2, node_x)/p_x 
        
        pdf_ygivenx_list.append(pdf_ygivenx)
        tmp_cdf = tmp_cdf + pdf_ygivenx * (y_ub-y_lb)
        cdf_ygivenx_list.append(tmp_cdf)

    cdf_ygivenx_list.insert(0,0)
    interpolate_y = interpolate.interp1d(cdf_ygivenx_list, ygivenx_list)
    
    P_ygivenx = np.random.uniform(min(cdf_ygivenx_list), min(max(cdf_ygivenx_list),1 - 1e-6), 1)
    return interpolate_y(P_ygivenx)[0]

def generate_x_cdf(polygon, p_xy):
    """Marginal cdf on the x-axis"""
    minx, miny, maxx, maxy = polygon.bounds
    grid_x = 200

    x_list = intervals(grid_x, minx, maxx)
    x_list[0] = minx
    x_list[-1] = maxx
    
    pdf_x_list = []
    cdf_x_list = []
    tmp_cdf = 0

    for i in range(len(x_list)-1):
        x_lb = x_list[i]
        x_ub = x_list[i+1]
        intersect_y_list = polygon_intersect_x(polygon, (x_lb+x_ub)/2)

        result = integrate.quad(p_xy, intersect_y_list[0], intersect_y_list[1],args = ((x_lb+x_ub)/2))[0]
        tmp_cdf = tmp_cdf + result * (x_ub-x_lb)
        pdf_x_list.append(result)
        cdf_x_list.append(tmp_cdf)

    cdf_x_list.insert(0,0)

    interpolate_cdf_x = interpolate.interp1d(cdf_x_list, x_list)
    return cdf_x_list, interpolate_cdf_x
    
def generate_random_node(number, polygon, p_xy, init_time, count, cdf_x_list, interpolate_cdf_x):
    """Sample points according to p_xy in the polygon"""
    demand_node_list = random_x(number, polygon, p_xy, interpolate_cdf_x, cdf_x_list, init_time, count)
    return demand_node_list
    

def random_x(number, polygon, p_xy, interpolate_cdf_x, cdf_x_list, init_time, count): 
    """ Base sampling function"""
    num_node = 0
    points = []
    output_node_y_list = []
#     count = 0
    while num_node < number:
        P_x = np.random.uniform(min(cdf_x_list),min(max(cdf_x_list),1 - 1e-6),number-num_node)   # Sheng
        node_x_list = interpolate_cdf_x(P_x)
        for node_x in node_x_list:
            y_list = polygon_intersect_x(polygon, node_x)
            f_x = integrate.quad(p_xy, y_list[0], y_list[1],args = (node_x))[0]
            node_y = random_node_y(p_xy,node_x,y_list,f_x)
            pnt = sg.Point(node_x, node_y)
            if polygon.contains(pnt):
                a = demand_node(count, node_x, node_y, init_time)
                points.append(a)
                num_node = len(points)
                count += 1
            if num_node >= number:
                break
    return points

def generate_cars(num_cars,spd=0.5):
    """Creates vehicles."""
    car_set=[]
    count = 0
    while len(car_set) < num_cars:
        dc = deliver_car(count,spd)
        car_set.append(dc)
        count += 1
    return car_set

def list_chunk(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def compute_manhattan_distance_matrix(locations, factor = 1):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (np.abs(from_node.x - to_node.x)\
                +np.abs(from_node.y - to_node.y))*factor
    return distances

def get_solution(manager, routing, solution, factor=1):
    """Prints solution on console."""
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    dn_tsp_list = []
    distance_list = []
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        dn_tsp_list.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        distance_list.append(route_distance/factor)
        
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Route distance: {}\n'.format(route_distance/factor)

    return dn_tsp_list,route_distance/factor

def dn_tsp(node_list,depot=None,num_vehicles=1):
    """Solves TSP for a set of points from node_list."""
    # Instantiate the data problem.
    
    if depot is not None:
        node_list.insert(0, depot)
    
    manager = pywrapcp.RoutingIndexManager(len(node_list),num_vehicles, 0)
    
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_manhattan_distance_matrix(node_list,10000)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)


    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        a,distance = get_solution(manager, routing, solution,10000)
    
    dn_tsp_list = []
    for i in a[1:]:
        dn_tsp_list.append(node_list[i])
    
    return distance,dn_tsp_list


def assign_job_to_car(car, serve_list, depot_node, cur_time, service_time_fct,f=None):
    """Assign waiting delivery jobs to vehicles"""
    car.state = 1
    car.service_list = serve_list
    ("\n"+'car.service_list: '+ str(len(car.service_list)))

    distance = np.abs(depot_node.x-car.service_list[0].x)+np.abs(depot_node.y-car.service_list[0].y)

    distance_list=[distance]

    for i in range(len(car.service_list)-1):
        if i == 0:
            car.service_list[i].finish_time = max(cur_time, car.available_time) + distance/car.speed + \
            service_time_fct(car.service_list[i].y,car.service_list[i].x)
        else:
            car.service_list[i].finish_time = car.service_list[i-1].finish_time + distance/car.speed + \
            service_time_fct(car.service_list[i].y,car.service_list[i].x)
        car.service_list[i].wait_time = car.service_list[i].finish_time-car.service_list[i].initial_time
        distance = np.abs(car.service_list[i].x-car.service_list[i+1].x)\
        +np.abs(car.service_list[i].y-car.service_list[i+1].y)
        distance_list.append(distance)

    if len(car.service_list)>1:
        car.service_list[-1].finish_time = car.service_list[-2].finish_time + distance/car.speed + \
                service_time_fct(car.service_list[-1].y,car.service_list[-1].x)
        car.service_list[-1].wait_time = car.service_list[-1].finish_time-car.service_list[-1].initial_time
    else:
        car.service_list[-1].finish_time = max(cur_time, car.available_time) + distance/car.speed + \
                service_time_fct(car.service_list[-1].y,car.service_list[-1].x)
        car.service_list[-1].wait_time = car.service_list[-1].finish_time - car.service_list[-1].initial_time

    distance = np.abs(car.service_list[-1].x-depot_node.x)\
        +np.abs(car.service_list[-1].y-depot_node.y)
    distance_list.append(distance)
    car.available_time = car.service_list[-1].finish_time + distance/car.speed
    #f.write("\n"+'car: '+ str( car.index))
    #f.write("\n"+'car available time: '+ str(car.available_time))
    #f.write("\n"+"distance list: "+ str( distance_list))
    #f.write("\n"+'======================================')
    
def simulation(polygon, polygon_list, depot_node, p_xy, service_time_fct, num_cars, car_capacity, spd, num_batch, lam, rho, folder, model = "new", T = 2000, seed = 10):
    """Simulate the region partitioning policy based on the partition (polygon_list)
    given arrival rate (lam), workload (rho), and batch size (num_batch); running for T periods."""
    # note T should be set according to the workload, a higher workload requires a larger T for a stable estimation
    fw = open(folder + ("Output_rho%s_batch%s_%s_seed%s.txt" % (int(rho*100), num_batch, model, seed)), 'w')
    num_order_set = num_batch
    cars_set = generate_cars(num_cars,spd)
    
    all_order_list = [[] for i in range(num_cars)]
    wait_package_list = [[] for i in range(num_cars)]
    wait_order_list = [[] for i in range(num_cars)]
    total_dn = [0 for i in range(num_cars)]
    cdf_x_list, interpolate_cdf_x = generate_x_cdf(polygon, p_xy)
    # set the seed for simulations
    np.random.seed(seed)
    for cur_time in range(T):
        num_order = np.random.poisson(lam, 1)[0]
        node_location = generate_random_node(num_order, polygon, p_xy, cur_time, sum(total_dn), cdf_x_list, interpolate_cdf_x)

        for i, poly in enumerate(polygon_list):
            tmp_node_list = []

            ## Get order's location
            count1 = 0
            num_node = len(node_location)
            tmp = []
            for idx in range(num_node):
                node = node_location[idx]
                pnt = sg.Point(node.x,node.y)
                if poly.buffer(-1e-6).contains(pnt):
                    tmp_node_list.append(node)
                    total_dn[i] += 1
                    count1 += 1
                    tmp.append(idx)

            # remove all visited node
            node_location = [element for (idx,element) in enumerate(node_location) if idx not in tmp]
            
            wait_order_list[i].extend(tmp_node_list)
            all_order_list[i].extend(tmp_node_list)
            assert(len(all_order_list[i])==total_dn[i] )

            if len(wait_order_list[i]) >= num_order_set:
                serve_order_list = wait_order_list[i][:num_order_set]
                wait_order_list[i] = wait_order_list[i][num_order_set:]
                distance,a = dn_tsp(serve_order_list,depot_node)
                job_list = list_chunk(a, math.ceil(car_capacity))
                wait_package_list[i].extend(job_list)

            if (cars_set[i].state==1) & (cars_set[i].available_time < cur_time+1):
                cars_set[i].state = 0

            if (cars_set[i].state==0) & (len(wait_package_list[i])!=0):
                serve_list = wait_package_list[i].pop(0)
                assign_job_to_car(cars_set[i], serve_list, depot_node, cur_time, service_time_fct, fw)
     
    Average_waiting_time = []
    waiting_time_node_list =[]
    initial_time_node_list =[]
    finish_time_node_list =[]
    Average_waiting_time_burned = []
    count_burned_list = []
    data_final = {}
    data_final['wait_time'] = []
    sum_total_wait_time = 0
    sum_count = 0
    burned_sum_wait_time = 0
    burned_sum_count = 0
    for i in range(len(polygon_list)):
        total_wait_time = 0
        count = 0
        burned_wait_time = 0
        burned_count = 0
        tmp_list0 = []
        tmp_list1 = []
        tmp_list2 = []
        for node in all_order_list[i]:
            if node.finish_time is not None:
                count += 1
                sum_count += 1
                node.wait_time = node.finish_time - node.initial_time
                sum_total_wait_time += node.wait_time
                total_wait_time += node.wait_time
                tmp_list0.append([node.wait_time,node.initial_time,node.finish_time])

                if node.initial_time > 0.4 * T:   # a burn-in parameter
                    burned_count += 1
                    burned_sum_count += 1
                    burned_wait_time += node.wait_time
                    burned_sum_wait_time += node.wait_time
                    waiting_time_node_list.append(node.wait_time)
        tmp_list0.sort(key = lambda i: i[1])
        tmp_list2 = [y for x, y, z in tmp_list0]
        tmp_list0 = [x for x, y, z in tmp_list0]

        Average_waiting_time.append(float(total_wait_time/count))
        Average_waiting_time_burned.append(float(burned_wait_time/(burned_count+1e-4)))
        count_burned_list.append(int(burned_count))
        data_final['wait_time'].append({'index': i, 'wait_time': float(total_wait_time/count), 'burned_wait_time': float(burned_wait_time/(burned_count+1e-4)), 'burned_count': int(burned_count)})
    waiting_time_node_list = np.array(waiting_time_node_list)
    overall_wait_time = float(sum_total_wait_time/sum_count)
    overall_wait_time_burned = float(burned_sum_wait_time/burned_sum_count)
    worst_wait_time_burned = max(Average_waiting_time_burned)
    fw.write("\n"+"Average waiting time: "+ str(Average_waiting_time))
    print("\n"+"Average waiting time: "+ str(Average_waiting_time))
    fw.write("\n"+"Burned Average waiting time: "+ str(Average_waiting_time_burned))
    print("\n"+"Burned Average waiting time: "+ str(Average_waiting_time_burned))
    fw.write("\n"+"Burned Worst waiting time: "+ str(worst_wait_time_burned))
    print("\n"+"Burned Worst waiting time: "+ str(worst_wait_time_burned))
    fw.write("\n"+"Overall average waiting time: "+ str(overall_wait_time))
    print("\n"+"Overall average waiting time: "+ str(overall_wait_time))
    fw.write("\n"+"Burned Overall average waiting time: "+ str(overall_wait_time_burned))
    print("\n"+"Burned Overall average waiting time: "+ str(overall_wait_time_burned))
    fw.write("\n" + "Burned waiting time std: " + str(np.std(waiting_time_node_list)))
    print("\n" + "Burned waiting time std: " + str(np.std(waiting_time_node_list)))
    fw.close()
    return overall_wait_time_burned, worst_wait_time_burned
        
        
