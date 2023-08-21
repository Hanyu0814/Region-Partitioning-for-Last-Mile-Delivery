import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import integrate
import datetime
import shapely.geometry as sg
import shapely
import random
from scipy import interpolate
from base import *
import math
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from simulation import *
from Two_Partition import *
from Three_Partition import *


def equitable_partition(p, f, g, L1, L2, n, tol, epsilon_theta_two, epsilon_theta_three, epsilon_r, epsilon_integral):
    """Iteratively partition region p into n equitable subregions wrt f and g, with targets L1 and L2."""
    Vertices1 = poly_vertices(p);

    m1 = float(intpoly(f, Vertices1[:, 0], Vertices1[:, 1], epsilon_integral));
    m2 = float(intpoly(g, Vertices1[:, 0], Vertices1[:, 1], epsilon_integral));

    print(m1, m2, n)
    if (m1 < L1 - tol) or (m2 < L2 - tol):
        print(
            "error: Measure of cutted area have already smaller than target measure value, please check tolerance and epsilon.")
        return -1

    elif (abs(L1 - m1) <= tol) & (abs(L2 - m2) <= tol):
        R = [m1, m2]
        V = Vertices1
        V = V[None, :, :]

    else:
        three_flag = False;
        # Decide three partition or two partition.
        if (n % 2) != 0:
            g_value = gvalue(p, n, f, g, epsilon_r, epsilon_integral)
            gi = g_value[0]
            print("gi: ", gi)
            boundary = g_value[1]
            if len(gi) == 2:
                n1 = gi[0]
                n2 = gi[1]
                target_m1 = n1 / n * m1
                target_m2 = n1 / n * m2
                [C1, C2] = ham_sandwich_cut(p, f, g, target_m1, target_m2, epsilon_theta_two, epsilon_r,
                                            epsilon_integral);  # find  C1 and C2
            else:
                three_flag = True;
                n1 = gi[0];
                n2 = gi[1];
                n3 = gi[2]
                l1 = min(boundary[0], boundary[1]);
                l2 = max(boundary[0], boundary[1]);
                for i in range(3):
                    n_temp = gi[i]
                    target_m1 = n_temp / n * m1
                    target_m2 = n_temp / n * m2
                    [C1, C2] = ham_sandwich_cut(p, f, g, target_m1, target_m2, epsilon_theta_two, epsilon_r,
                                                epsilon_integral);
                    if C1 is not None:
                        three_flag = False
                        n1 = n_temp
                        n2 = n - n1
                        break;
                if three_flag is True:
                    [C1, C2, C3] = three_cut(p, f, g, n1, n2, n3, l1, l2, epsilon_theta_three,
                                             epsilon_integral);  # find  C1 and C2
        else:
            n1 = n / 2
            n2 = n / 2
            gi = [n1, n2]
            print("gi: ", gi)
            target_m1 = n1 / n * m1
            target_m2 = n1 / n * m2
            [C1, C2] = ham_sandwich_cut(p, f, g, target_m1, target_m2, epsilon_theta_two, epsilon_r,
                                        epsilon_integral);  # find  C1 and C2

        poly_C1 = sp.Polygon(*(tuple(zip(C1[:, 0], C1[:, 1]))))
        poly_C2 = sp.Polygon(*(tuple(zip(C2[:, 0], C2[:, 1]))))

        [R1, V1] = equitable_partition(poly_C1, f, g, L1, L2, n1, tol, epsilon_theta_two, epsilon_theta_three,
                                       epsilon_r, epsilon_integral);
        [R2, V2] = equitable_partition(poly_C2, f, g, L1, L2, n2, tol, epsilon_theta_two, epsilon_theta_three,
                                       epsilon_r, epsilon_integral);
        if three_flag:  # len(gi) == 3:
            poly_C3 = sp.Polygon(*(tuple(zip(C3[:, 0], C3[:, 1]))))
            [R3, V3] = equitable_partition(poly_C3, f, g, L1, L2, n3, tol, epsilon_theta_two, epsilon_theta_three,
                                           epsilon_r, epsilon_integral);
        R = [R1, R2];
        V = [];
        for i in range(len(V1)):
            V.append(V1[i]);
        for j in range(len(V2)):
            V.append(V2[j])
        if three_flag:  # len(gi) == 3:
            R.append(R3)
            for k in range(len(V3)):
                V.append(V3[k])
    return R, V


def wedge_partition(f, polygon, depot_node, num_cars, target=None):
    """Wedge partition function (cuts into num_car wedges): this can also be used in simulating the flexible policy."""
    xs, ys = polygon.exterior.xy
    epsilon_integral = 0.00001

    total_L1 = float(intpoly(f, xs, ys, epsilon_integral));
    target = total_L1 / num_cars

    R = 2
    epsilon_r = 0.0001
    theta0 = 0
    poly_list = []
    for i in range(num_cars - 1):
        theta1 = theta0
        theta2 = 2 * np.pi
        mark_x = depot_node.x + math.cos(theta0 + 0.001) * R * 0.01
        mark_y = depot_node.y + math.sin(theta0 + 0.001) * R * 0.01
        mark_node = sg.Point(mark_x, mark_y)

        # Binary search to find the wedge angle.
        while True:
            theta3 = (theta1 + theta2) / 2

            x_tmp = depot_node.x + math.cos(theta3) * R
            y_tmp = depot_node.y + math.sin(theta3) * R

            if i == 0:
                line1 = sg.LineString([(depot_node.x, depot_node.y), (depot_node.x + R, depot_node.y)])
            line2 = sg.LineString([(depot_node.x, depot_node.y), (x_tmp, y_tmp)])

            if i == 0:
                new_polygons = polygon.difference(line1.buffer(1e-5)).difference(line2.buffer(1e-5))
            else:
                new_polygons = polygon.difference(line2.buffer(1e-5))


            count = 0
            for poly in new_polygons.geoms:
                xs, ys = poly.exterior.xy

                if poly.contains(mark_node):
                    cutted_poly = poly
                    count += 1
                    xs, ys = poly.exterior.xy
                    int_poly = abs(intpoly(f, xs, ys, epsilon_integral))

                else:
                    remain_poly = poly
            if count != 1:
                raise MyError('Error.')

            if int_poly < target - epsilon_r:  # if m1(inter) < m1(all area)/2 move up y1
                theta1 = theta3
            elif int_poly > target + epsilon_r:  # if m1(inter) < m1(all area)/2 move down y2
                theta2 = theta3
            else:
                poly_list.append(cutted_poly)
                if i == num_cars - 2:
                    poly_list.append(remain_poly)
                break
        polygon = remain_poly
        theta0 = theta3
    for poly in poly_list:
        xs, ys = poly.exterior.xy
        plt.plot(xs, ys, color='black')

    plt.show()
    return poly_list

##################################
####### Partition function #######
##################################

def partition_region(f, g, num_car, p):
    """Given measures f and g, partition the region p into n subregions. """
    ### Input parameters ###
    # f=lambda y,x: measure function 1
    # g=lambda y,x: measure function 2
    # p: polygon
    n = num_car  # number of parts to divide
    epsilon_r = 0.0005  # 0.002 # error of inner loop for r
    epsilon_theta_two = 0.003  # 0.003
    epsilon_theta_three = 0.003  # 0.003 # error of outer loop for theta
    tol = 0.006  # 0.006 # total tolerance(error) of measure
    epsilon_integral = 0.00001  # error of outer loop for double integral

    p_Vertices1 = poly_vertices(p)

    total_L1 = float(intpoly(f, p_Vertices1[:, 0], p_Vertices1[:, 1], epsilon_integral))
    total_L2 = float(intpoly(g, p_Vertices1[:, 0], p_Vertices1[:, 1], epsilon_integral))

    # Run equitable partition function and track the time.
    [R, V] = equitable_partition(p, f, g, total_L1 / n, total_L2 / n, n, tol, epsilon_theta_two, epsilon_theta_three,
                                 epsilon_r, epsilon_integral)

    Poly_list = []
    for i in range(n):
        Poly = np.concatenate((V[i], V[i][0, :].reshape((1, 2))))
        plt.plot(Poly[:, 0], Poly[:, 1], color='black')
        tmp_polygon = sg.Polygon(Poly)
        Poly_list.append(tmp_polygon)
    plt.show()
    return Poly_list


#############################################################
# Simulation function
#############################################################
def sim_par_all_uniform(rho, lam, T, model, num_batch, seed, num_cars):
    """Simulation of partition based policy for all partitions with uniform distribution."""
    # note the number of simulated periods T should be set according to the workload
    # a higher workload requires a larger T for a stable estimation
    # rho: workload, lam: arrival rate, num_batch is a tuning parameter
    points = np.array([[-0.5, 0], [0, -0.5], [0.5, 0], [0, 0.5]])
    hull = ConvexHull(points)  # Find convex hull vertices
    hull_vertices = tuple(zip(points[hull.vertices, 0], points[hull.vertices, 1]))
    p = sp.Polygon(*hull_vertices)  # Build polygon
    polygon = sg.Polygon(hull_vertices)
    depot_node = polygon.centroid
    depot_node = demand_node(-1, depot_node.x, depot_node.y, -1)

    car_capacity = 10
    spd = 0.08
    if num_cars == 16:
        folder = "uniform/"
    elif num_cars == 20:
        folder = "uniform2/"
    else:
        folder = "/"

    # Servie time function
    service_time_fct = lambda y, x: 0

    f = lambda y, x: 1
    int_region = intpoly(f, polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1])
    p_xy = lambda y, x: 1 / int_region
    g1 = lambda y, x: np.sqrt(1 / int_region)  # measure function 1 (this is essentially the same as f)
    g2 = lambda y, x: 2 * (np.abs(x - depot_node.x) + np.abs(y - depot_node.y)) * 1 / polygon.area  # measure function 2
    int_region1 = intpoly(g1, polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1])
    int_region2 = intpoly(g2, polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1])

    # for nonuniform distribution test:
    #     service_time_fct = lambda y, x: np.exp(x) / 3
    #
    #     f = lambda y, x: 1 / (np.abs((x - 0.5) * (x + 0.5)) + 0.1)  # /(np.abs(0.4-x) + y**2 + 0.2)
    #     int_region = intpoly(f, polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1])
    #     p_xy = lambda y, x: (1 / (np.abs((x - 0.5) * (x + 0.5)) + 0.1)) / int_region
    #     g1 = lambda y, x: np.sqrt((1 / (np.abs((x - 0.5) * (x + 0.5)) + 0.1)) / int_region)  # measure function 1
    #     g2 = lambda y, x: (np.exp(x) * spd * car_capacity / 3 + 2 * (
    #             np.abs(x - depot_node.x) + np.abs(y - depot_node.y))) * (
    #                               1 / (np.abs((x - 0.5) * (x + 0.5)) + 0.1)) / int_region  # measure function 2
    # then apply the scaling below

    # scale to get a smaller relative error
    scale1 = (0.006 / (int_region / num_cars)) / 1e-2
    scale2 = (0.006 / (int_region2 / num_cars)) / 1e-2

    g1 = lambda y, x: scale1  # scaled measure function 1
    g2 = lambda y, x: (np.abs(x - depot_node.x) + np.abs(
        y - depot_node.y)) * scale2 / polygon.area  # scaled measure function 2
    if model == "new":
        polygon_list_uniform = partition_region(g1, g2, num_cars, p)
        avg, wrst = simulation(polygon, polygon_list_uniform, depot_node, p_xy, service_time_fct,
                               num_cars, car_capacity, spd, num_batch, lam, rho, folder, model, T=T, seed=seed)
    if model == "hp":
        polygon_list_hp = partition_region(g1, g1, num_cars, p)
        avg, wrst = simulation(polygon, polygon_list_hp, depot_node, p_xy, service_time_fct,
                               num_cars, car_capacity, spd, num_batch, lam, rho, folder, model, T=T, seed=seed)
    if model == "hp2":
        polygon_list_hp2 = partition_region(g2, g2, num_cars, p)
        avg, wrst = simulation(polygon, polygon_list_hp2, depot_node, p_xy, service_time_fct,
                               num_cars, car_capacity, spd, num_batch, lam, rho, folder, model, T=T, seed=seed)
    return avg, wrst


def mod_simulation(polygon, p, depot_node, f, p_xy, service_time_fct, num_cars, car_capacity, spd, num_batch, lam, fw,
                   T=2000, seed=10):
    """Simulation of benchmark flexible policies."""
    polygon_list = wedge_partition(f, polygon, depot_node, num_cars)

    num_order_set = num_batch
    cars_set = generate_cars(num_cars, spd)

    all_order_list = [[] for i in range(num_cars)]
    wait_package_list = []
    wait_order_list = [[] for i in range(num_cars)]
    total_dn = [0 for i in range(num_cars)]
    cdf_x_list, interpolate_cdf_x = generate_x_cdf(polygon, p_xy)
    # set the seed for simulations
    np.random.seed(seed)

    # Calculate delivery system by time steps.
    for cur_time in range(T):
        num_order = np.random.poisson(lam, 1)[0]
        node_location = generate_random_node(num_order, polygon, p_xy, cur_time, sum(total_dn), cdf_x_list,
                                             interpolate_cdf_x)
        # fw.write("\n"+'####################################################################################')
        # fw.write("\n"+"=============Initial state==============")
        # fw.write("\n"+"Current time: "+ str(cur_time)+ " Upcoming order: "+ str(num_order))

        for i, poly in enumerate(polygon_list):
            tmp_node_list = []

            ## Get order's location
            count1 = 0
            num_node = len(node_location)
            tmp = []
            for idx in range(num_node):
                node = node_location[idx]
                pnt = sg.Point(node.x, node.y)
                if poly.contains(pnt):
                    tmp_node_list.append(node)
                    total_dn[i] += 1
                    count1 += 1
                    tmp.append(idx)

            # remove all visited node
            node_location = [element for (idx, element) in enumerate(node_location) if idx not in tmp]

            wait_order_list[i].extend(tmp_node_list)
            all_order_list[i].extend(tmp_node_list)
            assert (len(all_order_list[i]) == total_dn[i])
            # fw.write("\n"+"======================================")
            # fw.write("\n"+"Region: "+ str(i)+" Upcoming Order: "+str(count1))
            # fw.write("\n"+"Queueing order(include new orders): "+str(len(wait_order_list[i]))+", Total order: "+str(len(all_order_list[i])) )

            if len(wait_order_list[i]) >= num_order_set:
                serve_order_list = wait_order_list[i][:num_order_set]
                wait_order_list[i] = wait_order_list[i][num_order_set:]

                distance, a = dn_tsp(serve_order_list, depot_node)
                job_list = list_chunk(a, math.ceil(car_capacity))
                wait_package_list.extend(job_list)

            #    fw.write("\n"+"Queueing job: "+ str(len(wait_package_list)))
            #     print('i:, ', i)
            #     print('serve_order_list, ',serve_order_list)
            #     print('wait_order_list', wait_order_list[i])
            #     print('wait_package_list, ', wait_package_list[i])
            for car_idx in cars_set:
                if (car_idx.state == 1) & (car_idx.available_time < cur_time + 1):
                    car_idx.state = 0

                if (car_idx.state == 0) & (len(wait_package_list) != 0):
                    serve_list = wait_package_list.pop(0)
                    assign_job_to_car(car_idx, serve_list, depot_node, cur_time, service_time_fct, fw)

    # Get output.
    Average_waiting_time = []
    Average_waiting_time_burned = []
    waiting_time_node_list = []
    sum_total_wait_time = 0
    sum_count = 0
    burned_sum_wait_time = 0
    burned_sum_count = 0
    tmp_list0 = []  # record all wait times
    for i in range(len(polygon_list)):
        total_wait_time = 0
        count = 0
        burned_wait_time = 0
        burned_count = 0
        for node in all_order_list[i]:
            if node.finish_time is not None:
                count += 1
                sum_count += 1
                node.wait_time = node.finish_time - node.initial_time
                total_wait_time += node.wait_time
                sum_total_wait_time += node.wait_time
                tmp_list0.append([node.wait_time, node.initial_time, node.finish_time])
                if node.initial_time > 0.4 * T:  # a burn-in parameter
                    burned_count += 1
                    burned_sum_count += 1
                    burned_wait_time += node.wait_time
                    burned_sum_wait_time += node.wait_time
                    waiting_time_node_list.append(node.wait_time)
        Average_waiting_time.append(total_wait_time / count)
        if burned_count > 0:
            Average_waiting_time_burned.append(burned_wait_time / burned_count)
    overall_wait_time = sum_total_wait_time / sum_count
    overall_wait_time_burned = burned_sum_wait_time / burned_sum_count
    waiting_time_node_list = np.array(waiting_time_node_list)
    tmp_list0.sort(key=lambda i: i[1])
    tmp_list1 = [y for x, y, z in tmp_list0]
    tmp_list0 = [x for x, y, z in tmp_list0]
    plt.plot(tmp_list1, tmp_list0)
    plt.show()
    # f.write("\n"+"Total waiting time: "+ str( total_wait_time))
    fw.write("\n" + "Average waiting time: " + str(Average_waiting_time))
    print("\n" + "Average waiting time: " + str(Average_waiting_time))
    fw.write("\n" + "Burned Average waiting time: " + str(Average_waiting_time_burned))
    print("\n" + "Burned Average waiting time: " + str(Average_waiting_time_burned))
    fw.write("\n" + "Overall average waiting time: " + str(overall_wait_time))
    print("\n" + "Overall average waiting time: " + str(overall_wait_time))
    fw.write("\n" + "Burned Overall average waiting time: " + str(overall_wait_time_burned))
    print("\n" + "Burned Overall average waiting time: " + str(overall_wait_time_burned))
    fw.write("\n" + "Burned waiting time std: " + str(np.std(waiting_time_node_list)))
    print("\n" + "Burned waiting time std: " + str(np.std(waiting_time_node_list)))
    fw.close()
    return overall_wait_time_burned