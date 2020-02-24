import math
import time
import random
import copy

import numpy as np
from sko.ACA import ACA_TSP

def create_newtrip(trip, vertex_num, distance_matrix):
    # update current trip, using some operator
    # 1. multiple operator: vertex relocation, depot shift, depot insertion, depot removal √
    # 2. ruin and recreate

    if len(trip) > vertex_num * 0.7:
        prob = [0.25, 0.25, 0.5]
    elif len(trip) <= 2:
        prob = [0.3, 0.4, 0.3]
    else:
        prob = None

    operator = np.random.choice([1, 2, 3], size=1, p=prob)[0]
    new_trip = copy.deepcopy(trip)
    if operator == 1:  # relocate a vertex to another route contains neighbor
        new_trip = vertex_relocate(new_trip, vertex_num, distance_matrix)
    elif operator == 2:  # insert a depot in a route
        new_trip = route_split(new_trip, distance_matrix, vertex_num)
    elif operator == 3:  # remove a depot, combine two route
        new_trip = route_combine(new_trip, distance_matrix)

    return new_trip


def vertex_relocate(trip, vertex_num, distance_matrix):
    if len(trip) <= 1:
        return trip
    vertex = np.random.randint(1, vertex_num)
    origin_route = 0
    # vertex_dis = distance_matrix[vertex]
    # neighbors = vertex_dis.argsort()[-int(round(0.5 * vertex_num)):]
    # possible_routes = []  # (route, neighbor)
    for i in range(len(trip)):
        route = trip[i]
        if vertex in route:
            origin_route = i
            if len(route) <= 1:
                del trip[i]
            else:
                route.remove(vertex)  # mutable
                trip[origin_route] = single_route_optimize(trip[origin_route], distance_matrix)
            break

    des_route = np.random.randint(0, len(trip))
    if des_route == origin_route:
        des_route = np.random.randint(0, len(trip))
    trip[des_route].append(vertex)
    # optimize changed route
    trip[des_route] = single_route_optimize(trip[des_route], distance_matrix)
    return trip

def route_split(trip, distance_matrix, vertex_num):
    # TODO: heuristic: remove latest k as new trip
    if len(trip) >= vertex_num:
        return trip
    rand_index = np.random.randint(0, len(trip))
    while len(trip[rand_index]) <= 1:
        rand_index = np.random.randint(0, len(trip))
    rand_route = trip[rand_index]
    # choose a random position to split the route
    rand_pos = np.random.randint(1, len(rand_route))
    sub_route = rand_route[rand_pos:]  # second part
    sub_route = single_route_optimize(sub_route, distance_matrix)

    # trip[rand_index] = rand_route[:rand_pos]
    trip[rand_index] = single_route_optimize(rand_route[:rand_pos], distance_matrix)
    trip.insert(rand_index + 1, sub_route)
    return trip

def route_combine(trip, distance_matrix):
    # TODO: heuristic, combine close time or distance route
    if len(trip) > 1:
        depot_pos = np.random.randint(1, len(trip))
        route1 = trip[depot_pos - 1]
        route2 = trip[depot_pos]
        # trip[depot_pos - 1] = route1 + route2
        trip[depot_pos - 1] = single_route_optimize(route1 + route2, distance_matrix)
        del trip[depot_pos]
    return trip


def local_optimize(trip):
    pass


def schedule_route(trip, release_date):
    # schedule trip routes by route start time
    start_time = []
    for route in trip:
        start_time.append(release_date[route].max())

    index = np.argsort(start_time)
    new_trip = []
    for i in index:
        new_trip.append(trip[i])
    trip = new_trip
    return trip


def trip_optimize(trip, distance_matrix, release_date):
    """
    Optimize each route, pure tsp solved by SA
    :param trip:
    :param distance_matrix:
    :return:
    """
    # optimize each route
    optimal_trip = []
    for route in trip:
        new_route = single_route_optimize(route, distance_matrix)
        optimal_trip.append(new_route)

    optimal_trip = schedule_route(optimal_trip,release_date)
    print("Old trip:", trip)
    print("Optimized trip: ", optimal_trip)

    return optimal_trip


def single_route_optimize(route, distance_matrix):
    def cal_distance(routine):
        num_points, = routine.shape
        return sum([sub_dis_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    if len(route) < 2:
        return route

    sub_dis_matrix = []  # distance matrix for this route vertices
    route = [0] + route  # add depot to complete whole route
    for v in route:
        sub_dis_matrix.append(distance_matrix[v][route])
    sub_dis_matrix = np.array(sub_dis_matrix)
    num_points = len(route)
    aca = ACA_TSP(func=cal_distance,n_dim=num_points,size_pop=int(num_points),max_iter=100,distance_matrix=sub_dis_matrix)
    best_route_index, best_distance = aca.run()
    best_route_index = best_route_index.tolist()

    # construct route without depot
    depot_index = best_route_index.index(0)  # 0 is depot in route
    k = depot_index + 1  # counter
    new_route = []
    while k < len(best_route_index):  # add right part
        new_route.append(route[best_route_index[k]])
        k += 1
    if depot_index > 0:  # has left part
        k = 0
        while k < depot_index:
            new_route.append(route[best_route_index[k]])
            k += 1
    return new_route


def compare_trip(old_trip, new_trip):
    # 1. compare roughly, just by starting time saving and detour saving  [faster]

    pass
