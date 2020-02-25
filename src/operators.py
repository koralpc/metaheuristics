import copy

import numpy as np
from sko.ACA import ACA_TSP


def create_newtrip(trip, vertex_num, distance_matrix, heuristic_matrix, release_date):
    # update current trip, using some operator
    # 1. multiple operator: vertex relocation, depot shift, depot insertion, depot removal √
    # 2. ruin and recreate

    if len(trip) == 1:
        prob = [0, 1, 0, 0]
    elif len(trip) >= vertex_num - 1:
        prob = [0, 0, 0.5, 0.5]
    elif len(trip) <= 2:
        prob = [0.3, 0.5, 0.1, 0.1]
    elif len(trip) > vertex_num * 0.6:
        prob = [0.2, 0.2, 0.3, 0.3]
    else:
        prob = [0.3, 0.3, 0.1, 0.3]

    operator = np.random.choice([1, 2, 3, 4], size=1, p=prob)[0]
    new_trip = copy.deepcopy(trip)
    if operator == 1:
        new_trip = vertex_relocate(new_trip, vertex_num, heuristic_matrix)
    elif operator == 2:
        new_trip = route_split(new_trip, vertex_num, heuristic_matrix)
    elif operator == 3:
        new_trip = merge_single(new_trip, heuristic_matrix)
    elif operator == 4:
        new_trip = route_combine(new_trip, heuristic_matrix)

    return new_trip


def vertex_relocate(trip, vertex_num, heuristic_marix):
    if len(trip) <= 1:
        return route_split(trip, vertex_num, heuristic_marix)

    vertex = np.random.randint(1, vertex_num)
    ordered = list(np.argsort(heuristic_marix[vertex]))
    ordered.remove(0)
    neighbors_time = ordered[:3]

    records = []  # (route_index,neighbor_index)
    origin_route = 0
    for i in range(len(trip)):
        route = trip[i]
        if vertex in route:
            origin_route = i
            # trip[origin_route] = single_route_optimize(trip[origin_route], distance_matrix)
            continue
        else:
            for n in range(len(neighbors_time)):
                if neighbors_time[n] in route:
                    records.append((i, n))
                    break  # find closest neighbor in this route
    if len(records) > 0:
        records.sort(key=lambda tup: tup[1])
        des = records[0]
        trip[des[0]].append(vertex)
        trip[origin_route].remove(vertex)
        trip = list(filter(lambda r: r != [], trip))  # remove empty
    else:
        trip = route_split(trip, vertex_num, heuristic_marix)
    return trip


def route_split(trip, vertex_num, heuristic_matrix):
    if len(trip) >= vertex_num - 1:
        return merge_single(trip, heuristic_matrix)

    # split a random route
    rand_index = np.random.randint(0, len(trip))
    while len(trip[rand_index]) <= 1:
        rand_index = np.random.randint(0, len(trip))
    route = trip[rand_index]
    vertex_index = np.random.randint(0, len(route))  # random choose a vertex
    k = round(len(route) * np.random.rand())  # split portion
    new_route = [route[vertex_index]]

    sub_matrix = heuristic_matrix[route][:, route]
    for _ in range(k):
        vertex_index = np.argmax(sub_matrix[vertex_index])
        new_route.append(route[vertex_index])

    new_route = list(set(new_route))
    trip[rand_index] = list(set(route) - set(new_route))
    trip.insert(rand_index + 1, new_route)
    trip = list(filter(lambda r: r != [], trip))  # remove empty
    return trip


def route_combine(trip, heuristic_matrix):
    # combine by release date or distance
    route_index = np.random.randint(0, len(trip))
    route = trip[route_index].copy()
    for v in route:
        neighbor = np.argmax(heuristic_matrix[v][1:]) + 1
        for r in trip:
            if neighbor in r and r != route:
                idx = r.index(neighbor)
                r.insert(idx, v)
                trip[route_index].remove(v)
                break
    trip = list(filter(lambda r: len(r) >= 1, trip))  # remove empty route
    return trip


def merge_single(trip, heuristic_matrix):
    # merge single vertex route
    for route in trip:
        if len(route) <= 1:
            for v in route:
                neighbor = np.argmax(heuristic_matrix[v][1:]) + 1
                for r in trip:
                    if neighbor in r:
                        idx = r.index(neighbor)
                        r.insert(idx, v)
                        route.remove(v)
                        break
    trip = list(filter(lambda r: len(r) >= 1, trip))  # remove empty route
    return trip


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

    optimal_trip = schedule_route(optimal_trip, release_date)
    return optimal_trip


def single_route_optimize(route, distance_matrix):
    def cal_distance(routine):
        num_points, = routine.shape
        return sum([sub_dis_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    if len(route) <= 2:
        return route

    route = [0] + route
    sub_dis_matrix = distance_matrix[route][:, route]
    num_points = len(route)
    aca = ACA_TSP(func=cal_distance, n_dim=num_points, size_pop=int(num_points * 2), max_iter=150,
                  distance_matrix=sub_dis_matrix)
    best_route_index, best_distance = aca.run()
    best_route_index = best_route_index.tolist()

    # construct route without depot
    depot_index = best_route_index.index(0)  # 0 is depot in route
    route = np.array(route)
    route = route[best_route_index[depot_index + 1:]].tolist() + route[best_route_index[:depot_index][::-1]].tolist()
    return route


def fast_optimize_trip(trip, distance_matrix, release_date):
    optimal_trip = []
    for r in trip:
        route = fast_optimize(r, distance_matrix)
        optimal_trip.append(route)
    optimal_trip = schedule_route(optimal_trip, release_date)
    return optimal_trip


def fast_optimize(route, distance_matrix):
    # 2-op local optimize
    if len(route) <= 2:  # at least 3 vertices
        return route
    route = [0] + route
    num_vertex = len(route)
    maxlen = min(5, int(num_vertex / 2))
    for l in range(2, maxlen + 1):
        for start in range(num_vertex):
            end = (start + l - 1) % num_vertex
            start1 = (start + num_vertex - 1) % num_vertex
            end1 = (end + 1) % num_vertex
            gain = distance_matrix[route[start], route[start1]] + distance_matrix[route[end], route[end1]] - \
                   distance_matrix[route[start], route[end1]] - distance_matrix[route[end], route[start1]]

            if gain > 0:
                temp = route[end1]
                route[end1] = route[start1]
                route[start1] = temp

    # construct route without depot
    depot_index = route.index(0)  # 0 is depot in route
    route = route[depot_index + 1:] + route[:depot_index][::-1]
    return route


def compare_trip(old_trip, new_trip, release_date, vertex_num, distance_matrix):
    # compare roughly, just by starting time saving and detour saving  [faster]
    # avg start time
    # e.g. r1 start_time=5, 3 node; r2 start_time=10, 2 node; then avg start_time=(5*3+10*2)/5 = 7
    old_sum = sum([release_date[r].max() * len(r) for r in old_trip])
    new_sum = sum([release_date[r].max() * len(r) for r in new_trip])
    time_dif = old_sum / vertex_num - new_sum / vertex_num

    # compute route distance
    def get_distance(trip):
        total_distance = 0
        for route in trip:
            route = [0] + route + [0]  # start and end at depot
            for k in range(0, len(route) - 1):
                total_distance += distance_matrix[route[k], route[k + 1]]
        return total_distance

    old_distance = get_distance(old_trip)
    new_distance = get_distance(new_trip)
    distance_dif = old_distance - new_distance

    return time_dif + distance_dif
