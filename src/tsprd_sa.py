from scipy import spatial

from src.operators import *


def calculate_distance(city_coordinate_matrix):
    # Calculate distances of the paths between each city.
    distance_matrix = spatial.distance.cdist(city_coordinate_matrix, city_coordinate_matrix, metric='euclidean')
    return distance_matrix


def load_dataset(file_path):
    city_coordinates = []
    release_date = []
    with open(file_path) as file:
        for i in range(5):  # skip first 5
            next(file)
        for line in file:
            strs = line.split("\t")
            city_coordinates.append([int(strs[0]), int(strs[1])])
            release_date.append(int(strs[-1]))

    num_vertex = len(release_date)
    city_coordinates = np.array(city_coordinates)
    release_date = np.array(release_date)
    distance_matrix = calculate_distance(city_coordinates)
    return num_vertex, distance_matrix, release_date


def compute_cost(trip, distance_matrix, release_date):
    """
    Compute the whole trip time
    :param trip: a trip, contains routes
    :param distance_matrix: 2-d numpy array
    :param release_date: 1-d numpy array
    :return: time cost
    """
    total_time = 0
    for route in trip:
        # waiting time,
        max_release_time = release_date[route].max()  # route can only start later than maximum release date
        if total_time < max_release_time:
            total_time += max_release_time - total_time  # time to wait at depot
        # traveling time
        route = [0] + route + [0]  # start and end at depot
        for k in range(0, len(route) - 1):
            total_time += distance_matrix[route[k], route[k + 1]]

    return total_time


def sa_iteration(vertex_num, distance_matrix, release_date):
    trip = []
    for i in range(1, vertex_num):
        trip.append([i])
    trip_cost = compute_cost(trip, distance_matrix, release_date)
    print("[Initialization] trip: ", trip)
    print("[Initialization] Benchmark trip cost: ", trip_cost)
    best_trip = trip
    best_cost = trip_cost
    unchange_times = 0
    T = T0
    count = 0
    acceptable_criterion = trip_cost * 0.3
    while (T > Ts):
        for i in range(L):
            # early stopping if best solution unchanged long time
            if unchange_times > Max_unchanged:
                print("[Early stopping] Solution not improved in %s iterations." % Max_unchanged)
                best_trip = trip_optimize(best_trip, distance_matrix, release_date)
                best_cost = compute_cost(best_trip, distance_matrix, release_date)
                print("[Early stopping] best trip: ", best_trip)
                print("[Early stopping] best trip cost: ", best_cost)
                trip = trip_optimize(trip, distance_matrix, release_date)
                trip_cost = compute_cost(trip, distance_matrix, release_date)
                print("[Early stopping] end trip: ", trip)
                print("[Early stopping] end trip cost: ", trip_cost)
                if trip_cost < best_cost:
                    best_trip = trip
                    best_cost = trip_cost
                return best_trip, best_cost

            new_trip = create_newtrip(trip, vertex_num, distance_matrix, heuristic_matrix, release_date)
            # new_trip = fast_optimize_trip(new_trip,distance_matrix,release_date)
            new_trip = schedule_route(new_trip, release_date)
            print("[Tempareture%.2f Round%s] New Trip: " % (T, count), new_trip)

            cost_new = compute_cost(new_trip, distance_matrix, release_date)
            print("[Tempareture%.2f Round%s] New Trip cost: " % (T, count), cost_new, "Current trip cost: ", trip_cost)

            df = compare_trip(trip, new_trip, release_date, vertex_num, distance_matrix)  # avg_starttime old-new
            dif = trip_cost - cost_new
            if dif >= 0 or df > 0:  # better trip: always accept
                trip = new_trip
                trip_cost = cost_new
                unchange_times = 0
                if cost_new < best_cost:  # better than current best one
                    best_trip = new_trip
                    best_cost = cost_new
            else:  # worse trip: accept worse trip with a decreasing probability
                unchange_times += 1
                if np.random.rand() <= math.exp(dif / T):
                    trip = new_trip
                    trip_cost = cost_new
        T = T * q  # annealing, decreasing temperature
        count += 1

    best_trip = trip_optimize(best_trip, distance_matrix, release_date)
    best_cost = compute_cost(best_trip, distance_matrix, release_date)
    print("[End iteration] best trip: ", best_trip)
    print("[End iteration] best trip cost: ", best_cost)
    trip = trip_optimize(trip, distance_matrix, release_date)
    trip_cost = compute_cost(trip, distance_matrix, release_date)
    print("[End iteration] end trip: ", trip)
    print("[End iteration] end trip cost: ", trip_cost)
    if trip_cost < best_cost:
        best_trip = trip
        best_cost = trip_cost
    return best_trip, best_cost


if __name__ == '__main__':
    T0 = 100  # initial temperature
    Ts = 0.01  # stop temperature
    q = 0.9  # annealing, decrease rate
    L = 200  # link length, iteration number for each temperature
    Max_unchanged = 300  # number of iteration best solution not change

    vertex_num, distance_matrix, release_date = load_dataset("../datasets/20/C101_1.dat")
    heuristic_matrix = copy.deepcopy(distance_matrix)
    max_date = release_date.max()
    for (i, j), x in np.ndenumerate(distance_matrix):
        if i < j != 0:
            heuristic_matrix[i, j] = (10 / x) + (max_date - abs(release_date[i] - release_date[j])) / max_date
        elif i > j:
            heuristic_matrix[i, j] = heuristic_matrix[j, i]
        else:
            heuristic_matrix[i, j] = 0
    start_time = time.time()
    best_trip, best_cost = sa_iteration(vertex_num, distance_matrix, release_date)
    print("-----------------Result--------------------")
    print("Algorithm running %s seconds" % (time.time() - start_time))

    print("Best solution, completion time is: ", best_cost)
    print("Best trip is: ",
          best_trip)  # print("Release date: \n", release_date)  # print("Distance matrix: \n", distance_matrix)
