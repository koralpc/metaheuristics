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
        for i in range(5): # skip first 5
            next(file)
        for line in file:
            strs = line.split("\t")
            city_coordinates.append([int(strs[0]),int(strs[1])])
            release_date.append(int(strs[-1]))

    num_vertex = len(release_date)
    city_coordinates = np.array(city_coordinates)
    release_date = np.array(release_date)
    distance_matrix = calculate_distance(city_coordinates)
    return num_vertex,distance_matrix,release_date


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
    trip = list(range(1,vertex_num))
    trip = trip_optimize([trip], distance_matrix,release_date)
    trip_cost = compute_cost(trip, distance_matrix, release_date)
    print("[Initialization] Trip cost: ",trip_cost)
    best_trip = trip
    best_cost = trip_cost
    best_unchange = 0
    T = T0
    count = 0
    while (T > Ts):
        for i in range(L):
            # early stopping if best solution unchanged long time
            best_unchange += 1
            if best_unchange >= Max_unchanged:
                print("[Early stopping] Best solution not improved in %s iterations." %Max_unchanged)
                return best_trip,best_cost

            new_trip = create_newtrip(trip, vertex_num, distance_matrix)
            new_trip = schedule_route(new_trip,release_date)
            cost_new = compute_cost(new_trip,distance_matrix,release_date)

            print("[Tempareture%.2f Round%s] New Trip cost: " %(T,count),cost_new, "Current trip cost: ",trip_cost)
            print("[Tempareture%.2f Round%s] New Trip: " %(T,count), new_trip)
            df = trip_cost - cost_new

            if df >= 0:  # better trip: always accept
                trip = new_trip
                trip_cost = cost_new
                if cost_new < best_cost:  # better than current best one
                    best_trip = new_trip
                    best_cost = cost_new
                    best_unchange = 0
            else:  # worse trip: accept worse trip with a decreasing probability
                if np.random.rand() <= math.exp(df/T):
                    trip = new_trip
                    trip_cost = cost_new

        T = T * q  # annealing, decreasing temperature
        count += 1
    print("[End iteration] end trip: ", trip)
    print("[End iteration] end trip cost: ", trip_cost)
    return best_trip,best_cost


if __name__ == '__main__':
    T0 = 100  # initial temperature
    Ts = 0.1  # stop temperature
    q = 0.7  # annealing, decrease rate
    L = 20  # link length, iteration number for each temperature
    Max_unchanged = 150  # number of iteration best solution not change

    vertex_num, distance_matrix, release_date = load_dataset("../datasets/10/C101_1.dat")
    start_time = time.time()
    best_trip, best_cost = sa_iteration(vertex_num, distance_matrix, release_date)
    print("-----------------Result--------------------")
    print("Algorithm running %s seconds" % (time.time() - start_time))

    print("Best solution, completion time is: ", best_cost)
    print("Best trip is: ",best_trip)
    print("Release date: \n",release_date)
    print("Distance matrix: \n",distance_matrix)
