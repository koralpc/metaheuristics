#TSP-rd problem modelling
#Author : Koralp and Bingqian


## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from sko.ACA import ACA_TSP
from generate_dataset import *

## Methods

def main():
    def calculate_total_distance(route):
        num_points, = route.shape
        return sum([distance_matrix[route[i % num_points], route[(i + 1) % num_points]] for i in range(num_points)])

    def calculate_total_distance_with_rd(route):
        num_points, = route.shape
        total_dist = 0
        for i in range(num_points):
            total_dist += distance_matrix[route[i % num_points], route[(i + 1) % num_points]]
            if total_dist < release_dates[route[(i+1) % num_points]]:
                total_dist += release_dates[route[(i+1) % num_points]]
        return total_dist

    def plotRoute(city_x,city_y,route):
        fig, ax = plt.subplots()
        ax.scatter(city_x,city_y,color ='red')
        for idx in range(len(route)):
            ax.plot([city_x[route[idx % len(route)]],city_x[route[(idx+1) % len(route)]]],[city_y[route[idx % len(route)]],city_y[route[(idx+1) % len(route)]]],label =str(idx+1),color = 'blue')
            ax.annotate('Rd : {0} / City: {1}'.format(release_dates[idx],idx),(city_x[idx],city_y[idx]))
            #ax.annotate('Dist: {0:.3f}'.format(distance_matrix[route[idx % len(route)], route[(idx + 1) % len(route)]]),(np.mean([city_x[route[idx % len(route)]],city_x[route[(idx+1) % len(route)]]]),np.mean([city_y[route[idx % len(route)]],city_y[route[(idx+1) % len(route)]]])))
        plt.title('Total distance of this run is: {0:.4f}'.format(calculate_total_distance(route)))
        #plt.legend(loc ='lower right')
        #plt.show()

    def calculate_distance(city_coordinate_matrix):
        # Calculate distances of the paths between each city.
        distance_matrix = spatial.distance.cdist(city_coordinate_matrix, city_coordinate_matrix, metric='euclidean')
        return distance_matrix

    #Number of cities
    nCities = 6

    release_dates = generate_release_dates(nCities)
    city_coordinates = generate_city_coordinates(nCities)
    distance_matrix = calculate_distance(city_coordinates)

    # test:
    points = np.arange(nCities)  # generate index of points
    #calculate_total_distance(points)
    #calculate_total_distance_with_rd(points)


    city_x = [city_coordinates[i][0] for i in range(nCities)]
    city_y = [city_coordinates[i][1] for i in range(nCities)]

    plotRoute(city_x,city_y,points)

    #fig, ax = plt.subplots(1, 1)
    best_points_ = np.concatenate([points, [points[0]]])
    best_points_str = [str(p) for p in best_points_]
    best_points_coordinate = city_coordinates[best_points_, :]
    #ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    #plt.title('Initial route')
    #plt.legend(best_points_str)

    #ACO

    aca = ACA_TSP(func=calculate_total_distance, n_dim=nCities,
                  size_pop=50, max_iter=100,
                  distance_matrix=distance_matrix)

    best_x, best_y = aca.run()

    # %% Plot
    #fig, ax = plt.subplots(1, 1)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    plotRoute(city_x,city_y,best_x)
    best_points_str = [str(p) for p in best_points_]
    best_points_coordinate = city_coordinates[best_points_, :]
    #ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    #plt.title('ACA optimized route')
    #plt.legend(best_points_str)
    plt.show()


if __name__ == '__main__':
    main()
