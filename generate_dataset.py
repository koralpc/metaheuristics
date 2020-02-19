#TSP-rd dataset generation
#Author : Koralp and Bingqian

#Imports
import numpy as np



def generate_release_dates(tour_size):
    release_dates = np.random.randint(0,tour_size,tour_size)
    release_dates[0] = 0
    return release_dates


def generate_city_coordinates(tour_size):

    #Sample tour_size random coordinates
    city_coordinates = np.random.rand(tour_size, 2)
    return city_coordinates


def generate_grid(tour_size):
    grid_indexes =  np.arange(tour_size)
    #Coordinate system
    coordinates = np.array([(x,y) for x in grid_indexes for y in grid_indexes])
    return coordinates
