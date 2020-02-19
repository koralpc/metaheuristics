#TSP-rd dataset generation
#Author : Koralp and Bingqian

#Imports
import numpy as np
import tsplib95
import os
import re
import sys


def read_all_datasets(directory):
    files = os.listdir(directory)
    tsp_problems = []
    tsp_solutions = []
    for filename in files:
        if '.tsp' in filename:
            tsp_problems.append(directory + '/' + filename)
            trim_name = re.sub('.tsp','',filename)
            tsp_solutions.append(directory + '/'+ trim_name + '.opt.tour')
    return list(zip(tsp_problems,tsp_solutions))


def read_tsp_file(filepath):
    problem = tsplib95.load_problem(filepath)
    city_coordinates = problem.node_coords
    city_coordinates = np.array([pair[1] for pair in city_coordinates.items()])
    max_coord = np.max(city_coordinates)
    city_coordinates = city_coordinates / max_coord
    return city_coordinates


def read_optimal_solution(filepath):
    solution = tsplib95.load_solution(filepath)
    solution_tour = np.array((solution.tours)[0]) - 1
    return solution_tour

def generate_release_dates(tour_size,optimal_distance,Beta = 1):
    release_dates = np.random.randint(0,optimal_distance * Beta,tour_size)
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
