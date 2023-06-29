#!/bin/python

import matplotlib.pyplot as plt
import numpy as np
from hexplot import plot_hexagon

def plot_alive_ants_vs_time(model, title=None):
    y = model.datacollector.get_model_vars_dataframe()["alive_ants"]
    plt.figure(figsize=(10,10), dpi=600)
    plt.plot(y)
    plt.xlabel("time step")
    plt.ylabel("alive agents")
    if title is None:
        plt.savefig("alive_agents_over_time.eps")
    else:
        plt.savefig(f"{title}.png")


def plot_connectivity_vs_time(model, title=None):
    y = model.datacollector.get_model_vars_dataframe()["connectivity"]
    plt.figure(figsize=(10,10), dpi=600)
    plt.plot(y)
    plt.xlabel("time step")
    plt.ylabel("No. of food sources connected to the nest")
    if title is None:
        plt.savefig("connectivity_over_time.eps")
    else:
        plt.savefig(f"{title}.png")


def dead_ants_vs_time(model, title=None):
    y = np.cumsum(model.datacollector.get_model_vars_dataframe()["dying_ants"])
    plt.figure(figsize=(10,10), dpi=600)
    plt.plot(y)
    plt.xlabel("time step")
    plt.ylabel("dead agents")
    if title is None:
        plt.savefig("dead_agents_over_time.eps")
    else:
        plt.savefig(f"{title}.png")


def cum_successful_ants_vs_time(model, title=None):
    y = model.datacollector.get_model_vars_dataframe()["successful_walkers"]
    plt.figure(figsize=(10,10), dpi=600)
    plt.plot(y)
    plt.xlabel("time step")
    plt.ylabel("cummulative successful agents")
    if title is None:
        plt.savefig("cumsum_successful_agents_over_time.eps")
    else:
        plt.savefig(f"{title}.png")


def plot_heatmap(model, low=10, high=200):
    for time in np.arange(0, model.max_steps + 1, 1000):
        pheromone_concentration = model.datacollector.get_model_vars_dataframe()["pheromone_a"][time]
        a = pheromone_concentration
        #plot_hexagon(a)
        pheromone_concentration = model.datacollector.get_model_vars_dataframe()["pheromone_b"][time]
        b = pheromone_concentration
        #plot_hexagon(b)
        c = np.max([a,b], axis=0)
        c = a + b
        c = np.clip(c, 1, 1000000000)
        c = np.log(c)
        c = c/np.max(c)
        food_locations = np.nonzero(model.grid.fields['food'])
        x_food = [ food[0] for food in food_locations ]
        y_food = [ food[1] for food in food_locations ]
        plot_hexagon(c, title=f"cummulative pheromone density at timestep {time}")
