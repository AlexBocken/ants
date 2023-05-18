#!/bin/python
"""
main.py - Part of ants project
execute via `python main.py` in terminal or only UNIX: `./main.py`

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kagho
"""
from model import ActiveWalkerModel
from agent import RandomWalkerAnt
import numpy as np
import matplotlib.pyplot as plt
from mesa.space import Coordinate
from mesa.datacollection import DataCollector

def main():
    check_pheromone_exponential_decay()
    check_ant_sensitivity_linear_decay()
    check_ant_pheromone_exponential_decay()

def check_pheromone_exponential_decay():
    """
    Check whether wanted exponential decay of pheromones on grid is done correctly
    shows plot of pheromone placed on grid vs. equivalent exponential decay function
    """
    width = 21
    height = width
    num_initial_roamers = 0
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    max_steps = 1000

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              max_steps=max_steps)

    model.grid.fields["A"][5,5] = 10
    model.datacollector = DataCollector(
            model_reporters={"pheromone_a": lambda m: m.grid.fields["A"][5,5] },
                agent_reporters={}
                )
    model.run_model()
    a_test = model.datacollector.get_model_vars_dataframe()["pheromone_a"]

    plt.figure()
    xx = np.linspace(0,1000, 10000)
    yy = a_test[0]*np.exp(-model.decay_rates["A"]*xx)
    plt.plot(xx, yy, label="correct exponential function")
    plt.scatter(range(len(a_test)), a_test, label="modeled decay", marker='o')
    plt.title("Exponential grid pheromone decay test")
    plt.legend(loc='best')

    plt.show()


def check_ant_sensitivity_linear_decay():
    """
    Check whether wanted linear decay of ant sensitivity is done correctly
    shows plot of ant sensitivity placed on grid vs. equivalent linear decay function
    not food sources are on the grid for this run to not reset sensitivities
    """
    width = 50
    height = width
    num_initial_roamers = 1
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    max_steps = 1000
    num_food_sources = 0

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              num_food_sources=num_food_sources,
                              max_steps=max_steps)

    model.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"sensitivity": lambda a: a.sensitivity}
                )
    start = model.schedule.agents[0].sensitivity_decay_rate
    model.run_model()
    a_test = model.datacollector.get_agent_vars_dataframe().reset_index()["sensitivity"]

    plt.figure()
    xx = np.linspace(0,1000, 10000)
    yy = a_test[0] - start*xx
    plt.title("Linear Ant Sensitivity decay test")
    plt.plot(xx, yy, label="correct linear function")
    plt.scatter(range(len(a_test)), a_test, label="modeled decay", marker='o')
    plt.legend(loc='best')

    plt.show()

def check_ant_pheromone_exponential_decay():
    """
    Check whether wanted exponential decay of pheromone drop rate for ants is correctly modeled
    shows plot of pheromone placed on grid vs. equivalent exponential decay function
    """
    width = 50
    height = width
    num_initial_roamers = 1
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    max_steps = 1000

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              max_steps=max_steps)


    model.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"pheromone_drop_rate": lambda a: a.pheromone_drop_rate["A"]}
                )
    start = model.schedule.agents[0].pheromone_drop_rate["A"]
    model.run_model()
    a_test = model.datacollector.get_agent_vars_dataframe().reset_index()["pheromone_drop_rate"]

    plt.figure()
    xx = np.linspace(0,1000, 10000)
    yy = a_test[0]*np.exp(-model.schedule.agents[0].betas["A"]*xx)
    plt.plot(xx, yy, label="correct exponential function")
    plt.scatter(range(len(a_test)), a_test, label="modeled decay", marker='o')
    plt.title("Exponential pheromone drop rate decay test")
    plt.legend(loc='best')

    plt.show()

if __name__ == "__main__":
    main()



"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
