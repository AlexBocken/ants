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

def main():
    width = 21
    height = width
    num_initial_roamers = 5
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    max_steps = 100

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              max_steps=max_steps)

    # just initial testing of MultiHexGrid
    a = model.agent_density()
    for loc in model.grid.iter_neighborhood(nest_position):
        a[loc] = 3
    for agent in model.grid.get_neighbors(pos=nest_position, include_center=True):
        if agent.unique_id == 2:
            agent.look_for_chemical = "A"
            agent.prev_pos = (9,10)
            a[agent.prev_pos] = 1
            for pos in agent.front_neighbors:
                a[pos] = 6
            agent.step()
            print(f"{agent._next_pos=}")
            agent.advance()
            print(agent.front_neighbor)
            a[agent.front_neighbor] = 5

        print(agent.pos, agent.unique_id, agent.look_for_chemical)
    neighbors = model.grid.get_neighborhood(nest_position)
    print(neighbors)

    print(a)


if __name__ == "__main__":
    main()



"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
