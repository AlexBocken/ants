#!/bin/python
"""
main.py - Part of ants project
execute via `python main.py` in terminal or only UNIX: `./main.py`

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
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
    for agent in model.grid.get_neighbors(pos=nest_position, include_center=True):
        if agent.unique_id == 2:
            agent.do_follow_chemical_A = False
            agent.prev_pos = (9,10)
            print(agent.front_neighbors)
        print(agent.pos, agent.unique_id, agent.do_follow_chemical_A)


if __name__ == "__main__":
    main()



"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
