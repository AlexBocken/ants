"""
model.py - Part of ants project

This file implements the mesa model on which our ActiveRandomWalkerAnts
will act

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
"""

import numpy as np
from mesa.model import Model
from mesa.space import Coordinate, HexGrid, Iterable
from multihex import MultiHexGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import RandomWalkerAnt

class ActiveWalkerModel(Model):
    def __init__(self, width : int, height : int , num_max_agents : int,
                 num_initial_roamers : int,
                 nest_position : Coordinate,
                 max_steps:int=1000) -> None:
        super().__init__()
        self.schedule = SimultaneousActivation(self)
        self.grid = MultiHexGrid(width=width, height=height, torus=True) # TODO: replace with MultiHexGrid
        self._unique_id_counter : int = -1 # only touch via get_unique_id() or get_unique_ids(num_ids)

        self.max_steps = max_steps
        self.nest_position : Coordinate = nest_position
        self.num_max_agents = num_max_agents

        for agent_id in self.get_unique_ids(num_initial_roamers):
            agent = RandomWalkerAnt(unique_id=agent_id, model=self, do_follow_chemical_A=True)
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos=nest_position)

        self.datacollector = DataCollector(
                model_reporters={},
                agent_reporters={}
                )
        self.datacollector.collect(self) # keep at end of __init___

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

        if self.schedule.steps >= self.max_steps:
            self.running = False

    def get_unique_id(self) -> int:
        self._unique_id_counter += 1
        return self._unique_id_counter

    def get_unique_ids(self, num_ids : int):
        for _ in range(num_ids):
            yield self.get_unique_id()

"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
