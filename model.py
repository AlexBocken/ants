"""
model.py - Part of ants project

This file implements the mesa model on which our ActiveRandomWalkerAnts
will act

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kagho
"""

import numpy as np
from mesa.model import Model
from mesa.space import Coordinate, HexGrid, Iterable
from multihex import MultiHexGridScalarFields
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import RandomWalkerAnt

kwargs_paper_setup1 = {
        "width": 100,
        "height": 100,
        "N_0": 20,
        "N_m": 100,
        "N_r": 5,
        "alpha": 0.6,
        "gamma": 0.001,
        "beta": 0.0512,
        "d_s": 0.001,
        "d_e": 0.001,
        "s_0": 0.99,
        "e_0": 0.99,
        "q_0": 80,
        "q_tr": 1,
        "e_min": 0,
        "nest_position": (49,49),
        "N_f": 5,
        "food_size" : 55,
        "max_steps": 8000,
        "resistance_map_type" : None,
}

kwargs_paper_setup2 = {
        "width": 100,
        "height": 100,
        "N_0": 20,
        "N_m": 100,
        "N_r": 5,
        "alpha": 0.6,
        "gamma": 0.01,
        "beta": 0.0512,
        "d_s": 0.001,
        "d_e": 0.001,
        "s_0": 0.99,
        "e_0": 0.99,
        "q_0": 80,
        "q_tr": 1,
        "e_min": 0,
        "nest_position": (49,49),
        "N_f": 5,
        "food_size" : 550,
        "max_steps": 8000,
        "resistance_map_type" : None,
}


class ActiveWalkerModel(Model):
    def __init__(self, width : int, height : int,
                 N_0 : int, # number of initial roamers
                 N_m : int, # max number of ants
                 N_r : int, # number of new recruits
                 alpha : float, #biased random walk
                 beta : float, # decay rate drop rate
                 gamma : float, # decay rate pheromone concentration fields
                 d_s : float, # decay rate sensitvity
                 d_e : float, # decay rate energy
                 s_0 : float, # sensitvity reset
                 e_0 : float, # energy reset
                 q_0 : float,  # initial pheromone level
                 q_tr : float, # threshold under which ant cannot distinguish concentrations
                 e_min : float, # energy at which walker dies
                 nest_position : Coordinate,
                 N_f=5, #num food sources
                 food_size= 55,
                 max_steps:int=1000,
                 resistance_map_type=None,
                 ) -> None:
        super().__init__()

        self.N_m : int       = N_m   # max number of ants
        self.N_r : int       = N_r   # number of new recruits
        self.alpha : float   = alpha # biased random walk if no gradient
        self.gamma : float   = gamma # decay rate pheromone concentration fields
        self.beta : float    = beta  # decay rate drop rate
        self.d_s : float     = d_s   # decay rate sensitvity
        self.d_e : float     = d_e   # decay rate energy (get's multiplied with resistance)
        self.s_0 : float     = s_0   # sensitvity reset
        self.e_0 : float     = e_0   # energy reset
        self.q_0 : float     = q_0   # pheromone drop rate reset
        self.q_tr : float    = q_tr  # threshold under which ant cannot distinguish concentrations
        self.e_min : float   = e_min # energy at which walker dies
        self.N_f : int       = N_f #num food sources
        self.successful_ants = 0    # for viviane's graph
        self.connectivity    = 0    # for viviane's persistence
        self.dying_agents = 0

        fields=["A", "B", "nests", "food", "res"]
        self.schedule = SimultaneousActivation(self)
        self.grid = MultiHexGridScalarFields(width=width, height=height, torus=True, fields=fields)

        if resistance_map_type is None:
            self.grid.fields["res"] = np.ones((width, height)).astype(float)
        elif resistance_map_type == "perlin":
            # perlin generates anisotropic noise which may or may not be a good choice
            # pip3 install git+https://github.com/pvigier/perlin-numpy
            from perlin_numpy import (
                generate_fractal_noise_2d,
                generate_perlin_noise_2d,
            )
            noise = generate_perlin_noise_2d(shape=(width,height), res=((10,10)))
            # normalized to mean=1, min=0, and max=2
            normalized_noise = (noise - np.min(noise))/(np.max(noise) - np.min(noise)) * 2
            self.grid.fields["res"] = normalized_noise
        else:
            # possible other noise types: simplex or value
            raise NotImplemented(f"{resistance_map_type=} is not implemented.")


        self._unique_id_counter = -1

        self.max_steps = max_steps
        self.grid.add_nest(nest_position)


        for agent_id in self.get_unique_ids(N_0):
            if self.schedule.get_agent_count() < self.N_m:
                agent = RandomWalkerAnt(unique_id=agent_id, model=self, look_for_pheromone="A", drop_pheromone="A")
                self.schedule.add(agent)
                self.grid.place_agent(agent, pos=nest_position)

        for _ in range(N_f):
            self.grid.add_food(food_size)

        self.datacollector = DataCollector(
                # model_reporters={"agent_dens": lambda m: m.agent_density()},
                model_reporters = {"pheromone_a": lambda m: m.grid.fields["A"],
                                    "pheromone_b": lambda m: m.grid.fields["B"],
                                    "alive_ants": lambda m: m.schedule.get_agent_count(),
                                    "dying_ants": lambda m: m.dying_agents,
                                    "successful_walkers": lambda m: m.successful_ants,
                                    "connectivity": lambda m: m.connectivity,
                                   },
                agent_reporters={}
                )
        self.datacollector.collect(self) # keep at end of __init___

    # Breadth-first-search algorithm for connectivity
    def bfs(self):
        threshold = 0.5 # half of min sens
        connectivity = 0 #initial value of connectivity
        connected_food_sources = [] #empty list of connected food sources
        visited = [] #empty list of visited (by the algorithm) nodes

        nest = np.argwhere(self.grid.fields["nests"] == 1) #get nest location
        nest = (nest[0][0], nest[0][1])
        start_node = nest #rename

        neighbours_to_check = [start_node]  #start node gets checked first
        neighbours_to_check = neighbours_to_check + self.grid.get_neighborhood(start_node) #start node neighbours get added to the to check list

        while neighbours_to_check: #as long as there is something on the to check list
            current_node = neighbours_to_check[0] #the first list entry is taken
            del neighbours_to_check[0] #and deleted on the to check list

            if current_node not in visited: #if it has not previously been checked
                if np.max([self.grid.fields["B"][current_node], self.grid.fields["A"][current_node]]) >= threshold: #and its A value is above our threshold
                    new_neighbors = self.grid.get_neighborhood(current_node) #then we get its neighbours
                    for new_neighbor in new_neighbors:
                        if new_neighbor not in visited and new_neighbor not in neighbours_to_check:
                            neighbours_to_check.append(new_neighbor) #then they are also added to our to check list
                visited.append(current_node) #and the current node has now been checked

                if self.grid.fields["food"][current_node] > 0: #in case the node we check is food
                    connectivity += 1 #then we have found a connected path to a food source
                    connected_food_sources = connected_food_sources + list([current_node]) #and it is added to the list of connected food sources

        # why not normalize to 0-1 ?
        return connectivity #we want the connectivity (0-5)

    def agent_density(self):
        a = np.zeros((self.grid.width, self.grid.height))
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                a[i,j] = len(self.grid[(i,j)])
        return a


    def step(self):
        self.dying_agents = 0
        self.schedule.step()        # step() and advance() all agents

        if self.schedule.steps % 100 == 0:
            pass

        # apply decay rate on pheromone levels
        for key in ("A", "B"):
            field = self.grid.fields[key]
            self.grid.fields[key] =  field - self.gamma*field


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
