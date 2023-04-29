"""
agent.py - Part of ants project

This model implements the actual agents on the grid (a.k.a. the ants)

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
"""
import numpy as np
from mesa.agent import Agent
from mesa.space import Coordinate
from typing import overload


class RandomWalkerAnt(Agent):
    def __init__(self, unique_id, model, look_for_chemical=None,
                 energy_0=1, chemical_drop_rate_0=1, sensitvity_0=1, alpha=0.5, drop_chemical=None) -> None:
        super().__init__(unique_id=unique_id, model=model)

        self._next_pos : None | Coordinate = None

        self.prev_pos : None | Coordinate = None
        self.look_for_chemical = look_for_chemical
        self.drop_chemical = drop_chemical
        self.energy : float = energy_0
        self.sensitvity : float = sensitvity_0
        self.chemical_drop_rate : float = chemical_drop_rate_0 #TODO: check whether needs to be separated into A and B
        self.alpha = alpha


    def sensitvity_to_concentration(self, prop : float) -> float:
        # TODO
        return prop

    def step(self):
        # follow positive gradient
        if self.prev_pos is None:
            i = np.random.choice(range(6))
            self._next_pos = self.neighbors()[i]
            return
        if self.look_for_chemical is not None:
            front_concentration = [self.model.grid.fields[self.look_for_chemical][cell] for cell in self.front_neighbors ]
            gradient = front_concentration - np.repeat(self.model.grid.fields[self.look_for_chemical][self.pos], 3)
            index = np.argmax(gradient)
            if gradient[index] > 0:
                self._next_pos = self.front_neighbors[index]
                return

        # do biased random walk
        p = np.random.uniform()
        if p < self.alpha:
            self._next_pos = self.front_neighbor
        else:
            # need copy() as we would otherwise remove the tuple from all possible lists (aka python "magic")
            other_neighbors = self.neighbors().copy()
            other_neighbors.remove(self.front_neighbor)
            random_index = np.random.choice(range(len(other_neighbors)))
            self._next_pos = other_neighbors[random_index]

    def drop_chemicals(self) -> None:
        # should only be called in advance() as we do not use hidden fields
        if self.drop_chemical is not None:
            self.model.grid.fields[self.drop_chemical][self.pos] += self.chemical_drop_rate

    def advance(self) -> None:
        self.drop_chemicals()
        self.prev_pos = self.pos
        self.model.grid.move_agent(self, self._next_pos)

    # TODO: find out how to decorate with property properly
    def neighbors(self, pos=None, include_center=False):
        if pos is None:
            pos = self.pos
        return self.model.grid.get_neighborhood(pos, include_center=include_center)

    @property
    def front_neighbors(self):
        """
        returns all three neighbors which the ant can see
        """
        assert(self.prev_pos is not None)
        all_neighbors = self.neighbors()
        neighbors_at_the_back = self.neighbors(pos=self.prev_pos, include_center=True)
        return list(filter(lambda i: i not in neighbors_at_the_back, all_neighbors))

    @property
    def front_neighbor(self):
        """
        returns neighbor of current pos
        which is towards the front of the ant
        """
        neighbors_prev_pos = self.neighbors(self.prev_pos)
        for candidate in self.front_neighbors:
            # neighbor in front direction only shares current pos as neighborhood with prev_pos
            candidate_neighbors = self.model.grid.get_neighborhood(candidate)
            overlap = [x for x in candidate_neighbors if x in neighbors_prev_pos]
            if len(overlap) == 1:
                return candidate


"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
