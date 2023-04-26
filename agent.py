"""
agent.py - Part of ants project

This model implements the actual agents on the grid (a.k.a. the ants)

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
"""
import numpy as np
from mesa.agent import Agent
from mesa.space import Coordinate


class RandomWalkerAnt(Agent):
    def __init__(self, unique_id, model, do_follow_chemical_A=True,
                 energy_0=1, chemical_drop_rate_0=1, sensitvity_0=1, alpha=0.5)-> None:
        super().__init__(unique_id=unique_id, model=model)

        self._next_pos : None | Coordinate = None

        self.prev_pos = None
        self.do_follow_chemical_A : bool = True # False -> follow_chemical_B = True
        self.energy : float = energy_0
        self.sensitvity : float = sensitvity_0
        self.chemical_drop_rate : float = chemical_drop_rate_0 #TODO: check whether needs to be separated into A and B
        self.alpha = alpha

    def step(self):
        pass

    def advance(self) -> None:
        self.pos = self._next_pos

    @property
    def front_neighbors(self):
        if self.prev_pos is not None:
            x, y = self.pos
            x_prev, y_prev = self.prev_pos
            dx, dy = x - x_prev, y - y_prev
            front = [
                    (x, y + dy),
                    (x + dx, y),
                    (x + dx, y + dy),
                    ]
            return front #TODO: verify (do we need to sperate into even/odd?)
        else:
            # TODO: return all neighbors or raise Exception?
            pass




"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
