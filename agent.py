"""
agent.py - Part of ants project

This model implements the actual agents on the grid (a.k.a. the ants)

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kagho
"""
import numpy as np
import numpy.typing as npt
from mesa.agent import Agent
from mesa.space import Coordinate

class RandomWalkerAnt(Agent):
    def __init__(self, unique_id, model, look_for_pheromone=None,
                 energy_0=1,
                 pheromone_drop_rate_0 : dict[str, float]={"A": 80, "B": 80},
                 sensitivity_0=0.99,
                 alpha=0.6, drop_pheromone=None,
                 betas : dict[str, float]={"A": 0.0512, "B": 0.0512},
                 sensitivity_decay_rate=0.01,
                 sensitivity_max = 1
                 ) -> None:

        super().__init__(unique_id=unique_id, model=model)

        self._next_pos : None | Coordinate = None
        self.prev_pos : None | Coordinate = None

        self.look_for_pheromone = look_for_pheromone
        self.drop_pheromone = drop_pheromone
        self.energy = energy_0 #TODO: use
        self.sensitivity_0 = sensitivity_0
        self.sensitivity = self.sensitivity_0
        self.pheromone_drop_rate = pheromone_drop_rate_0
        self.alpha = alpha
        self.sensitivity_max = sensitivity_max
        self.sensitivity_decay_rate = sensitivity_decay_rate
        self.betas = betas
        self.threshold : dict[str, float] = {"A": 1, "B": 1}


    def sens_adj(self, props, key) -> npt.NDArray[np.float_] | float:
        """
        returns the adjusted value of any property dependent on the current
        sensitivity.
        The idea is to have a nonlinear response, where any opinion below a
        threshold (here: self.threshold[key]) is ignored, otherwise it returns
        the property
        Long-term this function should be adjusted to return the property up
        to a upper threshold as well.


          returns   ^
                    |
            sens_max|           __________
                    |          /
                    |         /
                q^tr|        /
                    |
                   0|________
                    -----------------------> prop
        """
        # if props iterable create array, otherwise return single value
        try:
            iter(props)
        except TypeError:
            # TODO: proper nonlinear response, not just clamping
            if props > self.sensitivity_max:
                return self.sensitivity_max
            if props > self.threshold[key]:
                return props
            else:
                return 0

        arr : list[float] = []
        for prop in props:
            arr.append(self.sens_adj(prop, key))
        return np.array(arr)

    def _choose_next_pos(self):
        if self.prev_pos is None:
            i = np.random.choice(range(6))
            self._next_pos = self.neighbors()[i]
            return

        if self.searching_food:
            for neighbor in self.front_neighbors:
                if self.model.grid.is_food(neighbor):
                    self.drop_pheromone = "B"
                    self.sensitivity = self.sensitivity_0

                    self.prev_pos = neighbor
                    self._next_pos = self.pos

        elif self.searching_nest:
            for neighbor in self.front_neighbors:
                if self.model.grid.is_nest(neighbor):
                    self.look_for_pheromone = "A" # Is this a correct interpretation?
                    self.drop_pheromone = "A"
                    self.sensitivity = self.sensitivity_0

                    #TODO: Do we flip the ant here or reset prev pos?
                    # For now, flip ant just like at food
                    self.prev_pos = neighbor
                    self._next_pos = self.pos

                    # recruit new ants
                    for agent_id in self.model.get_unique_ids(self.model.num_new_recruits):
                        agent = RandomWalkerAnt(unique_id=agent_id, model=self.model, look_for_pheromone="B", drop_pheromone="A")
                        agent._next_pos = self.pos
                        self.model.schedule.add(agent)
                        self.model.grid.place_agent(agent, pos=neighbor)

        # follow positive gradient
        if self.look_for_pheromone is not None:
            front_concentration = [self.model.grid.fields[self.look_for_pheromone][cell] for cell in self.front_neighbors ]
            front_concentration = self.sens_adj(front_concentration, self.look_for_pheromone)
            current_pos_concentration = self.sens_adj(self.model.grid.fields[self.look_for_pheromone][self.pos], self.look_for_pheromone)
            gradient = front_concentration - np.repeat(current_pos_concentration, 3)
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


    def step(self):
        self.sensitivity -= self.sensitivity_decay_rate
        self._choose_next_pos()
        self._adjust_pheromone_drop_rate()

    def _adjust_pheromone_drop_rate(self):
        if(self.drop_pheromone is not None):
            self.pheromone_drop_rate[self.drop_pheromone] -= self.pheromone_drop_rate[self.drop_pheromone] * self.betas[self.drop_pheromone]

    def drop_pheromones(self) -> None:
        # should only be called in advance() as we do not use hidden fields
        if self.drop_pheromone is not None:
            self.model.grid.fields[self.drop_pheromone][self.pos] += self.pheromone_drop_rate[self.drop_pheromone]

    def advance(self) -> None:
        self.drop_pheromones()
        self.prev_pos = self.pos
        self.model.grid.move_agent(self, self._next_pos)

    # TODO: find out how to decorate with property properly
    def neighbors(self, pos=None, include_center=False):
        if pos is None:
            pos = self.pos
        return self.model.grid.get_neighborhood(pos, include_center=include_center)

    @property
    def searching_nest(self) -> bool:
        return self.drop_pheromone == "B"

    @property
    def searching_food(self) -> bool:
        return self.drop_pheromone == "A"

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
