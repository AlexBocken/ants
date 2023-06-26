"""
agent.py - Part of ants project

This model implements the actual agents on the grid (a.k.a. the ants)

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kagho
"""

"""
TO DISCUSS:
Is the separation of energy and sensitivity useful?

"""
import numpy as np
import numpy.typing as npt
from mesa.agent import Agent
from mesa.space import Coordinate


class RandomWalkerAnt(Agent):
    def __init__(self, unique_id, model,
                 look_for_pheromone=None,
                 drop_pheromone=None,
                 sensitivity_max = 30000,
                 ) -> None:

        super().__init__(unique_id=unique_id, model=model)

        self._next_pos : None | Coordinate = None
        self._prev_pos : None | Coordinate = None

        self.look_for_pheromone : str|None = look_for_pheromone
        self.drop_pheromone : str|None = drop_pheromone
        self.energy : float = self.model.e_0
        self.sensitivity : float = self.model.s_0
        self.pheromone_drop_rate : float = self.model.q_0
        self.sensitivity_max = sensitivity_max

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
            if props > self.model.q_tr:
                return props
            else:
                return 0

        arr : list[float] = []
        for prop in props:
            arr.append(self.sens_adj(prop, key))
        return np.array(arr)

    def _get_resistance_weights(self, positions=None):
            if positions is None:
                positions = self.neighbors()
            # bit round-about but self.model.grid.fields['res'][positions]
            # gets interpreted as slices, not multiple singular positions
            resistance = np.array([ self.model.grid.fields['res'][x,y] for x,y in positions ])
            easiness = np.max(self.model.grid.fields['res']) - resistance + 1e-15 # + epsilon to not divide by zero
            weights = easiness/ np.sum(easiness)

            return weights

    def _choose_next_pos(self):
        def _combine_weights(res_weights, walk_weights):
            """
            If we have a resistance -> Infinity we want to have a likelihood -> 0 for this direction
            Therefore we should multiply our two probabilities.
            For the case of no resistance field this will return the normal walk_weights
            res_weights : resistance weights: based on resistance field of neighbours
                            see _get_resistance_weights for more info
            walk weights: In case of biased random walk (no positive pheromone gradient):
                             forward: alpha,
                             everywhere else: (1- alpaha)/5)
                          In case of positive pheromone gradient present in front:
                             max. positive gradient: self.sensitivity
                             everyhwere else: (1-self.sensitivity)/5
            """
            combined = res_weights * walk_weights
            normalized = combined / np.sum(combined)
            return normalized

        def _pick_from_remaining_five(remaining_five):
            """
            """
            weights = self._get_resistance_weights(remaining_five)
            random_index = np.random.choice(range(len(remaining_five)), p=weights)
            self._next_pos = remaining_five[random_index]
            self._prev_pos = self.pos

        if self._prev_pos is None:
            res_weights = self._get_resistance_weights()
            walk_weights = np.ones(6)
            weights = _combine_weights(res_weights, walk_weights)

            i = np.random.choice(range(6),p=weights)
            assert(self.pos is not self.neighbors()[i])
            self._next_pos = self.neighbors()[i]
            self._prev_pos = self.pos
            return

        if self.searching_food:
            for neighbor in self.front_neighbors:
                if self.model.grid.is_food(neighbor):
                    self.model.grid.fields['food'][neighbor] -= 1 # eat
                    #resets
                    self.pheromone_drop_rate = self.model.q_0
                    self.sensitivity = self.model.s_0
                    self.energy = self.model.e_0

                    #now look for other pheromone
                    self.look_for_pheromone = "A"
                    self.drop_pheromone = "B"

                    self._prev_pos = neighbor
                    self._next_pos = self.pos
                    return

        elif self.searching_nest:
            for neighbor in self.front_neighbors:
                if self.model.grid.is_nest(neighbor):
                    #resets
                    self.pheromone_drop_rate = self.model.q_0
                    self.sensitivity = self.model.s_0
                    self.energy = self.model.e_0

                    self.look_for_pheromone = "B"
                    self.drop_pheromone = "A"

                    self._prev_pos = neighbor
                    self._next_pos = self.pos
                    self.model.successful_ants += 1


                    # recruit new ants
                    for agent_id in self.model.get_unique_ids(self.model.N_r):
                        if self.model.schedule.get_agent_count() <  self.model.N_m:
                            agent = RandomWalkerAnt(unique_id=agent_id, model=self.model, look_for_pheromone="B", drop_pheromone="A")
                            agent._next_pos = self.pos
                            self.model.schedule.add(agent)
                            self.model.grid.place_agent(agent, pos=neighbor)
                    return

        # follow positive gradient with likelihood self.sensitivity
        if self.look_for_pheromone is not None:
            # Calculate gradient
            front_concentration = [self.model.grid.fields[self.look_for_pheromone][cell] for cell in self.front_neighbors ]
            front_concentration = self.sens_adj(front_concentration, self.look_for_pheromone)
            current_pos_concentration = self.sens_adj(self.model.grid.fields[self.look_for_pheromone][self.pos], self.look_for_pheromone)
            gradient = front_concentration - np.repeat(current_pos_concentration, 3).astype(np.float_)

            index = np.argmax(gradient)
            if gradient[index] > 0:
                # follow positive gradient with likelihood self.sensitivity * resistance_weight (re-normalized)

                all_neighbors_cells = self.neighbors()
                highest_gradient_cell = self.front_neighbors[index]
                highest_gradient_index_arr = np.where(all_neighbors_cells == highest_gradient_cell)
                assert(len(highest_gradient_index_arr) == 1)

                all_neighbors_index = highest_gradient_index_arr[0]
                sens_weights = np.ones(6) * (1-self.sensitivity)/5
                sens_weights[all_neighbors_index] = self.sensitivity

                res_weights = self._get_resistance_weights()
                weights = _combine_weights(res_weights, sens_weights)

                self._next_pos = np.random.choice(all_neighbors_cells, p=weights)
                self._prev_pos = self.pos
                return

        # do biased random walk
        all_neighbors_cells = self.neighbors()
        front_index_arr = np.where(all_neighbors_cells == self.front_neighbor)
        assert(len(front_index_arr) == 1 )
        front_index = front_index_arr[0]

        res_weights = self._get_resistance_weights()
        walk_weights = np.ones(6) * (1-self.model.alpha) / 5
        walk_weights[front_index] = self.model.alpha

        weights = _combine_weights(res_weights, walk_weights)
        self._nex_pos = np.random.choice(all_neighbors_cells, p=weights)
        self._prev_pos = self.pos

    def step(self):
        self.sensitivity -= self.model.d_s
        self.energy -= self.model.grid.fields['res'][self.pos] * self.model.d_e
        # Die and get removed if no energy
        if self.energy < self.model.e_min:
            self.model.schedule.remove(self)
        else:
            self._choose_next_pos()
            self._adjust_pheromone_drop_rate()

    def _adjust_pheromone_drop_rate(self):
        if(self.drop_pheromone is not None):
            self.pheromone_drop_rate -= self.pheromone_drop_rate * self.model.beta

    def drop_pheromones(self) -> None:
        # should only be called in advance() as we do not use hidden fields
        if self.drop_pheromone is not None:
            self.model.grid.fields[self.drop_pheromone][self.pos] += self.pheromone_drop_rate

    def advance(self) -> None:
        self.drop_pheromones()
        self.model.grid.move_agent(self, self._next_pos)
        self._next_pos = None # so that we rather crash than use wrong data

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
        all_neighbors = self.neighbors()
        neighbors_at_the_back = self.neighbors(pos=self._prev_pos, include_center=True)
        front_neighbors = list(filter(lambda i: i not in neighbors_at_the_back, all_neighbors))

        ########## DEBUG
        try:
            assert(self._prev_pos is not None)
            assert(self._prev_pos is not self.pos)
            assert(self._prev_pos in all_neighbors)
            assert(len(front_neighbors) == 3)
        except AssertionError:
            print(f"{self._prev_pos=}")
            print(f"{self.pos=}")
            print(f"{all_neighbors=}")
            print(f"{neighbors_at_the_back=}")
            print(f"{front_neighbors=}")
            raise AssertionError
        else:
            return front_neighbors

    @property
    def front_neighbor(self):
        """
        returns neighbor of current pos
        which is towards the front of the ant
        """
        neighbors__prev_pos = self.neighbors(self._prev_pos)
        for candidate in self.front_neighbors:
            # neighbor in front direction only shares current pos as neighborhood with _prev_pos
            candidate_neighbors = self.model.grid.get_neighborhood(candidate)
            overlap = [x for x in candidate_neighbors if x in neighbors__prev_pos]
            if len(overlap) == 1:
                return candidate


"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
