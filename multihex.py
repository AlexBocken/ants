"""
multihex.py - Part of ants project

This file impements a Mesa HexGrid while allowing for multiple agents to be
at the same location. The base for this code comes from the MultiGrid class
in mesa/space.py

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
"""

from sys import dont_write_bytecode
from mesa.space import HexGrid
from mesa.agent import Agent
import numpy as np
import numpy.typing as npt
from mesa.space import Coordinate, accept_tuple_argument
import itertools
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

MultiGridContent = list[Agent]

class MultiHexGrid(HexGrid):
    """Hexagonal grid where each cell can contain more than one agent.
    Mostly based of mesa's HexGrid
    Functions according to odd-q rules.
    See http://www.redblobgames.com/grids/hexagons/#coordinates for more.

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.

    Methods:
        get_neighbors: Returns the objects surrounding a given cell.
        get_neighborhood: Returns the cells surrounding a given cell.
        iter_neighbors: Iterates over position neighbors.
        iter_neighborhood: Returns an iterator over cell coordinates that are
            in the neighborhood of a certain point.
    """
    grid: list[list[MultiGridContent]]

    @staticmethod
    def default_val() -> MultiGridContent:
        """Default value for new cell elements."""
        return []

    def place_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Place the agent at the specified location, and set its pos variable."""
        x, y = pos
        if agent.pos is None or agent not in self._grid[x][y]:
            self._grid[x][y].append(agent)
            agent.pos = pos
            if self._empties_built:
                self._empties.discard(pos)

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the given location and set its pos attribute to None."""
        pos = agent.pos
        x, y = pos
        self._grid[x][y].remove(agent)
        if self._empties_built and self.is_cell_empty(pos):
            self._empties.add(pos)
        agent.pos = None

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[Agent]:
        """Returns an iterator of the agents contained in the cells identified
        in `cell_list`; cells with empty content are excluded.

        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.

        Returns:
            An iterator of the agents contained in the cells identified in `cell_list`.
        """
        return itertools.chain.from_iterable(
            self._grid[x][y]
            for x, y in itertools.filterfalse(self.is_cell_empty, cell_list)
        )


class MultiHexGridScalarFields(MultiHexGrid):
    def __init__(self, fields: list[str], width : int, height : int, torus : bool, scalar_initial_value : float=0) -> None:
        super().__init__(width=width, height=height, torus=torus)

        self.fields : dict[str, npt.NDArray[np.float_]] = {}

        for key in fields:
            self.fields[key] = np.ones((width, height)).astype(float) * scalar_initial_value

    def reset_field(self, key : str) -> None:
        self.fields[key] = np.zeros((self.width, self.height))

    def is_food(self, pos):
        assert('food' in self.fields.keys())
        return bool(self.fields['food'][pos])

    def add_food(self, size : int , pos=None):
        """
        Adds food source to grid.
        Args:
        pos (optional): if None, selects random place on grid which
                        is not yet occupied by either a nest or another food source
        size:           how much food should be added to field
        """
        assert('food' in self.fields.keys())
        if pos is None:
            def select_random_place():
                i = np.random.randint(0, self.width)
                j = np.random.randint(0, self.height)
                return i,j
            pos = select_random_place()
            while(self.is_nest(pos) or self.is_food(pos)):
                pos = select_random_place()

        self.fields['food'][pos] = size

    def is_nest(self, pos : Coordinate) -> bool:
        assert('nests' in self.fields.keys())
        return bool(self.fields['nests'][pos])

    def add_nest(self, pos:None|Coordinate=None):
        """
        Adds nest to grid.
        Args:
        pos:    if None, selects random place on grid which
                is not yet occupied by either a nest or another food source
        """
        assert('nests' in self.fields.keys())
        if pos is None:
            def select_random_place():
                i = np.random.randint(0, self.width)
                j = np.random.randint(0, self.height)
                return i,j
            pos = select_random_place()
            while(self.is_nest(pos) or self.is_food(pos)):
                pos = select_random_place()

        self.fields['nests'][pos] = True


"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
