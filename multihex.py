"""
multihex.py - Part of ants project

This file impements a Mesa HexGrid while allowing for multiple agents to be
at the same location. The base for this code comes from the MultiGrid class
in mesa/space.py

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
"""

from mesa.space import HexGrid
from mesa.agent import Agent
import numpy as np
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



"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
