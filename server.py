#!/bin/python
"""
server.py - Part of ants project

This file sets up the mesa built-in visualization server
and runs it on file execution. (python server.py or on UNIX: ./server.py)
For now it displays ant locations as well as pheromone A
concentrations on two seperate grids

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kagho
"""

import numpy as np
from mesa.visualization.modules import CanvasHexGrid, ChartModule, CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from model import ActiveWalkerModel
from collections import defaultdict

def setup(params=None):
    # Set the model parameters
    if params is None:
        params = {
                  "width": 50, "height": 50,
                  "num_max_agents" : 1000,
                  "nest_position" : (25,25),
                  "num_initial_roamers" : 20,
                  }


    class CanvasHexGridMultiAgents(CanvasHexGrid):
        """
        A modification of CanvasHexGrid to not run visualization functions on all agents
        but on all grid positions instead
        """
        package_includes = ["HexDraw.js", "CanvasHexModule.js", "InteractionHandler.js"]
        portrayal_method = None  # Portrayal function
        canvas_width = 500
        canvas_height = 500

        def __init__(self, portrayal_method, grid_width, grid_height, canvas_width=500, canvas_height=500, norm_method=None):
            super().__init__(portrayal_method, grid_width, grid_height, canvas_width, canvas_height)
            self.norm_method = norm_method

        def render(self, model):
            grid_state = defaultdict(list)
            norm = self.norm_method(model)
            for x in range(model.grid.width):
                for y in range(model.grid.height):
                    portrayal = self.portrayal_method(model, (x, y), norm)
                    if portrayal:
                        portrayal["x"] = x
                        portrayal["y"] = y
                        grid_state[portrayal["Layer"]].append(portrayal)

            return grid_state


    def get_color(level, normalization):
        """
        level: level to calculate color between white and black (linearly)
        normalization: value for which we want full black color
        """
        return max(int(255 - level * 255 / normalization), 0)


    def portray_ant_density(model, pos, norm):
        if model.grid.is_nest(pos):
            col = "red"
        elif model.grid.is_food(pos):
            col = "green"
        else:
            col = get_color(level=len(model.grid[pos]), normalization=norm)
            col = f"rgb({col}, {col}, {col})"


        return {
            "Shape": "hex",
            "r": 1,
            "Filled": "true",
            "Layer": 0,
            "x": pos[0],
            "y": pos[1],
            "Color":  col,
        }

    def get_max_grid_val(model, key):
        return np.max(model.grid.fields[key])

    def portray_pheromone_density(model, pos, norm):
        col_a = get_color(level=model.grid.fields["A"][pos], normalization=norm)
        col_b = get_color(level=model.grid.fields["B"][pos], normalization=norm)
        return {
            "Shape": "hex",
            "r": 1,
            "Filled": "true",
            "Layer": 0,
            "x": pos[0],
            "y": pos[1],
            "Color": f"rgb({col_a}, {col_b}, 255)"
        }



    width = params['width']
    height = params['height']
    pixel_ratio = 10
    grid_ants = CanvasHexGridMultiAgents(portray_ant_density,
                                width, height, width*pixel_ratio, height*pixel_ratio,
                                norm_method=lambda m: 5)

    def norm_ants(model):
        return 5

    def norm_pheromones(model):
        max_a = np.max(model.grid.fields["A"])
        max_b = np.max(model.grid.fields["B"])
        return np.ceil(np.max([max_a, max_b, 20]) + 1e-4).astype(int)

    grid_pheromones = CanvasHexGridMultiAgents(portray_pheromone_density,
                                width, height, width*pixel_ratio, height*pixel_ratio,
                                norm_method=norm_pheromones
                                )
    test_text = TextElement()
    return ModularServer(ActiveWalkerModel,
                         [lambda m: "<h3>Ant density</h3><h5>Nest: Red, Food: Green</h5>",
                          grid_ants,
                          lambda m: f"<h5>Normalization Value: {norm_ants(m)}</h5>",
                          lambda m: "<h3>Pheromone Density</h3><h5>Pheromone A: Cyan, Pheromone B: Pink</h5>",
                          grid_pheromones,
                          lambda m: f"<h5>Normalization Value: {norm_pheromones(m)}</h5>"
                          ],
                           "Active Random Walker Ants", params)

if __name__ == "__main__":
    server = setup()
    server.launch()

"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
