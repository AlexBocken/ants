#!/bin/python
"""
server.py - Part of ants project

This file sets up the mesa built-in visualization server
and runs it on file execution.
For now it displays ant locations as well as pheromone A
concentrations on two seperate grids

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kragho
"""

import numpy as np
from mesa.visualization.modules import CanvasHexGrid, ChartModule, CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from model import ActiveWalkerModel
from collections import defaultdict

def setup():
    # Set the model parameters
    params = {
              "width": 50, "height": 50,
              "num_max_agents" : 100,
              "nest_position" : (25,25),
              "num_initial_roamers" : 5,
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

        def __init__(self, portrayal_method, grid_width, grid_height, canvas_width=500, canvas_height=500,):
            super().__init__(portrayal_method, grid_width, grid_height, canvas_width, canvas_height)

        def render(self, model):
            grid_state = defaultdict(list)
            for x in range(model.grid.width):
                for y in range(model.grid.height):
                    portrayal = self.portrayal_method(model, (x, y))
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
        rgb = int(255 - level * 255 / normalization)
        mono = f"{rgb:0{2}x}" # hex value of rgb value with fixed length 2
        rgb = f"#{3*mono}"
        return rgb

    def portray_ant_density(model, pos):
        return {
            "Shape": "hex",
            "r": 1,
            "Filled": "true",
            "Layer": 0,
            "x": pos[0],
            "y": pos[1],
            "Color": get_color(level=len(model.grid[pos]), normalization=5)
        }

    def portray_pheromone_density(model, pos):
        return {
            "Shape": "hex",
            "r": 1,
            "Filled": "true",
            "Layer": 0,
            "x": pos[0],
            "y": pos[1],
            "Color": get_color(level=model.grid.fields["A"][pos], normalization=3)
        }



    width = params['width']
    height = params['height']
    pixel_ratio = 10
    grid_ants = CanvasHexGridMultiAgents(portray_ant_density, width, height, width*pixel_ratio, height*pixel_ratio)
    grid_pheromones = CanvasHexGridMultiAgents(portray_pheromone_density, width, height, width*pixel_ratio, height*pixel_ratio)
    return ModularServer(ActiveWalkerModel, [grid_ants, grid_pheromones],
                           "Active Random Walker Ants", params)

if __name__ == "__main__":
    server = setup()
    server.launch()

"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
