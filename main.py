#!/bin/python
"""
main.py - Part of ants project
execute via `python main.py` in terminal or only UNIX: `./main.py`

License: AGPL 3 (see end of file)
(C) Alexander Bocken, Viviane Fahrni, Grace Kagho
"""
from model import ActiveWalkerModel
import numpy as np
import matplotlib.pyplot as plt
from mesa.space import Coordinate
from mesa.datacollection import DataCollector

#from multihex import MultiHexGrid

def check_pheromone_exponential_decay():
    """
    Check whether wanted exponential decay of pheromones on grid is done correctly
    shows plot of pheromone placed on grid vs. equivalent exponential decay function
    """
    width = 21
    height = width
    num_initial_roamers = 0
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    max_steps = 1000

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              max_steps=max_steps)

    model.grid.fields["A"][5,5] = 10
    model.datacollector = DataCollector(
            model_reporters={"pheromone_a": lambda m: m.grid.fields["A"][5,5] },
                agent_reporters={}
                )
    model.run_model()
    a_test = model.datacollector.get_model_vars_dataframe()["pheromone_a"]

    plt.figure()
    xx = np.linspace(0,1000, 10000)
    yy = a_test[0]*np.exp(-model.decay_rates["A"]*xx)
    plt.plot(xx, yy, label="correct exponential function")
    plt.scatter(range(len(a_test)), a_test, label="modeled decay", marker='o')
    plt.title("Exponential grid pheromone decay test")
    plt.legend(loc='best')

    plt.show()


def check_ant_sensitivity_linear_decay():
    """
    Check whether wanted linear decay of ant sensitivity is done correctly
    shows plot of ant sensitivity placed on grid vs. equivalent linear decay function
    not food sources are on the grid for this run to not reset sensitivities
    """
    width = 50
    height = width
    num_initial_roamers = 1
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    max_steps = 1000
    num_food_sources = 0

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              num_food_sources=num_food_sources,
                              max_steps=max_steps)

    model.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"sensitivity": lambda a: a.sensitivity}
                )
    start = model.schedule.agents[0].sensitivity_decay_rate
    model.run_model()
    a_test = model.datacollector.get_agent_vars_dataframe().reset_index()["sensitivity"]

    plt.figure()
    xx = np.linspace(0,1000, 10000)
    yy = a_test[0] - start*xx
    plt.title("Linear Ant Sensitivity decay test")
    plt.plot(xx, yy, label="correct linear function")
    plt.scatter(range(len(a_test)), a_test, label="modeled decay", marker='o')
    plt.legend(loc='best')

    plt.show()

def check_ant_pheromone_exponential_decay():
    """
    Check whether wanted exponential decay of pheromone drop rate for ants is correctly modeled
    shows plot of pheromone placed on grid vs. equivalent exponential decay function
    """
    width = 50
    height = width
    num_initial_roamers = 1
    num_max_agents = 100
    nest_position : Coordinate = (width //2, height //2)
    num_food_sources = 0;
    max_steps = 1000

    model = ActiveWalkerModel(width=width, height=height,
                              num_initial_roamers=num_initial_roamers,
                              nest_position=nest_position,
                              num_max_agents=num_max_agents,
                              max_steps=max_steps)


    model.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"pheromone_drop_rate": lambda a: a.pheromone_drop_rate["A"]}
                )
    start = model.schedule.agents[0].pheromone_drop_rate["A"]
    model.run_model()
    a_test = model.datacollector.get_agent_vars_dataframe().reset_index()["pheromone_drop_rate"]

    plt.figure()
    xx = np.linspace(0,1000, 10000)
    yy = a_test[0]*np.exp(-model.schedule.agents[0].betas["A"]*xx)
    plt.plot(xx, yy, label="correct exponential function")
    plt.scatter(range(len(a_test)), a_test, label="modeled decay", marker='o')
    plt.title("Exponential pheromone drop rate decay test")
    plt.legend(loc='best')

    plt.show()

def check_ants_follow_gradient():
    """
    Create a path of neighbours with a static gradient.
    Observe whether ant correctly follows gradient once found. via matrix printouts
    8 = ant
    anything else: pheromone A density.
    The ant does not drop any new pheromones for this test
    """
    width, height = 20,20
    params = {
            "width": width, "height": height,
            "num_max_agents": 1,
            "num_food_sources": 0,
            "nest_position": (10,10),
            "num_initial_roamers": 1,
            }
    model = ActiveWalkerModel(**params)
    def place_line(grid : MultiHexGrid, start_pos=None):
        strength = 5
        if start_pos is None:
            start_pos = (9,9)
        next_pos = start_pos
        for _ in range(width):
            grid.fields["A"][next_pos] = strength
            strength += 0.01
            next_pos = grid.get_neighborhood(next_pos)[0]

    place_line(model.grid)

    ant = model.schedule._agents[0]
    ant.looking_for_pheromone = "A"
    ant.drop_pheromone = None
    ant.threshold["A"] = 0
    ant.sensitivity_max = 100
    #model.grid.fields["A"] = np.diag(np.ones(width))
    model.decay_rates["A"] = 0

    while model.schedule.steps < 100:
        display_field = np.copy(model.grid.fields["A"])
        display_field[ant.pos] = 8
        print(display_field)
        print(20*"#")
        model.step()

def viviane_bfs_example_run():
    # Breadth-first-search algorithm for connectivity
    def bfs(graph, start_node, threshold): #graph=grid, start_node=nest, threshold=TBD?
        from collections import deque
        visited = set()
        queue = deque([(start_node, [])])
        paths = {}
        connected_food_sources = set()

        while queue:
            current_node, path = queue.popleft()
            #current_node = tuple(current_node)
            visited.add(current_node)

            if current_node in graph:
                for neighbor, m.grid.fields["A"] in graph[current_node].items():
                    if neighbor not in visited and m.grid.fields["A"] >= threshold:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))

                        # Check if the neighbor is a food source
                        if neighbor in self.grid_food:
                            if neighbor not in paths:
                                paths[neighbor] = new_path
                                connected_food_sources.add(neighbor)

        connectivity = len(connected_food_sources)

        return connectivity


    # Calculate connectivity through BFS

    current_paths = bfs(self.grid, self.grid.fields["nests"], 0.000001)

    import numpy as np

    N = 121
    N_X = int(np.sqrt(N))
    N_Y = N // N_X
    # fancy way of saying absolutely nothing but 11

    xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')


    print(f"{N_X=}")

    print(f"{N_Y=}")

    print(f"{(xv, yv)=}")

    print(f"{xv=}")


from model import kwargs_paper_setup1 as kwargs
if __name__ == "__main__":
    print("Test")
    kwargs["resistance_map_type"] = "perlin"
    print(kwargs)
    model = ActiveWalkerModel(**kwargs)

    from hexplot import plot_hexagon
    a = np.zeros_like(model.grid.fields['food'])
    a[np.nonzero(model.grid.fields['food'])] = 1
    plot_hexagon(a, title="Nest locations")
    plot_hexagon(model.grid.fields['res'], title="Resistance Map")


    from tqdm import tqdm as progress_bar
    for _ in progress_bar(range(model.max_steps)):
        model.step()



    # Access the DataCollector
    #datacollector = model.datacollector
    ## Get the data from the DataCollector
    #model_data = datacollector.get_model_vars_dataframe()
    #print(model_data.columns)

    ## Plot the number of alive ants over time
    #plt.plot(model_data.index, model_data['alive_ants'])
    #plt.xlabel('Time')
    #plt.ylabel('Number of Alive Ants') #this should probably be "active" ants, since it is not considering those in the nest
    #plt.title('Number of Alive Ants Over Time')
    #plt.grid(True)
    #plt.show()

    ## Plot the number of sucessful walkers over time
    #plt.plot(model_data.index, model_data['sucessful_walkers'])
    #plt.xlabel('Time')
    #plt.ylabel('Number of Sucessful Walkers')
    #plt.title('Number of Sucessful Walkers Over Time')
    #plt.grid(True)
    #plt.show()


    ## Calculate the cumulative sum
    #model_data['cumulative_sucessful_walkers'] = model_data['sucessful_walkers'].cumsum()

    ## Plot the cumulative sum of sucessful walkers over time
    #plt.plot(model_data.index, model_data['cumulative_sucessful_walkers'])
    #plt.xlabel('Time')
    #plt.ylabel('Cumulative Sucessful Walkers')
    #plt.title('Cumulative Sucessful Walkers Over Time')
    #plt.grid(True)
    #plt.show()

    ## Values over 100 are to be interpreted as walkers being sucessfull several times since the total max number of ants is 100

    # # Connectivity measure
    #def check_food_source_connectivity(food_sources, paths): #food_sources = nodes.is_nest, paths=result from BFS
    #    connected_food_sources = set()

    #    for source in food_sources:
    #        if source in paths:
    #            connected_food_sources.add(source)

    #    connectivity = len(connected_food_sources)


    #    return connectivity


    #    # Calculate connectivity through BFS

    #    current_paths = bfs(self.grid, self.grid.fields["nests"], 0.000001)
"""
    This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>
"""
