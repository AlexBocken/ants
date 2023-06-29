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



def fixed_distance_tests():
    """
    position a target food source a known distance away from nest
    check for no. successful ants for n runs
    """

    from tqdm import tqdm
    runs = 10
    from model import kwargs_paper_setup1 as kwargs
    kwargs["N_f"] = 0
    kwargs["gamma"] /= 2 # field decays three times slower
    kwargs["beta"] /= 2 # drop rates decays three times slower
    kwargs["d_s"] /= 2 # drop rates decays three times slower
    kwargs["d_e"] /= 2 # drop rates decays three times slower
    successful_walkers = {}
    for distance in tqdm(range(5,30), position=0, desc="dis"):
        successful_walkers[distance] = []
        for _ in tqdm(range(runs), position=1, desc="run", leave=False):
            model = ActiveWalkerModel(**kwargs)
            nest_location = kwargs["nest_position"]
            food_location =  (nest_location[0] - distance, nest_location[1])
            model.grid.add_food(size=100, pos=food_location)
            for _ in tqdm(range(model.max_steps), position=2, desc="step", leave=False):
                model.step()
            successful_walkers[distance].append(model.datacollector.get_model_vars_dataframe().reset_index()["successful_walkers"][kwargs["max_steps"]])
    return successful_walkers

def fixed_distance_object_between():
    """
    diameter of object: floor(50% of distance)
    """

    from tqdm import tqdm
    runs = 10
    from model import kwargs_paper_setup1 as kwargs
    kwargs["N_f"] = 0
    kwargs["gamma"] /= 2 # field decays slower
    kwargs["beta"] /= 2 # drop rates decays slower
    kwargs["d_e"] /= 2 # live longer, search longer
    kwargs["d_s"] /= 2 # live longer, search longer
    successful_walkers = {}
    for distance in tqdm(range(5,30), position=0, desc="dis"):
        successful_walkers[distance] = []
        for _ in tqdm(range(runs), position=1, desc="run", leave=False):
            model = ActiveWalkerModel(**kwargs)
            nest_location = kwargs["nest_position"]
            food_location =  (nest_location[0] - distance, nest_location[1])
            object_location =  (nest_location[0] - distance//2, nest_location[1])
            place_blocking_object(object_location, radius=distance//4, model=model)
            model.grid.add_food(size=100, pos=food_location)
            for _ in tqdm(range(model.max_steps), position=2, desc="step", leave=False):
                model.step()
            successful_walkers[distance].append(model.datacollector.get_model_vars_dataframe().reset_index()["successful_walkers"][kwargs["max_steps"]])
    return successful_walkers

def place_blocking_object(center, radius, model):
    positions = [center]
    next_outside = [center]
    # We grow from the center and add all neighbours of the outer edge of our blocking object
    # Add all neighbours of next_outside that aren't in positions to the object
    # by doing this radius times we should get an object of diameter 2 * radius + 1
    # positions: accumulator for all positions inside the object of radius radius
    # next_outside: keep track what we added in the last go-around. These will be used in the next step.
    for _ in range(radius):
        outside = next_outside
        next_oustide = []

        #otherwise interprets the tuple as something stupid
        for i in range(len(outside)):
            cell = outside[i]
            neighbours = model.grid.get_neighborhood(cell)
            for n in neighbours:
                if n not in positions:
                    positions.append(n)
                    next_outside.append(n)

    # some large number in comparison to the rest of the resistance field
    # such that the probability of stepping on these grid spots tend towards zero
    infinity = 1e20
    for pos in positions:
        model.grid.fields['res'][pos] = infinity


def run_model():
    from tqdm import tqdm
    # nests rather far away but also partially clumped.
    np.random.seed(6)

    from model import kwargs_paper_setup1 as kwargs
    kwargs["gamma"] /= 2
    kwargs["beta"] /= 2
    kwargs["d_e"] /= 5 # live longer, search longer
    kwargs["d_s"] /= 5 # live longer, search longer
    kwargs["N_0"] *= 2 # more initial roamers/scouts
    kwargs["max_steps"] *= 2 # more initial roamers/scouts

    model = ActiveWalkerModel(**kwargs)
    a = np.zeros_like(model.grid.fields['food'])
    a[np.nonzero(model.grid.fields['food'])] = 1
    a[np.nonzero(model.grid.fields['nests'])] = -1
    for _ in tqdm(range(model.max_steps)):
        model.step()
    return model



from model import kwargs_paper_setup1 as kwargs
kwargs["gamma"] /= 2
kwargs["beta"] /= 2
kwargs["d_e"] /= 5 # live longer, search longer
kwargs["d_s"] /= 5 # live longer, search longer
kwargs["N_0"] *= 2 # more initial roamers/scouts
kwargs["max_steps"] *= 2 # more initial roamers/scouts

def run_model_objects(step, seed=None, title=None):
    from tqdm import tqdm
    # nests rather far away but also partially clumped.
    np.random.seed(6)
    from hexplot import plot_hexagon
    model = ActiveWalkerModel(**kwargs)
    a = np.zeros_like(model.grid.fields['food'])
    a[np.nonzero(model.grid.fields['food'])] = 1
    a[np.nonzero(model.grid.fields['nests'])] = -1
    for current_step in tqdm(range(model.max_steps)):
        if current_step == step:
            if seed is not None:
                np.random.seed(seed)
            for _ in range(10):
                coord = np.random.randint(0, 100, size=2)
                coord = (coord[0], coord[1])
                place_blocking_object(center=coord,radius=5, model=model)
            a = model.grid.fields["res"]
            if title is not None:
                plot_hexagon(a, title=title)
        model.step()
    return model

#if __name__ == "__main__":
#plot_heatmap()
#res = run_model_no_objects()
for i in range(10):
    res = run_model_objects(step=6000, seed=i+100, title=f"objects/blockings_run_{i}")
    from plot import plot_alive_ants_vs_time, dead_ants_vs_time, plot_connectivity_vs_time
    plot_alive_ants_vs_time(res, title=f"objects/run_{i}")
    dead_ants_vs_time(res, title=f"objects/dead_ants_run_{i}")
    plot_connectivity_vs_time(res, title=f"objects/conn_run_{i}")


#print("DISTANCE TEST VS SUCCESSFUL ANTS OBJECT INBETWEEN")
#res = fixed_distance_tests()
#res = fixed_distance_object_between()
    # print("Test")
#from model import kwargs_paper_setup1 as kwargs
#kwargs["resistance_map_type"] = "perlin"
    # print(kwargs)
#model = ActiveWalkerModel(**kwargs)
#model.step()

    # a = np.zeros_like(model.grid.fields['food'])
    # a[np.nonzero(model.grid.fields['food'])] = 1
    # plot_hexagon(a, title="Nest locations")
    # plot_hexagon(model.grid.fields['res'], title="Resistance Map")


    # from tqdm import tqdm as progress_bar
    # for _ in progress_bar(range(model.max_steps)):
    #     model.step()



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
