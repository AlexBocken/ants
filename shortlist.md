# SHORTLIST

- nonlinear response to concentration of pheromones (with upper and lower threshold)
	-> needs a separation of sensitivity and internal energy

## More chaos
- what happens if you disrupt the trail with an obstacle?
- limited node capacity


# Model setup

## Agents
- previous_position
- position
- sensitivity
- internal energy
- pheromone drop rate (A/B)
- what chemical we´re looking for
- optional: how much food we have with us (and decrement to prevent dying if energy is low)
- alpha

### step function
- probablistic forward step based on concentrations and sensitvity
	- follow highest concentration probabilistically and be random otherwise
- drop pheromones
- have we found food -> change behaviour and decrease food amount + reset stuff
- are we at the nest (having found food)? -> recruit new ants + reset stuff
- decrement sensitivity
- decrement energy (optional: only without food)
	- do i need to die

## Model

### hexagonal grid
	- pheromone a/b concentration
	- nest location
	- food location/concentration
	- for later: node capacity
- N total ants
- a few other constants which need to be set

### step
	advance ants (call ants step function)
	decrease pheromone concentration on grid
