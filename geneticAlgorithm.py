from simulator import Simulator
import argparse
import random
from deap import base, creator, tools, algorithms
import os
import multiprocessing
import numpy as np


# Genetic Algorithm parameters
POPULATION_SIZE = 5
NUM_GENERATIONS = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
WEIGHTS_PER_PLAYER = 5  # Number of weights per player
LOG_DIR = "logs"

# Additional variables for dynamic range adjustment
RANGE_LIMIT = 2.5
RANGE_EXPANSION_FACTOR = 1.5
EXPANSION_THRESHOLD = 0.1  # Threshold to decide when to expand the range

# Starting weights
STARTING_WEIGHTS = [1.7583857200546573, 3.4629054768370144, 2.927490491901529, 1.3448708098474154, 2.9477359896052686, 1.4685137585053325, 2.570458388569774, 1.2760380235198903, 1.513138392794098, 0.8079408142090223]

# Initialize argparse.Namespace object with default values
args = argparse.Namespace(
    player_1="student_agent",
    player_2="student_agent",
    board_size=None,
    board_size_min=6,
    board_size_max=12,
    display=True,
    display_delay=0.4,
    display_save=False,
    display_save_path="plots/",
    autoplay=True,
    autoplay_runs=10
)

# Function to adjust weight range and starting weights
def adjust_weight_range():
    global RANGE_LIMIT
    max_weight = max(STARTING_WEIGHTS)
    min_weight = min(STARTING_WEIGHTS)

    # Adjust the range if starting weights are outside the current range
    while max_weight > RANGE_LIMIT or min_weight < -RANGE_LIMIT:
        RANGE_LIMIT *= RANGE_EXPANSION_FACTOR

    print(f"Initial weight range adjusted to +/-{RANGE_LIMIT}")

# Adjusted evaluate function
def evaluate(player_1_weights, player_2_weights):
    local_args = argparse.Namespace(**vars(args))
    local_args.player_1_weights = player_1_weights
    local_args.player_2_weights = player_2_weights
    p1_win_count, p2_win_count = Simulator(local_args).autoplay()
    return p1_win_count - p2_win_count,

# Adjusted evaluate_with_annealing function
def evaluate_with_annealing(player_1_weights, temperature):
    player_2_weights = np.random.uniform(-RANGE_LIMIT, RANGE_LIMIT, WEIGHTS_PER_PLAYER) if random.random() < temperature else STARTING_WEIGHTS[WEIGHTS_PER_PLAYER:]
    return evaluate(player_1_weights, player_2_weights)

# Call the function to adjust range based on starting weights
adjust_weight_range()

def log_best_weights(best_weights, generation):
    with open(os.path.join(LOG_DIR, f"best_weights_gen_{generation}.txt"), 'w') as f:
        f.write(str(best_weights) + "\n")

# Set up DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Adjusted attribute function to consider the new dynamic range
toolbox.register("attr_float", random.uniform, -RANGE_LIMIT, RANGE_LIMIT)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=WEIGHTS_PER_PLAYER * 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUTATION_RATE)
toolbox.register("select", tools.selTournament, tournsize=3)

historical_best = []



# Main function with all changes
def main():
    # Ensure log directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Setup the pool of workers
    num_processes = 4  # Example number, can be set as needed
    pool = multiprocessing.Pool(processes=num_processes)

    toolbox.register("map", pool.map)

    # Genetic Algorithm
    population = toolbox.population(n=POPULATION_SIZE)

    # Initialize population with starting weights
    for ind in population:
        ind[:len(STARTING_WEIGHTS)] = STARTING_WEIGHTS

    historical_best = []  # Store historical best players
    temperature = 1.0  # Initial temperature for simulated annealing

    for gen in range(NUM_GENERATIONS):
        # Use a historical best player as the opponent after a certain number of generations
        if gen >= 10 and historical_best:
            historical_opponent = random.choice(historical_best)
            for ind in population:
                ind[WEIGHTS_PER_PLAYER:] = historical_opponent

        offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)
        
        # Evaluate offspring using annealing or normal evaluation
        for ind in offspring:
            if random.random() < temperature:  # Apply simulated annealing
                ind.fitness.values = evaluate_with_annealing(ind[:WEIGHTS_PER_PLAYER], temperature)
            else:
                ind.fitness.values = evaluate(ind[:WEIGHTS_PER_PLAYER], ind[WEIGHTS_PER_PLAYER:])

        # Update population
        population = toolbox.select(offspring, k=len(population))
        best_ind = tools.selBest(population, 1)[0]

        # Log and save the historical best
        log_best_weights(best_ind, gen)
        print(f"Generation {gen}: Best individual is {best_ind}, {best_ind.fitness.values}")

        # Save best player of this generation in historical best
        if gen % 10 == 0:
            historical_best.append(best_ind[:WEIGHTS_PER_PLAYER])
            log_best_weights(best_ind[:WEIGHTS_PER_PLAYER], f"historical_best_gen_{gen}")


        # Decrease temperature for simulated annealing
        temperature *= 0.99

    # Find and print the best weights found
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    log_best_weights(best_ind, "final")

    # Close the pool of workers
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
