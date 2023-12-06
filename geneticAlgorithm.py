from simulator import Simulator
import argparse
import random
from deap import base, creator, tools, algorithms
import os
import multiprocessing
import numpy as np



# Genetic Algorithm parameters
POPULATION_SIZE = 24
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.35
CROSSOVER_RATE = 0.8
WEIGHTS_PER_PLAYER = 8  # Number of weights per player
LOG_DIR = "logs"

# Additional variables for dynamic range adjustment
RANGE_LIMIT = 2.5
RANGE_EXPANSION_FACTOR = 1.5

# Starting weights
myAgentWeights = [1,1,1,1,1]
#expansionWeight=1.0, centerDistanceWeight=1.0, controlWeight=1.0, barrierWeight=1.0, immediateBarrierWeight=1.0)
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
    autoplay_runs=5
)

def adjust_weight_range():
    global RANGE_LIMIT
    max_weight = max(myAgentWeights)
    min_weight = min(myAgentWeights)

    while max_weight > RANGE_LIMIT:
        RANGE_LIMIT *= RANGE_EXPANSION_FACTOR

    print(f"Initial weight range adjusted to +{RANGE_LIMIT}")

adjust_weight_range()

def log_best_weights(best_weights, generation):
    with open(os.path.join(LOG_DIR, f"best_weights_gen_{generation}.txt"), 'w') as f:
        f.write(str(best_weights) + "\n")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual):
    player_1_weights = individual[:WEIGHTS_PER_PLAYER]
    player_2_weights = individual[WEIGHTS_PER_PLAYER:]
    
    local_args = argparse.Namespace(**vars(args))
    local_args.player_1_weights = player_1_weights
    local_args.player_2_weights = player_2_weights
    p1_win_count, p2_win_count = Simulator(local_args).autoplay()
    return p1_win_count - p2_win_count,

def diversity(individuals):
    """ Calculate population diversity as the average pairwise distance between individuals. """
    distances = []
    for i in range(len(individuals)):
        for j in range(i+1, len(individuals)):
            distances.append(np.linalg.norm(np.array(individuals[i]) - np.array(individuals[j])))
    return np.mean(distances) if distances else 0

def update_mutation_rate(generation, diversity, base_rate=0.2, diversity_threshold=0.1):
    """ Dynamically adjust mutation rate based on generation and population diversity. """
    if diversity < diversity_threshold:
        return base_rate / (1 + 0.05 * generation)  # Decrease mutation rate over generations if diversity is low
    return min(base_rate * (1 + 0.05 * generation), 1)  # Increase mutation rate if diversity is high

def evaluate_co_wrapper(args_tuple):
    player_1_ind, player_2_population = args_tuple
    return evaluate_co(player_1_ind, player_2_population)

def evaluate_co(player_1_ind, player_2_population):
    """
    Evaluate an individual's fitness by competing against a selection of opponents.
    """
    p1_win_count = 0

    for player_2_ind in player_2_population:
        local_args = argparse.Namespace(**vars(args))
        local_args.player_1_weights = player_1_ind[:WEIGHTS_PER_PLAYER]
        local_args.player_2_weights = player_2_ind[WEIGHTS_PER_PLAYER:]

        p1_wins, _ = Simulator(local_args).autoplay()
        p1_win_count += p1_wins

    return p1_win_count,

def evolve_population(population, opponents, toolbox):
    """
    Evolve a population for one generation.
    """
    # Prepare tuples of arguments for multiprocessing
    args_tuples = [(ind, opponents) for ind in population]

    # Use multiprocessing pool's map function
    fitnesses = toolbox.map(evaluate_co_wrapper, args_tuples)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Apply the genetic operators
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_RATE:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTATION_RATE:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    return offspring


toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, RANGE_LIMIT)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=WEIGHTS_PER_PLAYER * 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("evaluate_co", evaluate_co)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUTATION_RATE)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    num_processes = 12
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)

    # Create two separate populations
    population1 = toolbox.population(n=POPULATION_SIZE)
    population2 = toolbox.population(n=POPULATION_SIZE)

    # Initialize both populations with the starting weights
    for ind in population1 + population2:
        ind[:len(STARTING_WEIGHTS)] = STARTING_WEIGHTS

    historical_best1 = []
    historical_best2 = []
    diversity_history1 = []
    diversity_history2 = []

    # Evolution loop
    for gen in range(NUM_GENERATIONS):
        # Evolve both populations
        population1 = evolve_population(population1, population2, toolbox)
        population2 = evolve_population(population2, population1, toolbox)

        # Calculate diversity and adjust mutation rate
        current_diversity1 = diversity(population1)
        current_diversity2 = diversity(population2)
        diversity_history1.append(current_diversity1)
        diversity_history2.append(current_diversity2)

        dynamic_mutation_rate1 = update_mutation_rate(gen, current_diversity1)
        dynamic_mutation_rate2 = update_mutation_rate(gen, current_diversity2)
        for ind in population1:
            toolbox.mutate(ind, indpb=dynamic_mutation_rate1)
            del ind.fitness.values
        for ind in population2:
            toolbox.mutate(ind, indpb=dynamic_mutation_rate2)
            del ind.fitness.values

        # Select the next generation
        population1 = toolbox.select(population1, k=len(population1))
        population2 = toolbox.select(population2, k=len(population2))

        # Log best weights
        best_ind1 = tools.selBest(population1, 1)[0]
        best_ind2 = tools.selBest(population2, 1)[0]
        log_best_weights(best_ind1, f"population1_gen_{gen}")
        log_best_weights(best_ind2, f"population2_gen_{gen}")

        # Store historical best for reintroduction
        if gen % 10 == 0:
            historical_best1.append(best_ind1)
            historical_best2.append(best_ind2)

        
        # Reintroduce historical best and new random individuals for diversity
        if gen % 25 == 0:
            num_to_replace = min(2, len(historical_best1))
            for i in range(num_to_replace):
                population1[random.randint(0, len(population1) - 1)] = toolbox.clone(historical_best1[i])
                population2[random.randint(0, len(population2) - 1)] = toolbox.clone(historical_best2[i])
        
        if gen % 50 == 0 and gen > 0:
            population1[random.randint(0, len(population1) - 1)] = toolbox.individual()
            population2[random.randint(0, len(population2) - 1)] = toolbox.individual()

    # Final output
    final_best_ind1 = tools.selBest(population1, 1)[0]
    final_best_ind2 = tools.selBest(population2, 1)[0]
    print("Final Best Individual in Population 1: %s, %s" % (final_best_ind1, final_best_ind1.fitness.values))
    print("Final Best Individual in Population 2: %s, %s" % (final_best_ind2, final_best_ind2.fitness.values))

    pool.close()
    pool.join()

if __name__ == '__main__':
    time_taken = 0
    main()