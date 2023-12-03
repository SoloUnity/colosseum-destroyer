from simulator import Simulator
import argparse
import random
from deap import base, creator, tools, algorithms
import os

# Genetic Algorithm parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
WEIGHTS_PER_PLAYER = 5  # Number of weights per player
LOG_DIR = "logs"

# Additional variables for dynamic range adjustment
RANGE_LIMIT = 2.5
RANGE_EXPANSION_FACTOR = 1.5
EXPANSION_THRESHOLD = 0.1  # Threshold to decide when to expand the range

# Initialize argparse.Namespace object with default values
args = argparse.Namespace(
    player_1="student_agent",
    player_2="student_agent",
    board_size=6,
    board_size_min=6,
    board_size_max=12,
    display=True,
    display_delay=0.4,
    display_save=False,
    display_save_path="plots/",
    autoplay=True,
    autoplay_runs=2
)


def adjust_weight_range(population):
    global RANGE_LIMIT
    max_weight = max(max(individual) for individual in population)
    min_weight = min(min(individual) for individual in population)

    # Check if the weights are near the boundaries
    if max_weight > RANGE_LIMIT - EXPANSION_THRESHOLD or min_weight < -RANGE_LIMIT + EXPANSION_THRESHOLD:
        RANGE_LIMIT *= RANGE_EXPANSION_FACTOR
        print(f"Expanding weight range to +/-{RANGE_LIMIT}")

# Create fitness function
def evaluate(individual):
    player_1_weights = individual[:WEIGHTS_PER_PLAYER]
    player_2_weights = individual[WEIGHTS_PER_PLAYER:]
    args.player_1_weights = player_1_weights
    args.player_2_weights = player_2_weights
    p1_win_count, p2_win_count = Simulator(args).autoplay()
    return p1_win_count - p2_win_count,  # Fitness could be the difference in wins

# Set up DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.5, 2.5)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=WEIGHTS_PER_PLAYER * 2)  # Twice the weights for 2 players
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUTATION_RATE)
toolbox.register("select", tools.selTournament, tournsize=3)

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Function to log best weights
def log_best_weights(best_weights, generation):
    with open(os.path.join(LOG_DIR, f"best_weights_gen_{generation}.txt"), 'w') as f:
        f.write(str(best_weights) + "\n")

# Genetic Algorithm
population = toolbox.population(n=POPULATION_SIZE)

for gen in range(NUM_GENERATIONS):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    best_ind = tools.selBest(population, 1)[0]

    adjust_weight_range(population)

    log_best_weights(best_ind, gen)
    print(f"Generation {gen}: Best individual is {best_ind}, {best_ind.fitness.values}")

# Find and print the best weights found
best_ind = tools.selBest(population, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
log_best_weights(best_ind, "final")
