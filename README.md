# COMP-424 AI Final Project

## Preface
This GitHub repository features an engaging Colosseum Survival AI environment, tailored to test and evolve the `StudentAgent` from a strategic board game to a dynamic survival scenario. Utilizing a sophisticated genetic algorithm, the repository facilitates the enhancement of the agent's decision-making and survival strategies in a competitive Colosseum setting. It serves as an innovative platform for demonstrating the adaptability and robustness of the AI agent in diverse and challenging environments. Perfect for AI enthusiasts and researchers keen on exploring advanced AI behavior in survival simulations.

## Description
### Student Agent
This repository contains the implementation of `StudentAgent`, an advanced AI agent designed for a strategic two-player board game where players move on a grid and place barriers to control space. The `StudentAgent` uses sophisticated game theory algorithms and heuristics to make intelligent decisions. At its core, the agent employs:

- alpha-beta pruning with a transposition table
- iterative deepening search strategies 
- a genetic algorithm with crossover and mutation to train heuristic weights 

to efficiently explore possible moves, optimizing its decisions within a limited time frame. A transposition table is utilized to cache and recall previously evaluated game states, significantly speeding up the decision process. The agent's behavior is further refined by several custom heuristic functions, each assessing the game state from different strategic perspectives such as expansion potential, aggression, positional advantage, and the ability to extend barriers effectively. These heuristics are weighted and combined to form a comprehensive evaluation of each move. `StudentAgent` is an example of applying complex algorithms and heuristics to create a competitive and adaptable AI for grid-based board games.

### Genetic Algorithm
This repository hosts a genetic algorithm for optimizing the heuristic weights of `StudentAgent`, an AI agent for a strategic two-player board game. Key features include:

- **Robust Evolutionary Approach**: Fine-tunes decision-making against various enemy strategies.
- **Customizable Parameters**: Includes population size, mutation and crossover rates, and number of weights.
- **Fitness Evaluation**: Based on performance against diverse enemy agents.
- **Evolutionary Operations**: Utilizes two-point crossover, Gaussian mutation, and tournament selection.
- **Diversity Maintenance**: Periodically reintroduces successful individuals to the population.
- **Detailed Logging**: Tracks best weights and win percentages against opponents for each generation.
- **Outcome**: Produces a strategically versatile AI agent capable of effective competition.

## Testing
This repository includes a simulator for running and testing a strategic two-player board game where AI agents, including the `StudentAgent`, compete against each other. The simulator can be run from an Integrated Development Environment (IDE) or directly from the terminal, offering flexibility for different development and testing scenarios.

**Installation:**

Before running the simulator, you need to install the necessary Python libraries. You can install these dependencies using the following command:

```bash
pip install matplotlib==3.5.1 numpy pytest==7.0.1 tqdm==4.62.3 click==8.0.4 deap
```

These libraries include 
- `matplotlib` for plotting and visualizing game states
- `numpy` for numerical operations
- `pytest` for running tests
- `tqdm` for progress bars in the terminal
- `click` for creating command-line interfaces
- `deap` for evolutionary algorithms

**Running the Simulator:**

- **From an IDE**: Simply import the required modules, instantiate the `Simulator` class with desired arguments, and call the `run()` or `autoplay()` method based on your testing needs.
- **From the Terminal**: Use command-line arguments to customize the simulation. Here's an example command to run the simulator: 
  ```bash
  python simulator.py --player_1 student_agent --player_2 random_agent --board_size 10 --autoplay
  ```

**Command-Line Arguments:**

- `--player_1`: Specifies the first player agent (default: "random_agent").
- `--player_2`: Specifies the second player agent (default: "random_agent").
- `--board_size`: Sets a fixed board size for the game.
- `--board_size_min`: In autoplay mode, sets the minimum board size (default: 6).
- `--board_size_max`: In autoplay mode, sets the maximum board size (default: 12).
- `--display`: Enables the display of the game board (default: False).
- `--display_delay`: Sets the delay between moves in display mode (default: 0.4 seconds).
- `--display_save`: Enables saving of the game state (default: False).
- `--display_save_path`: Sets the path for saving game plots (default: "plots/").
- `--autoplay`: Enables autoplay mode to run multiple simulations automatically (default: False).
- `--autoplay_runs`: Specifies the number of games to run in autoplay mode (default: 1000).

## About
This is a class project for COMP 424, McGill University

## License

[MIT](LICENSE)
