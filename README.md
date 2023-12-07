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
This repository features a sophisticated genetic algorithm designed to optimize the heuristic weights of StudentAgent, an AI agent for a strategic two-player board game. The algorithm employs a robust evolutionary approach, fine-tuning the agent's decision-making abilities against a diverse set of enemy strategies. It is configured with parameters like population size, mutation and crossover rates, and the number of heuristic weights, facilitating extensive experimentation and customization. The fitness of each candidate solution is evaluated based on performance against various predefined enemy agents, ensuring that the agent learns to adapt to different play styles. Key features include two-point crossover, Gaussian mutation for generating new solutions, tournament selection for evolution, and periodic reintroduction of historically successful individuals to maintain genetic diversity. Detailed logging of the best-performing weights and win percentages against different opponents after each generation provides valuable insights into the agent's learning progress. The culmination of this training process is an AI agent well-versed in diverse strategies and capable of competing effectively in the specified board game.

## Testing
This repository includes a simulator for running and testing a strategic two-player board game where AI agents, including the `StudentAgent`, compete against each other. The simulator can be run from an Integrated Development Environment (IDE) or directly from the terminal, offering flexibility for different development and testing scenarios.

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
