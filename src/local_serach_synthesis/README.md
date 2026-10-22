# DFA Optimization with Local Search

This program demonstrates the use of local search strategies to optimize Deterministic Finite Automata (DFA) for classifying strings. It supports several search strategies, including Hill Climbing, Simulated Annealing, and Tabu Search.

## Features
- Implements multiple local search heuristics:
  - **Hill Climbing** (default)
  - **Simulated Annealing**
  - **Tabu Search**
- Visualizes the resulting DFA using Graphviz.
- Automatically generates positive and negative samples for the language \((ab)^*\).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/dfa-optimization.git
   cd automata-local-search
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have `Graphviz` installed on your system for DFA visualization. You can install it using:
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get install graphviz
     ```
   - On macOS:
     ```bash
     brew install graphviz
     ```
   - On Windows: Download from the [Graphviz website](https://graphviz.org/download/).

## Usage

Run the script using Python. You can specify the search strategy using the `--strategy` flag.

### Syntax
```bash
python your_script.py [--strategy STRATEGY]
```

### Options
- `--strategy`: The search strategy to use. Choices:
  - `hill_climbing`: Simple greedy optimization.
  - `simulated_annealing`: Avoid local optima using a probabilistic acceptance function.
  - `tabu_search`: Prevent cycling back to recently explored states.

Default: `hill_climbing`

### Example Commands

#### Run Hill Climbing (default):
```bash
python your_script.py
```

#### Run Simulated Annealing:
```bash
python your_script.py --strategy simulated_annealing
```

#### Run Tabu Search:
```bash
python your_script.py --strategy tabu_search
```

## Output

1. The script prints the progress of the search strategy and updates on score improvements.
2. The optimized DFA is visualized and saved as `dfa_visualization.png` in the working directory.
3. Final score and DFA performance details are displayed at the end of execution.

## Example Output

```bash
Starting Hill Climbing...
Iteration 1: Improved score to 0.68
Iteration 5: Improved score to 0.82
Iteration 12: Improved score to 1.0
Target score 1.0 achieved at iteration 12.
Final Score: 1.0
Automaton visualization saved as dfa_visualization.png
```

## Customization
You can modify the following parameters in the script:
- **Alphabet**: The set of symbols used in the language.
- **Number of States**: The size of the DFA.
- **Sample Generation**: Adjust the number and length of positive/negative samples.

## Contributing
Feel free to fork the repository and submit pull requests to improve the project or add new features.

## License
This project is licensed under the MIT License.