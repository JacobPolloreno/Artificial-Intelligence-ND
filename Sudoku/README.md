# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: The naked twins technique allows us to take two boxes within the same unit, our domain
knowledge about the game's rules, and use inferences to perform elimination on their peers.
This technique allows us to reduce the number of legal possibilities for peers within the
same unit which will continue to propogate. More specifically, if we have two boxes on the
same column unit with values '12' and '12' then we know that either '1' or '2' will be
distributed among those two boxes and not their peers. Therefore, we can put constraints
on their peers by eliminating illegal possibilities(i.e. '1' and '2'). Later we may be able
to further reduce the legal moves of other boxes.  

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: With constraint propagation we can limit the legal moves on the diagonal units of a 
sudoku puzzle. First we must add the diagonal units to the main unit list along with columns,
rows, and square boxes. By doing this, we're constraining the legal moves of non-diagonal
units which then reduces the possibilites for each diagonal box.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.
