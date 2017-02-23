import utils

assignments = []


def assign_value(values, box, value):
    """
    Assigns a value to a given box. If it updates the board record it.

    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
        box: box(e.g. 'B3') that'll be updated
        value: new value, digit between 1-9

    Returns:
        the values dictionary with updated board
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def naked_twins(values):
    """
    Eliminate values using the naked twins strategy.

    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    def find_twins_and_eliminate():
        """
        Find all instances of naked twins in values dict

        Call eliminate_twins subroutine after instances are found for a
        particular unit
        """

        for unit in utils.unitlist:
            unit_values = [values[box] for box in unit]
            found_twins = {twin: values[twin] for twin in unit if
                           unit_values.count(values[twin]) == 2 and
                           len(values[twin]) == 2}
            peers = [box for box in unit if box not in found_twins.keys()]
            eliminate_twins(set(found_twins.values()), peers)

    def eliminate_twins(twin_values, peers):
        """
        Eliminate the naked twins as possibilities for their peers
        """

        for val in twin_values:
            for peer in peers:
                new_value = values[peer]
                for digit in val:
                    new_value = new_value.replace(digit, '')
                assign_value(values, peer, new_value)

    find_twins_and_eliminate()

    return values


def eliminate(values):
    """
    Go through all the boxes, and whenever there is a box with a value,
        eliminate this value from the values of all its peers.

    Args:
        A sudoku in dictionary form.

    Returns:
        The resulting sudoku in dictionary form.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in utils.peers[box]:
            new_value = values[peer].replace(digit, '')
            values = assign_value(values, peer, new_value)
    return values


def only_choice(values):
    """
    Go through all the units, and whenever there is a unit with a value
        that only fits in one box, assign the value to this box.

    Args:
        A sudoku in dictionary form.

    Returns:
        The resulting sudoku in dictionary form.
    """
    for unit in utils.unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values = assign_value(values, dplaces[0], digit)
    return values


def reduce_puzzle(values):
    """
    Iterate eliminate() and only_choice(). If at some point, there is a
        box with no available values, return False.
    If the sudoku is solved, return the sudoku.
    If after an iteration of both functions, the sudoku remains the same,
        return the sudoku.

    Args:
        A sudoku in dictionary form.

    Returns:
        The resulting sudoku in dictionary form.
    """
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys()
                                    if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys()
                                   if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """
    Using depth-first search and propagation, try all possible values.

    Args:
        A sudoku in dictionary form.

    Returns:
        The resulting sudoku in dictionary form.
    """

    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False  # Failed earlier
    if all(len(values[s]) == 1 for s in utils.boxes):
        return values  # Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n, s = min((len(values[s]), s) for s in utils.boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt


def solve(grid):
    """
    Find the solution to a Sudoku grid.

    Args:
        grid(string): a string representing a sudoku grid.
            Example:
        '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns:
        The dictionary representation of the final sudoku grid.
            False if no solution exists.
    """
    return search(utils.grid_values(grid))


if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    utils.display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)
    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
