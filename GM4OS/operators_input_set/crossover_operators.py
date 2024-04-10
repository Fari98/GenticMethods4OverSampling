import random

def crossover_input_set(input_set1, input_set2):

    """
    Performs crossover between two input sets.

    Parameters
    ----------
    input_set1 : list
        First input set for crossover.

    input_set2 : list
        Second input set for crossover.

    Returns
    -------
    tuple
        Two new input sets obtained after crossover.
    """


    crossover_point = random.randint(1, len(input_set1))

    new_input_set1 = input_set1[:crossover_point].copy() + input_set2[crossover_point:].copy()

    new_input_set2 = input_set2[:crossover_point].copy() + input_set1[crossover_point:].copy()

    return new_input_set1, new_input_set2