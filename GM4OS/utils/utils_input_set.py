import random

def create_random_input_set(sizes, minority_class_sets):

    """
    Generates a random input set composed of samples from minority class sets.

    Parameters
    ----------
    sizes : list
        List of sizes for each subset of the input set.

    minority_class_sets : list
        List of minority class sets used for sampling.

    Returns
    -------
    list
        A list representing the random input set with samples from the minority class sets.
    """

    individual = []
    for i in range(len(sizes)):
        individual.extend([random.sample(minority_class_sets[i], k=2) for _ in range(int(sizes[i]))])

    return individual