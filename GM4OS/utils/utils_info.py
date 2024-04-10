import csv

def verbose_reporter(generation, pop_val_fitness, pop_test_fitness, timing, nodes):
    """
        Prints a formatted report of generation, fitness values, timing, and node count.

        Parameters
        ----------
        generation : int
            Current generation number.
        pop_val_fitness : float
            Population's validation fitness value.
        pop_test_fitness : float
            Population's test fitness value.
        timing : float
            Time taken for the process.
        nodes : int
            Count of nodes in the population.

        Returns
        -------
        None
            Outputs a formatted report to the console.
    """
    digits_generation = len(str(generation))
    digits_val_fit = len(str(pop_val_fitness))
    digits_test_fit = len(str(pop_test_fitness))
    digits_timing = len(str(timing))
    digits_nodes = len(str(nodes))

    if generation == 0:
        print(
            "                                          Verbose Reporter                                              ")
        print(
            "------------------------------------------------------------------------------------------------------------------")
        print(
            "| Generation |   Validation Fitness   |      Test Fitness      |        Timing          |         Nodes          |")
        print(
            "------------------------------------------------------------------------------------------------------------------")
        print("|" + " "*6 + str(generation) + " "*(6 - digits_generation) + "|"
              + " "*3 + str(pop_val_fitness)
              + " "*(21 - digits_val_fit) + "|" + " "*3 + str(pop_test_fitness) + " "*(21 - digits_test_fit) + "|" +
              " "*3 + str(timing) + " " * (21 - digits_timing) + "|" +
              " "*10 + str(nodes) + " " * (14 - digits_nodes) + "|")
    else:
        print("|" + " " * 6 + str(generation) + " " * (6 - digits_generation) + "|"
              + " " * 3 + str(pop_val_fitness)
              + " " * (21 - digits_val_fit) + "|" + " " * 3 + str(pop_test_fitness) + " " * (
                          21 - digits_test_fit) + "|" +
              " "*3 + str(timing) + " " * (21 - digits_timing) + "|" +
              " "*10 + str(nodes) + " " * (14 - digits_nodes) + "|")


def logger(path, generation, pop_val_fitness, timing, nodes,
           pop_test_report = None, run_info = None):
    """
        Logs information into a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        generation : int
            Current generation number.
        pop_val_fitness : float
            Population's validation fitness value.
        timing : float
            Time taken for the process.
        nodes : int
            Count of nodes in the population.
        pop_test_report : float or list, optional
            Population's test fitness value(s). Defaults to None.
        run_info : list, optional
            Information about the run. Defaults to None.

        Returns
        -------
        None
            Writes data to a CSV file as a log.
    """

    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if run_info != None:
            infos = [run_info[0], run_info[1], run_info[2], generation, pop_val_fitness, timing, nodes]
        else:
            infos = [generation, pop_val_fitness, timing]
        if pop_test_report != None and isinstance(pop_test_report, list):
            infos.extend(pop_test_report)
        elif pop_test_report != None:
            infos.extend([pop_test_report])
        writer.writerow(infos)