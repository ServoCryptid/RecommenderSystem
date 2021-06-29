import logging
import random

import numpy as np
import pandas as pd
import neat.utils as utils
from neat.genotype.genome import Genome
from neat.species import Species
from neat.crossover import crossover
from neat.mutation import mutate
import time

from neat.visualize import draw_net

logger = logging.getLogger(__name__)


class Population:
    __global_innovation_number = 0
    current_gen_innovation = []  # Can be reset after each generation according to paper
    fitness_not_improved = 0

    def __init__(self, config):
        self.Config = config()
        # pool = Pool(processes=8)
        self.population = self.set_initial_population()
        print(f"initial population: {self.population}")
        self.species = []

        for genome in self.population:
            self.speciate(genome, 0)

    def run(self):
        last_fitness = []
        folder_path = r"C:\Users\laris\PycharmProjects\KaggleRS\experiments\overall_view"
        acc_arr = []
        for generation in range(1, self.Config.NUMBER_OF_GENERATIONS + 1):
            start_time = time.time()
            # Get Fitness of Every Genome
            for genome in self.population:
                genome.fitness = max(0, self.Config.fitness_fn(genome))

            best_genome = utils.get_best_genome(self.population)

            # Reproduce
            all_fitnesses = []
            remaining_species = []

            for species, is_stagnant in Species.stagnation(self.species, generation):
                if is_stagnant:
                    self.species.remove(species)
                else:
                    all_fitnesses.extend(g.fitness for g in species.members)
                    remaining_species.append(species)

            min_fitness = min(all_fitnesses)
            max_fitness = max(all_fitnesses)

            last_fitness.append(max_fitness)

            fit_range = max(1.0, (max_fitness-min_fitness))
            for species in remaining_species:
                # Set adjusted fitness
                avg_species_fitness = np.mean([g.fitness for g in species.members])
                species.adjusted_fitness = (avg_species_fitness - min_fitness) / fit_range

            adj_fitnesses = [s.adjusted_fitness for s in remaining_species]
            adj_fitness_sum = sum(adj_fitnesses)

            # Get the number of offspring for each species
            new_population = []
            for species in remaining_species:
                if species.adjusted_fitness > 0:
                    size = max(2, int((species.adjusted_fitness/adj_fitness_sum) * self.Config.POPULATION_SIZE))
                else:
                    size = 2

                # sort current members in order of descending fitness
                cur_members = species.members
                cur_members.sort(key=lambda g: g.fitness, reverse=True)
                species.members = []  # reset

                # save top individual in species
                new_population.append(cur_members[0])
                size -= 1

                # Only allow top x% to reproduce
                purge_index = int(self.Config.PERCENTAGE_TO_SAVE * len(cur_members))
                purge_index = max(2, purge_index)
                cur_members = cur_members[:purge_index]

                for i in range(size):
                    parent_1 = random.choice(cur_members)
                    parent_2 = random.choice(cur_members)

                    child = crossover(parent_1, parent_2, self.Config)
                    mutate(child, self.Config)
                    new_population.append(child)

            # Set new population
            self.population = new_population
            Population.current_gen_innovation = []

            # Speciate
            for genome in self.population:
                self.speciate(genome, generation)

            if not self.fitness_improved(last_fitness):
                mae, r2 = self.Config.get_preds_and_labels(best_genome, generation)
                # return best_genome, generation, mae, r2  # TODO: uncomment this if you want early stopping
                acc_arr.append([generation, self.Config.NUMBER_OF_GENERATIONS, self.Config.POPULATION_SIZE,
                                mae, r2, "yes", time.time()-start_time])
            else:
                mae, r2 = self.Config.get_preds_and_labels(best_genome, generation)
                acc_arr.append([generation, self.Config.NUMBER_OF_GENERATIONS, self.Config.POPULATION_SIZE,
                                mae, r2, "no", time.time()-start_time])

            self.get_details_best_genome(generation, best_genome)

        pd.DataFrame(data=acc_arr, columns=["generation", "num_generations", "num_population",
                                            "mae", "r2", "fitness_improv", "time_seconds"]).to_csv(f"{folder_path}/{self.Config.NUMBER_OF_GENERATIONS}_"
                                                                 f"p{self.Config.POPULATION_SIZE}")
        return None, None, None, None

    def speciate(self, genome, generation):
        """
        Places Genome into proper species - index
        :param genome: Genome be speciated
        :param generation: Number of generation this speciation is occuring at
        :return: None
        """
        for species in self.species:
            if Species.species_distance(genome, species.model_genome) <= self.Config.SPECIATION_THRESHOLD:
                genome.species = species.id
                species.members.append(genome)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(self.species), genome, generation)
        genome.species = new_species.id
        new_species.members.append(genome)
        self.species.append(new_species)

    def assign_new_model_genomes(self, species):
        species_pop = self.get_genomes_in_species(species.id)
        species.model_genome = random.choice(species_pop)

    def get_genomes_in_species(self, species_id):
        return [g for g in self.population if g.species == species_id]

    def set_initial_population(self):
        pop = []

        for i in range(int(self.Config.POPULATION_SIZE)):
            new_genome = Genome()
            inputs = []
            outputs = []
            bias = None

            # Create nodes
            for j in range(self.Config.NUM_INPUTS):
                # print(f"\nnum inputs: {j}/{self.Config.NUM_INPUTS}")
                n = new_genome.add_node_gene('input')
                inputs.append(n)

            for j in range(self.Config.NUM_OUTPUTS):
                # print(f"\nnum outputs: {j}/{self.Config.NUM_OUTPUTS}")
                n = new_genome.add_node_gene('output')
                outputs.append(n)

            if self.Config.USE_BIAS:
                bias = new_genome.add_node_gene('bias')

            # Create connections
            cnt = 0
            for input in inputs:
                for output in outputs:
                    # print(f"\ncreating connections {cnt}")
                    # cnt+=1
                    new_genome.add_connection_gene(input.id, output.id)
            cnt = 0
            if bias is not None:
                for output in outputs:
                    # print(f"\nadding bias {cnt}")
                    # cnt+=1
                    new_genome.add_connection_gene(bias.id, output.id)

            pop.append(new_genome)

        return pop

    @staticmethod
    def get_new_innovation_num():
        # Ensures that innovation numbers are being counted correctly
        # This should be the only way to get a new innovation numbers
        ret = Population.__global_innovation_number
        Population.__global_innovation_number += 1
        return ret

    def fitness_improved(self, fitness_arr, num_gen_to_wait=20):
        '''
        If fitness doesn't improve after a defined number of generations, end the process
        :return: True/False
        '''

        improvement_value = 0.01

        if len(fitness_arr) > 2:
            diff = (fitness_arr[-1] - fitness_arr[-2])/fitness_arr[-2]
            if np.abs(diff) > improvement_value:
                Population.fitness_not_improved = 0
            else:
                Population.fitness_not_improved += 1

            print(f"Fitness arr: {fitness_arr}")
            print(f"Last:{fitness_arr[-1]}; {num_gen_to_wait} element: {fitness_arr[-2]}")

        print(f"FITNESS IMPROVEMENT COUNT: {Population.fitness_not_improved}")
        if Population.fitness_not_improved == num_gen_to_wait:
            return False

        return True

    def get_details_best_genome(self, generation, solution):
        "Get details about the best genome from each generation"

        num_of_solutions = 0

        avg_num_hidden_nodes = 0
        min_hidden_nodes = 100000
        max_hidden_nodes = 0
        found_minimal_solution = 0

        avg_num_generations = 0
        min_num_generations = 100000

        avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
        min_num_generations = min(generation, min_num_generations)

        num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
        avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
        min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
        max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)

        if num_hidden_nodes == 1:
            found_minimal_solution += 1

        num_of_solutions += 1
        draw_net(solution, filename='./images/solution-' + f"{generation}_{self.Config.NUMBER_OF_GENERATIONS}_"
                                                                 f"p{self.Config.POPULATION_SIZE}",
                 show_disabled=True)

        print('Total Number of Solutions: ', num_of_solutions)
        print('Average Number of Hidden Nodes in a Solution', avg_num_hidden_nodes)
        print('Solution found on average in:', avg_num_generations, 'generations')
        print('Minimum number of hidden nodes:', min_hidden_nodes)
        print('Maximum number of hidden nodes:', max_hidden_nodes)
        print('Minimum number of generations:', min_num_generations)
        print('Found minimal solution:', found_minimal_solution, 'times')

