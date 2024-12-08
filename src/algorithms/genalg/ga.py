import numpy as np


"""
Sub module status

SUB MODULE            │ TODO 
──────────────────────┼────────────────────────────────────────────
ga                    │ rewrite, cleanup, docstrings
Selection             │ cleanup, docstrings
Crossover             │ cleanup, docstrings
Mutation              │ Up-to-date
Social distaster      │ Implement packing and judgement day
        
"""                



# TODO: popoulation gets smaller and smallers


class GeneticAlgorithm:
    
    def __init__(self, selection, crossover, mutations, disaster=None,
                 num_elites=0):
        """
        Parameters
        ----------
        selection : Selection instance
            Selects individuals from the population based on fitness.
        crossover : Crossover instance
            Generates offspring from selected parent individuals.
        mutations : List of mutation instances
            Mutates genomes, passed as a list to allow for different types of
            mutation with different probability functions. Applied in order of 
            the passed list.
        social_disaster : disaster instance 
            Initialized
        elitism : Integer
            The number of elites to be passed without evolving.
        sd_sim : Float betwen 0 and 1, optional
            The similiraty threshold that triggers a disaster. 
        """
        self.selection = selection
        self.crossover = crossover
        self.mutations = mutations
        self.disaster = disaster
        self.num_elites = num_elites

    def evolve_population(self, genomes, fitness, force_disaster=False):
        """

        Parameters
        ----------
        new_genomes : numpy array of floats with shape (pop_size, genome_size)
            Population genomes.
        fitness : numpy array
            1d Array with shape (population_size) in the same order als genomes.

        Returns
        -------
        new_population_genomes : numpy array
            The evolved generation genomes.
        """
        
        # Update instances
        self.selection.set_population(genomes, fitness)
        if self.disaster is not None:
            self.disaster.set_population(genomes, fitness)


        # NEW: REMOVED MUTATION FUNC, SET EXTERNALLY
        #for mutation in self.mutations:
        #    mutation.update_p(fitness)


        # Extract elites
        if self.num_elites > 0:
            elite_indices = np.argsort(fitness)[-self.num_elites:]
            elite_genomes = genomes[elite_indices]
            
        # Social disaster
        if self.disaster is not None and (force_disaster or self.disaster.trigger()):
            print('DISASTER')
            new_genomes = self.disaster.apply(elite_indices)
            return new_genomes
                
        # Run evolution to get new mutated offspring indivuals      
        
        # TODO: init as numpy array?
        offspring_genomes = []
        while len(offspring_genomes) < (len(genomes) - self.num_elites):
            
            # TODO: why not return as genomes?
            parent_idx = self.selection.get_n_unique(2)
            
            # TODO: add parameter for number of offspring for get_offspring
            offspring = self.crossover.get_offspring(genomes[parent_idx])
            for m in self.mutations:
                offspring = m.mutate_genome(offspring)
           
            offspring_genomes.append(offspring)
        offspring_genomes = np.array(offspring_genomes)
        
        # Combining elites and and offspring into new population
        if self.num_elites > 0:
            new_genomes = np.vstack((elite_genomes, offspring_genomes))
        else:
            new_genomes = offspring_genomes

        return new_genomes

