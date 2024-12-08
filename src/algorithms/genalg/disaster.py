import numpy as np



class Disaster:
    
    def __init__(self, similarity_threshold, genome_domain):
        """
        Event that is triggered based on the similarity of genomes in the 
        population. When this similarity is larger that a given threshold,
        a "social disaster" is applied to the population that restores
        genetic diversity in an attempt to escape local minima. This is the 
        base class for disasters. sub classes override the "apply" function.

        Parameters
        ----------
        similarity_threshold : float between 0 and 1
            The similarity score at which a disaster is triggered. The score 
            should be set such that disasters are triggered only in the case
            of near-maximum achievable similarity in the population.
        genome_domain : tuple of size 2
            The (min, max) values for genes.
        """
        self.similarity_threshold = similarity_threshold    
        self.genome_domain = genome_domain
        self.genomes = None
        self.fitness = None
        self.indices = None
    
    def set_population(self, genomes, fitness):
        """
        Set the current population genomes and fitness.
        """
        self.genomes = genomes
        self.fitness = fitness    
        self.indices = np.arange(len(genomes))
    
    @staticmethod
    def get_genome_similarity(genomes, genome_domain):
        """
        Calculates the mean standard deviation, min/max normalized to (0, 1)
        based on the given (min, max) domain, and inverted such that 1 is 
        the most similarity and 0 is the least similarity. Defined as a static
        method so that it can be used outside of the disaster class without 
        requiring an instance.
        
        Parameters
        ----------
        genomes : numpy array of floats with shape (pop_size, genome_size)
            Population genomes.
        genome_domain : tuple of size 2
            The (min, max) values for genes.

        Returns
        -------
        similarity : float between 0 and 1
            The genome similarity score for the current population.
        """
        mean_std = genomes.std(axis=0).mean()
        scaled_mean_std = mean_std / np.std(genome_domain)
        similarity = 1 - scaled_mean_std
        return similarity

    def trigger(self):
        """
        Check if the current genome similarity is larger or equal to the 
        similarity trigger.
        """
        similarity = self.get_genome_similarity(
            self.genomes, self.genome_domain)
        return similarity >= self.similarity_threshold
        
    def apply(self, elite_indices):
        """
        Override in sub classes.
        """
        raise NotImplementedError



class SuperMutation(Disaster):
    
    def __init__(self, similarity_threshold, mutations, genome_domain=(-1,1)):
        """
        Performs extreme mutation over the full population with the 
        exception of elites.
        
        Extra parameters
        ----------
        mutations : List of mutation instances
            Mutation instances are passed as a list an applied in order. 
        """
        super().__init__(similarity_threshold, genome_domain)
        self.mutations = mutations
        
    def apply(self, elite_indices):
        """
        Apply the super mutation disaster.

        Parameters
        ----------
        elite_indices : 1d numpy array of integers
            Indices of self.genomes representing individuals that are selected
            as elites.

        Returns
        -------
        new_genomes : numpy array of floats with shape (pop_size, genome_size)
            New mutated population genomes.
        """
        non_elites_indices = ~np.in1d(self.indices, elite_indices)
        non_elites_genomes = self.genomes[non_elites_indices].copy()
        new_genomes = np.zeros(non_elites_genomes.shape, float)
        
        # Run mutations over the non elite genomes
        for i, genome in enumerate(non_elites_genomes):
            for m in self.mutations:
                genome = m.mutate_genome(genome)
            new_genomes[i] = genome
        
        # Add unmutated elite genomes to new_genomes and return array 
        new_population = np.vstack((self.genomes[elite_indices], new_genomes))
        return new_population
    


class JudgementDay(Disaster):
    
    # TODO
    
    def __init__(self, similarity_threshold, genome_domain=(1,1)):
        """
        Completely randomize the whole population with the exception of elites.
        """
        super().__init__(similarity_threshold, genome_domain)
        raise NotImplementedError



class Packing(Disaster):
    
    
    # TODO
    
    def __init__(self, similarity_threshold, genome_domain=(1,1)):
        """
        Group individuals with similar genomes. For each group, only let the 
        individual with the highest fitness remain unchanged, and fully 
        randomize the rest (with the exception of elites). Identical to 
        JudgementDay of all genomes are similar (and the number of groups is 1)
        """
        super().__init__(similarity_threshold, genome_domain)
        raise NotImplementedError







