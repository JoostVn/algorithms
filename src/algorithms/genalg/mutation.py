import numpy as np


class Mutation:
    
    def __init__(self, p, genome_domain):
        
        """
        Base mutations class. 
        
        Parameters
        ----------
        p: float
            Mutation probability
        genome_domain : tuple of size 2
            A tuple depicting the minimum and maximum value for genes in the
            population.
            
        """
        assert type(p) in (float, int)
        
        self.genome_domain = genome_domain
        self.p = p
    
    def mutate_genome(self, genome):
        """
        Iterate over all genes in a single genome, apply mutations based on the
        mutation probability self.p, and return the new mutated genome.

        Parameters
        ----------
        genome : 1d numpy array
            Input genome to be mutated.

        Returns
        -------
        new_genome : 1d numpy array
            Mutated output genome.
        """
        new_genome = genome.copy()
        for i, gene in enumerate(genome):
            if np.random.uniform(0,1) < self.p:
                new_genome[i] = self.mutate_gene(new_genome[i])
        return new_genome
        
    def mutate_gene(self, gene):
        """
        Override in sub classes.
        """
        raise NotImplementedError



class UniformReplacement(Mutation):
    
     def __init__(self, p, genome_domain=(-1,1)):
        """
        Mutated genes are replaced by a uniformly distributed random variable.
        """
        super().__init__(p, genome_domain)
         
     def mutate_gene(self, gene):
        new_gene = np.random.uniform(*self.genome_domain)
        return new_gene


class Adjustment(Mutation):
    
    def __init__(self, p, genome_domain=(-1,1), adjustment_domain=(-0.1,0.1)):
        """
        Mutated genes are adjusted by substracting or adding a random value.
        
        Extra parameters
        ----------
        adjustment_domain : tuple of size 2
            Determines the (min, max) range of random values added to or 
            subtracted from each gene.
        """
        super().__init__(p, genome_domain)
        self.adjustment_domain = adjustment_domain
    
    def mutate_gene(self, gene):
        new_gene = gene + np.random.uniform(*self.adjustment_domain)
        new_gene = np.clip(new_gene, *self.genome_domain)
        return new_gene
        


class Thrors(Mutation):

     def __init__(self, p, genome_domain=(-1,1), max_points=3):
        super().__init__(p, genome_domain)
        self.max_points = max_points

     def mutate_genome(self, genome):
        """
        https://arxiv.org/ftp/arxiv/papers/1203/1203.3099.pdf
        """
        new_genome = genome.copy()

        if np.random.uniform() < self.p:
            n_index_points = np.random.randint(2, self.max_points+1)
            indices = np.indices(genome.shape).flatten()
            selection = np.random.choice(
                indices, n_index_points, replace=False)
            
            pairs = zip(selection, np.roll(selection, -1))
            for (i,j) in pairs:
                new_genome[j] = genome[i]
            
        return new_genome
     

class RSM(Mutation):

     def __init__(self, p, genome_domain=(-1,1)):
        super().__init__(p, genome_domain)

     def mutate_genome(self, genome):
        """
        https://arxiv.org/ftp/arxiv/papers/1203/1203.3099.pdf
        """
        new_genome = genome.copy()
        n = new_genome.shape[0]

        if np.random.uniform() < self.p:
            i = np.random.randint(0, n + 1)
            j = np.random.randint(i, n + 1)
            new_genome[i:j] = np.flip(new_genome[i:j])
            
        return new_genome
      



if __name__ == '__main__':


    genome = np.arange(10)
    np.random.shuffle(genome)
    print(genome)


    RSM = Thrors(1.0)

    genome = RSM.mutate_genome(genome)
    print(genome)
    

    
