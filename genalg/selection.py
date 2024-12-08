import numpy as np


class Selection:
    """
    Base class inherited by selection algorithms. The get_singe method is
    implemented differently for the different versions.
    """
    def __init__(self):
        self.genomes = None
        self.fitness = None
        self.indices = None
    
    def set_population(self, genomes, fitness):
        self.genomes = genomes
        self.fitness = fitness
        self.indices = np.arange(genomes.shape[0])

    def get_single(self):
        """
        Override in sub classes.
        """
        raise NotImplementedError
    
    def get_n_unique(self, n, exclude_indices=np.empty(0)):
        """
        Select n genomes from the population. The optional exclude_indices 
        parameter may contain an array of indices self.genomes that 
        should not be included in the selection. Returns the selection as
        indices for self.genomes.
        """
        
        # TODO cleaner code
        # TODO option for allow repeats or force unique
        
        selected = []
        while len(selected) < n:
            selected_index = self.get_single()
            if len(exclude_indices) > 0 and np.isin(selected_index, exclude_indices).all():
                continue
            else:
                selected.append(selected_index)
        return selected
                


class Ranked(Selection):
    """
    Selects two unqiue individuals with a probability inversely proportional 
    to their rank.
    """
    def __init__(self):
        super().__init__()

    def get_single(self):
        order = np.argsort(self.fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(self.fitness)) + 1
        p = ranks / ranks.sum()
        selection_index = np.random.choice(self.indices, 1, replace=False, p=p)
        return selection_index[0]



class Roulette(Selection):
    """
    Selects two unqiue individuals with a probability proportional to fitness.
    """
    def __init__(self):
        super().__init__()

    def get_single(self):
        p = self.fitness / self.fitness.sum()
        selection_index = np.random.choice(self.indices, 1, replace=False, p=p)
        return selection_index[0]



class Tournament(Selection):
    """
    Selects two unqiue individuals based on tournament selection. A larger
    k (tournament size) gives a greater relative selection probability on the
    best fitness individuals. A k of 2 equals rank-based selection.
    """
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def get_single(self):
        tournament_picks = np.random.choice(
            self.indices, self.k, replace=False)
        picks_fitness = self.fitness[tournament_picks]
        selection_index = tournament_picks[np.argmax(picks_fitness)]
        return selection_index
