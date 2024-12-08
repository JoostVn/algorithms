import numpy as np

# TODO docstrings
# TODO code cleanup
# TODO alwats return both offspring and handle pairs in GA module


class Multipoint:

    def __init__(self, n):
        """
        Produces offspring from two parents by combining their genomes by 
        stiching them together based on n random indices.
        """
        self.n = n
    
    def get_offspring(self, parent_genomes):
        genome_size = len(parent_genomes[0])
        indices = np.arange(genome_size)
        cross_indices = sorted(np.random.choice(indices, self.n, replace=False))
        offspring_genome = np.zeros(genome_size)
        current_parent = np.random.randint(0,2)
        for i in range(genome_size):
            if i in cross_indices:
                current_parent = 1 - current_parent
            offspring_genome[i] = parent_genomes[current_parent][i]
        return offspring_genome
        
    
    
class Bitmask:

    def __init__(self):
        """
        Produces offspring from two parents by combining their genomes based 
        on a random bit mask that selects a random parent for each genome.
        """
        pass
 
    def get_offspring(self, parent_genomes):
        genome_size = len(parent_genomes[0])
        bitmask = np.random.randint(0,2,genome_size).astype(bool)
        new_genome = np.zeros(genome_size)
        new_genome[bitmask] = parent_genomes[0][bitmask]
        new_genome[~bitmask] = parent_genomes[0][~bitmask]
        return new_genome



class PartiallyMapped:

    def __init__(self):
        pass

    def get_offspring(self, parent_genomes):
        """
        https://www.hindawi.com/journals/cin/2017/7430125/
        """

        # Get parents
        p1, p2 = parent_genomes.tolist()
        n = len(p1)

        # Random substring points
        cut1 = np.random.randint(0, n-2)
        cut2 = np.random.randint(cut1+1, n-1)

        # Get middle sections from parents
        p1_mid = p1[cut1:cut2]
        p2_mid = p2[cut1:cut2]

        # Get mappings and list of mapped genes
        mappings = tuple(zip(p1_mid, p2_mid))
        map_genes = set(p1_mid + p2_mid)
        
        # Initialize offspring
        o1, o2 = [['x'] * n] * 2

        # Set middle sections for offspring
        o1 = o1[:cut1] + p2_mid + o1[cut2:]
        o2 = o2[:cut1] + p1_mid + o2[cut2:]

        # Fill in non-conflicting genes from parents
        o1 = self.set_non_conflicting_genes(cut1, cut2, p1, o1, map_genes)
        o2 = self.set_non_conflicting_genes(cut1, cut2, p2, o2, map_genes)

        # Use mapping to set rest of offspring
        o1 = self.set_mapped_genes(cut1, cut2, p1, o1, mappings)
        o2 = self.set_mapped_genes(cut1, cut2, p2, o2, mappings)


        # TODO return both offspring and handle pairs in GA module
        if np.random.uniform() < 0.5:
            return np.array(o1)
        else:
            return np.array(o2)

    def set_non_conflicting_genes(self, cut1, cut2, parent, child, map_genes):
        """
        Copy all genes that do not appear within the cut sections directly
        for a parent genome to a child genome.
        """
        for i in range(len(parent)):
            if i < cut1 or i >= cut2:
                if not parent[i] in map_genes:
                    child[i] = parent[i]
        return child 
        
    def set_mapped_genes(self, cut1, cut2, parent, child, mappings):
        """
        Set a gene in a child genome based on the gene mappings. If a
        retreived mapped gene already exists within the child, find the
        next mapping based on that gene. Continue until gene that doesn't
        already exist is found.
        """
        for i in range(len(parent)):
            if i < cut1 or i >= cut2:
                if child[i] == 'x' and parent[i] in child:

                    gene = parent[i]
                    temp_mappings = list(mappings)     
                    
                    while True:
                        
                        # Get mapping for gene and remove mapping pair from temp_mappings
                        gene_mapping = [pair for pair in temp_mappings if gene in pair][0]
                        temp_mappings.remove(gene_mapping)

                        # Get new gene from mapping pair
                        gene_mapping = list(gene_mapping)
                        gene_mapping.remove(gene)
                        new_gene = gene_mapping[0]
                        
                        # Test for new_gene occurence in child and apply mapping
                        if new_gene in child:
                            gene = new_gene
                            continue
                        else:
                            child[i] = new_gene
                            break
        return child




if __name__ == '__main__':

    # Creating parents (random)
    p1 = np.arange(8)
    np.random.shuffle(p1)
    p2 = np.arange(8)
    np.random.shuffle(p2)
    parent_genomes = np.stack((p1,p2))


    # Create offspring
    crossover = PartiallyMapped()
    offspring = crossover.get_offspring(parent_genomes)


    ############## debug
    o1, o2 = offspring

    print(o1)
    print(o2)
    
















    






