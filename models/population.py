import config
from models.genome import Genome
from models.gene import NodeGene, EdgeGene
from models.species import Species, DistanceCache
from models.indexer import Indexer
from models.utils import mean, median, variance, stdev, softmax
import random
import math
import sys
from pprint import pprint


class Population(object):
    ''' Clase que establece la especiación
        y reprodicción
    '''

    config = None
    champion = None
    genomes = []
    ancestors = []


    def __init__(self,config):
        #print("[population] Init population")
        self.config = config
        # Config 
        self.genomes = {}
        self.ancestors = {}
        self.species = {}
        self.species_fitness_func = max     # Variable param from the config file
        self.genome_to_species = {}
        #self.species = {}
        self.champion = None
        self.generation = 0
        self.elitism = self.config.get('elitism')
        self.pop_size = self.config.get('pop_size')
        # Species stagnation
        self.species_stagnation = self.config.get('species_stagnation')
        self.max_stagnation = self.config.get('max_stagnation')
        self.species_elitism = self.config.get('species_elitism')
        self.min_species_size = 2
        self.survival_threshold = self.config.get('survival_threshold')
        # Indexers to increment the object's key
        self.genome_indexer = Indexer(1)
        self.species_indexer = Indexer(1)
        # Create an initial population of genomes
        self.new()


    def new(self):
        ''' Create a new population of n individuals
        '''
        #print("[population] Creating new population")
        new_genomes = {}
        for i in range(0,self.pop_size):
            # Getting genome's key from the indexer
            gid = self.genome_indexer.get_next()
            # Create a genome object and append it to the genomes list of the population
            g = Genome(self.config,gid)
            #print("[population] new genome:")
            #print(g)
            #self.genomes.append(g)
            new_genomes[g.key] = g
            self.ancestors[g.key] = tuple()

        self.genomes = new_genomes
        #print("[population] Finish creating population")
        #for k,g in self.genomes.items():
        #    print(k)
        #    print(g)
        # Divide species
        self.speciate()
        #print("[population] new speciation")

        #for s in self.species.values():
            #print(s.representative)
        # Increment generation
        self.generation=1



    def set_genome_fitness(self, fitness_function):
        ''' Iterate throught the genomes of a population and set its fitness
            Set the champion genome of the population
        '''
        #print("[population] Starting population fitness evaluation")
        self.champion = None
        for key, genome in self.genomes.items():
            #print("[population] Iterating genomes")
            #print(genome)
            genome.fitness = fitness_function(genome)
            if self.champion == None or genome.fitness > self.champion.fitness:
                #print("[population] setting new population champion with fitness:" + str(genome.fitness))
                self.champion = genome
        return True



    def speciate(self):
        ''' Speciate:
            1. Tomar un representante de cada especie de la generación anterior
            2. Medir distancia de compatibilidad entre genoma g y cada representante
            3. Agregar genoma a la primera especie en donde la distancia es menor que el umbral
            4. Si g no es compatible con ninguna, crear nueva especie
            {
                "species" : {
                    "representative" : <genome>,
                    "members" : <genomes>
                }
            }
        '''
        assert type(self.genomes) is dict
        compatibility_threshold = float(self.config.get('compatibility_threshold'))

        unspeciated = set(self.genomes.keys())
        distances = DistanceCache(self.config)
        new_representatives = {}
        new_members = {}
        #print("[population] [speciate] getting distances")
        for sid, s in self.species.items():
            #print(s)
            candidates = []
            for gid in unspeciated:
                g = self.genomes[gid]
                d = distances(s.representative, g)
                #print("[speciate] distance %s - %s is: %s " % (s.representative.key,gid, d))
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)


        # Partition population into species based on genetic similarity.
        #print("[population] Unspeciated")
        #print(unspeciated)
        while unspeciated:
            gid = unspeciated.pop()
            g = self.genomes[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = self.genomes[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))
            #print("[species] candidates")
            #print(candidates)
            if candidates:
                sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                #print("[species] creating new species")
                sid = self.species_indexer.get_next()
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                #print("[population] Creating species: gen %s"% (self.generation))
                s = Species(self.config, sid, self.generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, self.genomes[gid]) for gid in members)
            #print("[population] speciation updating reps")
            self.species[sid].update(self.genomes[rid], member_dict)
            #print(self.species[sid].representative)

        #print("[speciate] genome to species: " )
        #pprint(self.genome_to_species)

        #print("[population] finish speciating")
        #pprint(self.species)

        gdmean = mean(distances.distances.values())
        gdstdev = stdev(distances.distances.values())
        print('Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))
        



    def stagnation(self):
        ''' With the speces dict (all the species) and the generation,
            return the stagnated speces
        '''
        species_data = []
        # Loop the species and get data: fitness_history, adjusted_fitness, last_improved
        for sid, s in self.species.items():
            prev_fitness = max(s.fitness_history) if s.fitness_history else -sys.float_info.max
            # Get max fitness of the members of the species
            s.fitness = self.species_fitness_func(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = self.generation
            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for sid, s in species_data:
            stagnant_time = self.generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.species_elitism:
                is_stagnant = stagnant_time >= self.max_stagnation
            if is_stagnant:
                num_non_stagnant -= 1
            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        #print("[population] stagnated? " + str(result))
    

        return result


    def reproduce(self):
        ''' Needs species, generation & pop_size
            - Stagnation update
            - Seeds computation: list of number of seeds per species
            - Elites per species
            - Crossover genome
            - Save ancestors
        '''
        #print("[population] init reproduction process")
        species = self.species
        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        all_fitnesses = []
        for sid, s in species.items():
            all_fitnesses.extend(m.fitness for m in s.genomes.values())
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(1.0, max_fitness - min_fitness)

        remaining_species = []
        for sid, s, stagnant in self.stagnation():
            if stagnant:
                print("Species stagnated, removing it...", str((sid, s)))
            else:
                # Compute adjusted fitness.
                msf = mean([m.fitness for m in s.genomes.values()])
                af = (msf - min_fitness) / fitness_range
                s.adjusted_fitness = af
                remaining_species.append(s)


        #print("[population] remaining species: "+str(remaining_species))

        # No species left.
        if not remaining_species:
            species = {}
            return []

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)

        # Compute the number of new memebers for each species in the new generation.
        previous_sizes = [len(s.genomes) for s in remaining_species]
        min_species_size = self.min_species_size
        seed_amounts = self.seeds(adjusted_fitnesses, previous_sizes, self.pop_size, self.min_species_size)

        #print("[population] seeds amount: "+str([ (seeds,s) for seeds, s in zip(seed_amounts, remaining_species )]))

        new_genomes = {}
        species = {}
        for seeds, s in zip(seed_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            seeds = max(seeds, self.elitism)

            assert seeds > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.genomes.items())
            s.genomes = {}
            species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            #print("[population] species %s, old members %s, seeds %s" % (s.key,len(old_members),seeds))
            #print([ (i,v.fitness) for i,v in old_members ])

            # Transfer elites to new generation.
            if self.elitism > 0:
                for i, m in old_members[:self.elitism]:
                    new_genomes[i] = m
                    seeds -= 1
            if seeds <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.survival_threshold * len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            #print("[population] starting crossover species "+ str(s.key))
            while seeds > 0:
                seeds -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = self.genome_indexer.get_next()   
                child = Genome(self.config, gid, init=False)
                child.crossover(parent1, parent2)
                #print("[population] parent1")
                #print(parent1)
                #print("[population] parent2")
                #print(parent2)
                #print("[population] child")
                #print(child)

                child.mutate()
                new_genomes[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        #print("[population] reproduce new genomes:")
        #for g in new_genomes.values():
        #    print(g)

        self.species = species
        self.genomes = new_genomes
        self.champion = None
        #self.generation += self.generation+1



    @staticmethod
    def seeds(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        ''' Return a list of the amount of seed genomes for every species
        '''
        af_sum = sum(adjusted_fitness)
        seed_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            seed = ps
            if abs(c) > 0:
                seed += c
            elif d > 0:
                seed += 1
            elif d < 0:
                seed -= 1
            seed_amounts.append(seed)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_seeds = sum(seed_amounts)
        norm = pop_size / total_seeds
        seed_amounts = [max(min_species_size, int(round(n * norm))) for n in seed_amounts]

        return seed_amounts




    @staticmethod
    def print_species(config, population, species, generation):
        ''' Print the population species information
        '''
        ng = len(population)
        ns = len(species)

        print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
        sids = list(species.keys())
        sids.sort()
        print("   ID   age  size  fitness  adj fit  stag")
        print("  ====  ===  ====  =======  =======  ====")
        for sid in sids:
            s = species[sid]
            a = generation - s.created
            n = len(s.genomes)
            f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
            af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
            st = generation - s.last_improved
            print("  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))


